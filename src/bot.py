import os
import numpy as np
import torch
import sys
import glob
import math

# Add root directory to path so we can import rewards.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rlbot.flat import BallAnchor, ControllerState, GamePacket, MatchPhase
from rlbot.managers import Bot
from rlgym_compat import GameState, common_values
from collections import OrderedDict

# Import Custom Rewards
from rewards import FaceBallReward, InAirReward, JumpTouchReward, SpeedTowardBallReward, TouchReward, VelocityBallToGoalReward, AdvancedTouchReward, KickoffReward, FirstTouchKickoffReward
from rlgym.rocket_league.reward_functions import CombinedReward

# Your imports here
from act import LookupTableAction
from obs import DefaultObs, AdvancedObs
from discrete import DiscreteFF

from rlgym_compat.sim_extra_info import SimExtraInfo

# --- Helper Class for Breakdown ---
class TrackingCombinedReward(CombinedReward):
    """
    A wrapper around CombinedReward that stores the most recent reward breakdown
    so we can visualize it in the overlay.
    """
    def __init__(self, *rewards_and_weights):
        super().__init__(*rewards_and_weights)
        self.last_breakdown = {} # {agent_id: {reward_name: value}}
        self.last_non_zero_values = {} # {agent_id: {reward_name: value}}
        
        # Explicitly store them to avoid AttributeError if parent doesn't expose them publicly
        # CombinedReward usually takes arguments like (Reward(), 1.0), (Reward2(), 0.5)...
        self.stored_rewards = [r for r, w in rewards_and_weights]
        self.stored_weights = [w for r, w in rewards_and_weights]

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        combined_rewards = {agent: 0.0 for agent in agents}
        self.last_breakdown = {agent: {} for agent in agents}
        
        # Init history dict for new agents
        for agent in agents:
            if agent not in self.last_non_zero_values:
                self.last_non_zero_values[agent] = {}

        # Use our stored list instead of self.rewards
        for reward_fn, weight in zip(self.stored_rewards, self.stored_weights):
            vals = reward_fn.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            for agent in agents:
                val = vals[agent]
                weighted_val = val * weight
                combined_rewards[agent] += weighted_val
                
                name = type(reward_fn).__name__
                if name in self.last_breakdown[agent]:
                    name = f"{name}_2"
                
                # Store current value
                self.last_breakdown[agent][name] = weighted_val
                
                # Update history if non-zero
                if abs(weighted_val) > 0.001:
                    self.last_non_zero_values[agent][name] = weighted_val
                # If zero, ensure it exists in history as 0.0 if not present
                elif name not in self.last_non_zero_values[agent]:
                    self.last_non_zero_values[agent][name] = 0.0
                    
        return combined_rewards

# Find latest checkpoint
def find_latest_checkpoint():
    # Use the directory where this script (bot.py) is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Look in the same folder as this script
    checkpoints = glob.glob(os.path.join(script_dir, "**", "*POLICY.*t"), recursive=True)
    
    return checkpoints[0]

model_path = find_latest_checkpoint()


def model_info_from_dict(loaded_dict):
    state_dict = OrderedDict(loaded_dict)

    bias_counts = []
    weight_counts = []
    for key, value in state_dict.items():
        if ".weight" in key:
            weight_counts.append(value.numel())
        if ".bias" in key:
            bias_counts.append(value.size(0))

    # Detect LayerNorm
    # In GigaLearn, Linear is followed by LayerNorm. 
    # Linear.bias size == LayerNorm.weight size == LayerNorm.bias size
    is_layernorm = False
    if len(weight_counts) > 1 and weight_counts[1] == bias_counts[0]:
        print("Detected LayerNorm in model structure")
        is_layernorm = True

    if is_layernorm:
        # Every pair of (Linear, LayerNorm) counts as one logical layer for our DiscreteFF
        inputs = int(weight_counts[0] / bias_counts[0])
        outputs = bias_counts[-1]
        # bias_counts: [L1, LN1, L2, LN2, ..., Lout]
        # We want [L1, L2, ...]
        layer_sizes = bias_counts[:-1:2]
    else:
        inputs = int(weight_counts[0] / bias_counts[0])
        outputs = bias_counts[-1]
        layer_sizes = bias_counts[:-1]

    return inputs, outputs, layer_sizes, is_layernorm


# --- C++ Compatible Observation Builder ---
class CppDefaultObs(DefaultObs):
    def __init__(self, pos_coef=1/2300, ang_coef=1/math.pi, lin_vel_coef=1/2300, ang_vel_coef=1/math.pi, boost_coef=1/100.0):
        super().__init__()
        self.pos_coef = pos_coef
        self.ang_coef = ang_coef
        self.lin_vel_coef = lin_vel_coef
        self.ang_vel_coef = ang_vel_coef
        self.boost_coef = boost_coef

    def build_obs(self, agents: list, state: GameState, shared_info: dict) -> dict:
        obs = {}
        for agent in agents:
            # We need the previous action. In this script, it is passed via shared_info or handled externally
            # For now, we assume shared_info['prev_actions'][agent] exists, or we default to zeros
            prev_action = shared_info.get('prev_actions', {}).get(agent, np.zeros(8))
            obs[agent] = self._build_obs(agent, state, prev_action)
        return obs

    def _build_obs(self, agent, state: GameState, prev_action: np.ndarray) -> np.ndarray:
        car = state.cars[agent]
        if car.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pad_timers
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pad_timers

        # 1. Ball (9)
        ball_obs = np.concatenate([
            ball.position * self.pos_coef,
            ball.linear_velocity * self.lin_vel_coef,
            ball.angular_velocity * self.ang_vel_coef
        ])

        # 2. Previous Action (8)
        prev_action_obs = prev_action

        # 3. Pads (34)
        # C++ DefaultOBS does not normalize pad timers, so we use raw values
        pads_obs = pads
        
        # 4. Self (19)
        car_obs = self._get_player_obs(car, inverted)

        # 5. Others (Teammates + Opponents) (19 each)
        # C++ DefaultOBS iterates all other players. 
        # In 1v1, it's just the one opponent.
        others_obs = []
        
        # Sort cars to ensure consistent order (e.g. by ID) if multiple
        other_cars = [c for id, c in state.cars.items() if id != agent]
        # In 1v1 C++ code, it just iterates. We should probably sort by ID to be safe or rely on insertion order.
        # For strict 1v1, there is only one.
        
        for other in other_cars:
             others_obs.append(self._get_player_obs(other, inverted))

        # Combine
        return np.concatenate([
            ball_obs,
            prev_action_obs,
            pads_obs,
            car_obs,
            np.concatenate(others_obs) if others_obs else np.array([])
        ])

    def _get_player_obs(self, car, inverted):
        if inverted:
            phys = car.inverted_physics
        else:
            phys = car.physics

        return np.concatenate([
            phys.position * self.pos_coef,
            phys.forward, # Rotation matrix forward
            phys.up,      # Rotation matrix up
            phys.linear_velocity * self.lin_vel_coef,
            phys.angular_velocity * self.ang_vel_coef,
            [
                car.boost_amount * self.boost_coef,
                1.0 if car.on_ground else 0.0,
                # hasFlip in C++: typically (!doubleJumped && !flipped && (onGround || !airTimeExceeded))
                # RLGym GameState usually tracks has_flip
                1.0 if car.has_flip else 0.0,
                1.0 if car.is_demoed else 0.0
            ]
        ])

class MyBot(Bot):

    def initialize(self):
        self.deterministic = True #NOTE: Set to True if you want to use deterministic actions, default is False
        self.ticks = self.tick_skip = 8 #NOTE: Set the amount of ticks to skip, default is 8
        self.device = torch.device("cpu") #NOTE: Set the device to use, default is cpu
        
        print(f"Loading model from: {model_path}")

        # Get the bot data from model
        # Handle both full learner states and raw policy states
        loaded_file = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(loaded_file, dict) and "policy_state_dict" in loaded_file:
            print("Detected full Learner state, extracting policy...")
            model_file = loaded_file["policy_state_dict"]
        else:
            model_file = loaded_file

        # Helper to extract state_dict if it's a ScriptModule (from C++ LibTorch)
        if not isinstance(model_file, dict) and hasattr(model_file, "state_dict"):
            print("Detected ScriptModule (likely from C++), extracting state_dict...")
            model_file = model_file.state_dict()

        input_amount, action_amount, layer_sizes, is_layernorm = model_info_from_dict(model_file)
        print(f"Model detected: {input_amount} inputs, {action_amount} actions, layers {layer_sizes}, LayerNorm={is_layernorm}")

        # Make the policy
        self.policy = DiscreteFF(input_amount, action_amount, layer_sizes, self.device, add_layer_norm=is_layernorm)
        
        # Load the state dict
        # If the keys start with numbers (e.g. "0.weight"), it matches the internal sequential model
        first_key = next(iter(model_file.keys()))
        if first_key[0].isdigit():
             print("Loading state dict into internal model sequence...")
             self.policy.model.load_state_dict(model_file)
        else:
             print("Loading state dict into policy...")
             self.policy.load_state_dict(model_file)

        torch.set_num_threads(1)

        #! update this with your action and obs
        action_parser = LookupTableAction()
        
        # DYNAMICALLY CHOOSE OBS BUILDER BASED ON MODEL INPUTS
        if input_amount == 109:
            print("Using GigaLearn compatible AdvancedObs builder (109 features)")
            Obs = AdvancedObs()
        elif input_amount == 89:
            print("Using C++ compatible observation builder (89 features)")
            Obs = CppDefaultObs(
                pos_coef=np.array([1/common_values.SIDE_WALL_X, 1/common_values.BACK_NET_Y, 1/common_values.CEILING_Z]),
                lin_vel_coef=1/common_values.CAR_MAX_SPEED,
                ang_vel_coef=1/common_values.CAR_MAX_ANG_VEL,
                boost_coef=1/100.0
            )
        else:
            print(f"Using standard Python observation builder ({input_amount} features)")
            # Fallback to standard DefaultObs with padding=3 (172 features)
            Obs = DefaultObs(
                zero_padding=3,
                pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 
                                     1 / common_values.BACK_NET_Y, 
                                     1 / common_values.CEILING_Z]),
                ang_coef=1 / np.pi,
                lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
                ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
                boost_coef=1 / 100.0
            )
        
        # Setup Reward Function (Match train.py but use Tracking)
        self.reward_fn = TrackingCombinedReward(
            (SpeedTowardBallReward(), 5),
            (JumpTouchReward(), 200),
            (FaceBallReward(), 0.5),
            (VelocityBallToGoalReward(), 20),
            # (GoalReward(), 1500), # Goals handled by state reset usually
            (AdvancedTouchReward(touch_reward=0.05, acceleration_reward=1, min_ball_speed=300), 75),
            (KickoffReward(), 10),
            (FirstTouchKickoffReward(), 500)
        )

        # Some variables that are going to be used
        self.game_state = GameState()
        self.prev_time = 0.0
        self.prev_control = ControllerState()
        self.controls = ControllerState()
        self.prev_actions = np.zeros(8) # Track previous action array
        self.obs = Obs
        self.action_parser = action_parser
        self.sent_more_than_one_ball_warning = False

        
        self.extra_info = SimExtraInfo(self.field_info, tick_skip=self.tick_skip)
        self.game_state = self.game_state.create_compat_game_state(self.field_info)
        
        # Enable Rendering Explicitly
        self.update_rendering_status(True)

    def get_output(self, packet: GamePacket) -> ControllerState:
        """
        This function will be called by the framework many times per second. This is where the bot makes any decisions.
        """

        # Calculate the time elapsed since the last frame
        cur_time = packet.match_info.frame_num
        ticks_elapsed = cur_time - self.prev_time
        self.prev_time = cur_time

        self.ticks += ticks_elapsed

        # Some checks
        if len(packet.balls) == 0 or packet.match_info.match_phase == MatchPhase.Ended:
            # If there are no balls current in the game (likely due to being in a replay) or game alredy ended, random movements.
            # Just use 5 random actions -1 to 1 and after that 3 random 0 or 1
            # Do a celebration :D
            return ControllerState(throttle=np.random.uniform(-1, 1), steer=np.random.uniform(-1, 1), pitch=np.random.uniform(-1, 1), yaw=np.random.uniform(-1, 1), roll=np.random.uniform(-1, 1), jump=np.random.choice([True, False]), boost=np.random.choice([True, False]), handbrake=np.random.choice([True, False]))
        if len(packet.balls) > 1 and self.sent_more_than_one_ball_warning == False:
            print("WARNING: More than one ball detected. This is unexpected and may cause issues.")
            self.sent_more_than_one_ball_warning = True

        # Get the model and use it to predict the next action using the obs and action parser
        
        extra_info = self.extra_info.get_extra_info(packet)
        self.game_state.update(packet, extra_info=extra_info)
        
        # --- REWARD CALCULATION & RENDERING ---
        # We calculate this every frame (or tick) to update the visual feedback live
        
        # Reset reward on kickoff/new round
        if packet.match_info.match_phase == MatchPhase.Kickoff and packet.match_info.frame_num % 10 == 0:
             self.reward_fn.reset([self.player_id], self.game_state, {})

        rewards = self.reward_fn.get_rewards([self.player_id], self.game_state, {self.player_id: False}, {self.player_id: False}, {})
        current_reward = rewards[self.player_id]
        
        # Render (Throttled to 30fps / every 4 frames)
        # Only render for the first bot (Index 0) to avoid overlapping text in self-play
        if packet.match_info.frame_num % 4 == 0 and self.index == 0:
            try:
                with self.renderer.context():
                    # Signature: red, green, blue, alpha
                    # Palette
                    white = self.renderer.create_color(255, 255, 255, 255)
                    lime = self.renderer.create_color(50, 255, 50, 255)    
                    red = self.renderer.create_color(255, 80, 80, 255)     
                    silver = self.renderer.create_color(220, 220, 220, 255) 
                    
                    # Draw Header
                    # Move down to avoid RLBot status overlay (y=0.25)
                    box_x, box_y = 0.02, 0.25
                    header_y = box_y + 0.01
                    self.renderer.draw_string_2d(f"Bot Reward: {current_reward:.2f}", box_x + 0.01, header_y, 1.5, white)
                    
                    # Prepare Data
                    if self.reward_fn.last_breakdown:
                        current_values = self.reward_fn.last_breakdown.get(self.player_id)
                        if not current_values:
                             current_values = self.reward_fn.last_breakdown.get(str(self.player_id), {})
                        
                        history_values = self.reward_fn.last_non_zero_values.get(self.player_id)
                        if not history_values:
                             history_values = self.reward_fn.last_non_zero_values.get(str(self.player_id), {})
        
                        all_keys = list(current_values.keys())

                        y = header_y + 0.04
                        line_height = 0.025
                        
                        for k in all_keys:
                            val = current_values.get(k, 0.0)
                            last_val = history_values.get(k, 0.0)
                            
                            is_active = abs(val) > 0.001
                            
                            if is_active:
                                color = lime if val > 0 else red
                                prefix = ">>" 
                                display_val = val
                            else:
                                color = silver
                                prefix = "  "
                                display_val = last_val
                                
                            display_name = k.replace("Reward", "")
                            
                            self.renderer.draw_string_2d(f"{prefix} {display_name}: {display_val:.2f}", box_x + 0.01, y, 1, color)
                            y += line_height

            except Exception as e:
                print(f"Render Error: {e}")
        # --------------------------------------
        
        if self.ticks >= self.tick_skip - 1:
            self.ticks = 0

            # Get the car ids
            cars_ids = self.game_state.cars.keys()
			
            # Build the obs with the ids
            # PASS PREVIOUS ACTIONS
            shared_info = {'prev_actions': {self.player_id: self.prev_actions}}
            obs = self.obs.build_obs(cars_ids, self.game_state, shared_info=shared_info)

            # Get the obs of the current car
            obs = obs.get(self.player_id)
            obs = np.asarray(obs).flatten()
            obs_tensor = torch.tensor(np.array(obs, dtype=np.float32), dtype=torch.float32, device=self.device)


            # Get the prediction
            with torch.no_grad():
                action_idx, probs = self.policy.get_action(obs_tensor, deterministic=self.deterministic)
                #print(f"Action idx: {action_idx}")

                if self.deterministic == True:
                    # If it's true, we should place it inside a tensor
                    action_idx = torch.tensor([action_idx], device=self.device)

            # Based on the action, parse it into an array of 8 values
            parsed_actions = self.action_parser.parse_actions(actions={self.player_id: action_idx}, state=self.game_state, shared_info={}).get(self.player_id)
            
            # Store parsed actions for next frame obs
            self.prev_actions = parsed_actions

            # Check for errors
            if len(parsed_actions.shape) == 2:
                if parsed_actions.shape[0] == 1:
                    parsed_actions = parsed_actions[0]
		
            if len(parsed_actions.shape) != 1:
                raise Exception("Invalid action:", parsed_actions)

        else:
            # Still tick skip is not reached, return the previous control
            return self.prev_control

        # Update the controls, but if not able to, just return the previous control
        try:
            self.update_controls(parsed_actions)
        except:
            return self.prev_control
        
        # Save the previous control
        self.prev_control = self.controls

        return self.controls
    

    def update_controls(self, action):
        """
        Based on the action, update the controls
        """
        actions = []
        for actio in action:
            actions.append(float(actio))
        action = actions
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0


if __name__ == "__main__":
    # Connect to RLBot and run
    # Having the agent id here allows for easier development,
    # as otherwise the RLBOT_AGENT_ID environment variable must be set.
    MyBot("Matt/MattBot/v0").run()
