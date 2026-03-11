import os
import numpy as np
import torch
import sys
import glob
import math
import datetime

# Add root directory to path so we can import rewards.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rlbot.flat import BallAnchor, ControllerState, GamePacket, MatchPhase
from rlbot.managers import Bot
from rlgym_compat import GameState, common_values
from collections import OrderedDict

# Your imports here
from act import LookupTableAction
from obs import DefaultObs, AdvancedObs
from discrete import DiscreteFF

from rlgym_compat.sim_extra_info import SimExtraInfo

# Find latest checkpoint
def find_latest_checkpoint():
    # Use the directory where this script (bot.py) is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Look in the same folder as this script
    checkpoints = glob.glob(os.path.join(script_dir, "**", "POLICY.lt"), recursive=True)
    if not checkpoints:
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
    is_layernorm = False
    if len(weight_counts) > 1 and weight_counts[1] == bias_counts[0]:
        is_layernorm = True

    if is_layernorm:
        # Every pair of (Linear, LayerNorm) counts as one logical layer
        # If the number of bias tensors is odd, the last one is the output layer (no LN)
        if len(bias_counts) % 2 == 1:
            layer_sizes = bias_counts[:-1:2]
            outputs = bias_counts[-1]
        else:
            # No separate output layer (typical for Shared Heads)
            layer_sizes = bias_counts[::2]
            outputs = bias_counts[-1]
    else:
        # No LayerNorm. 
        # Heuristic: If last layer is significantly smaller, it's probably the output layer
        if len(bias_counts) > 1 and bias_counts[-1] < bias_counts[-2] and bias_counts[-1] < 100:
            layer_sizes = bias_counts[:-1]
            outputs = bias_counts[-1]
        else:
            layer_sizes = bias_counts
            outputs = bias_counts[-1]

    inputs = int(weight_counts[0] / bias_counts[0])

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
        # --- Match Log Setup ---
        self.match_log_path = None
        if self.index == 0:
            try:
                # Store logs in a shared 'logs' folder at the root of the project
                logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
                if not os.path.exists(logs_dir):
                    os.makedirs(logs_dir)
                
                # Create a unique filename for this specific match session
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.match_log_path = os.path.join(logs_dir, f"match_log_{timestamp}.txt")
                
                with open(self.match_log_path, "w") as f:
                    f.write(f"--- Match Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            except Exception as e:
                print(f"Error initializing match log: {e}")
        # -----------------------

        self.deterministic = True #NOTE: Set to True if you want to use deterministic actions, default is False
        self.tick_skip = 8 #NOTE: Set the amount of ticks to skip, default is 8
        self.ticks = self.tick_skip
        self.device = torch.device("cpu") #NOTE: Set the device to use, default is cpu
        
        print(f"Loading model from: {model_path}")

        # Get the bot data from model
        # GigaLearn separates Shared Head and Policy into two files
        checkpoint_dir = os.path.dirname(model_path)
        shared_path = os.path.join(checkpoint_dir, "SHARED_HEAD.lt")
        policy_path = model_path

        model_file = OrderedDict()
        layer_offset = 0

        if os.path.exists(shared_path):
            print(f"Loading Shared Head from {shared_path}...")
            shared_dict = torch.load(shared_path, map_location=self.device, weights_only=False)
            if not isinstance(shared_dict, dict) and hasattr(shared_dict, "state_dict"):
                 shared_dict = shared_dict.state_dict()
            
            # Dynamically determine the correct offset based on modules (including activation)
            _, _, shared_layer_sizes, shared_is_ln = model_info_from_dict(shared_dict)
            stride = 3 if shared_is_ln else 2
            layer_offset = len(shared_layer_sizes) * stride
            
            for k, v in shared_dict.items():
                model_file[k] = v
            print(f"Shared head has {len(shared_layer_sizes)} layers, offset set to {layer_offset}")

        print(f"Loading Policy from {policy_path}...")
        policy_dict = torch.load(policy_path, map_location=self.device, weights_only=False)
        if not isinstance(policy_dict, dict) and hasattr(policy_dict, "state_dict"):
            policy_dict = policy_dict.state_dict()

        # Handle full learner states
        if "policy_state_dict" in policy_dict:
            policy_dict = policy_dict["policy_state_dict"]

        for k, v in policy_dict.items():
            parts = k.split('.')
            if parts[0].isdigit():
                new_key = f"{int(parts[0]) + layer_offset}.{'.'.join(parts[1:])}"
                model_file[new_key] = v
            else:
                model_file[k] = v

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
        
        # Some variables that are going to be used
        self.game_state = GameState()
        self.prev_time = 0.0
        self.prev_control = ControllerState()
        self.controls = ControllerState()
        self.next_action = np.zeros(8)
        self.obs = Obs
        self.action_parser = action_parser
        self.sent_more_than_one_ball_warning = False

        
        self.extra_info = SimExtraInfo(self.field_info, tick_skip=self.tick_skip)
        self.game_state = self.game_state.create_compat_game_state(self.field_info)
        


    def get_output(self, packet: GamePacket) -> ControllerState:
        """
        This function will be called by the framework many times per second. This is where the bot makes any decisions.
        """
        # --- Score Logging ---
        # Use frame_num / 120 to get actual match time (ignoring replays/pauses)
        current_match_time = packet.match_info.frame_num / 120.0
        if not hasattr(self, 'last_log_time'):
            self.last_log_time = -60.0
            
        if current_match_time - self.last_log_time >= 60.0:
            self.last_log_time = current_match_time
            if self.match_log_path:
                try:
                    score_blue = packet.teams[0].score
                    score_orange = packet.teams[1].score
                    
                    # Get names from players in the packet
                    name_blue = "Blue"
                    name_orange = "Orange"
                    for i in range(len(packet.players)):
                        p = packet.players[i]
                        if p.team == 0: name_blue = p.name
                        if p.team == 1: name_orange = p.name

                    with open(self.match_log_path, "a") as f:
                        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"[{now}] Match Time: {int(current_match_time // 60)}m {int(current_match_time % 60)}s | {name_blue}: {score_blue} - {name_orange}: {score_orange}\n")
                except Exception as e:
                    print(f"Error logging score: {e}")
        # ---------------------

        # Calculate the time elapsed since the last frame
        cur_time = packet.match_info.frame_num
        ticks_elapsed = cur_time - self.prev_time
        if self.prev_time == 0.0:
            ticks_elapsed = 1
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
        
        
        if self.ticks >= self.tick_skip:
            # We use modulo instead of -= so that if ticks spikes to 60 due to lag, 
            # we drop the missed ticks and cleanly sync back to the 8-tick cadence,
            # avoiding a PyTorch "catch-up" death loop.
            self.ticks %= self.tick_skip

            # Get the car ids
            cars_ids = self.game_state.cars.keys()
			
            # Build the obs with the ids
            # PASS PREVIOUS ACTIONS (in C++ this is the action chosen at the previous step)
            shared_info = {'prev_actions': {self.player_id: self.next_action}}
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
            
            # Store parsed action
            self.next_action = parsed_actions

            # Check for errors
            if len(self.next_action.shape) == 2:
                if self.next_action.shape[0] == 1:
                    self.next_action = self.next_action[0]
		
            if len(self.next_action.shape) != 1:
                raise Exception("Invalid action:", self.next_action)

        # No action delay — apply the latest action immediately
        action_to_apply = self.next_action

        # Update the controls, but if not able to, just return the previous control
        try:
            self.update_controls(action_to_apply)
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
