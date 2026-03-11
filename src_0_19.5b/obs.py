import math
from typing import List, Dict, Any, Tuple

import numpy as np

from rlgym.api import ObsBuilder, AgentID
#from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym_compat.game_state import GameState
from rlgym_compat.physics_object import PhysicsObject
from rlgym_compat.car import Car

#from rlgym.rocket_league.common_values import ORANGE_TEAM
from rlgym_compat.common_values import ORANGE_TEAM
from rlgym_tools.rocket_league.math.relative import relative_physics, dodge_relative_physics


class DefaultObs(ObsBuilder[AgentID, np.ndarray, GameState, Tuple[str, int]]):
    """
    The default observation builder.
    """

    def __init__(self, zero_padding=3, pos_coef=1/2300, ang_coef=1/math.pi, lin_vel_coef=1/2300, ang_vel_coef=1/math.pi,
                 pad_timer_coef=1/10, boost_coef=1/100):
        """
        :param zero_padding: Number of max cars per team, if not None the obs will be zero padded
        :param pos_coef: Position normalization coefficient
        :param ang_coef: Rotation angle normalization coefficient
        :param lin_vel_coef: Linear velocity normalization coefficient
        :param ang_vel_coef: Angular velocity normalization coefficient
        :param pad_timer_coef: Boost pad timers normalization coefficient
        """
        super().__init__()
        self.POS_COEF = pos_coef
        self.ANG_COEF = ang_coef
        self.LIN_VEL_COEF = lin_vel_coef
        self.ANG_VEL_COEF = ang_vel_coef
        self.PAD_TIMER_COEF = pad_timer_coef
        self.BOOST_COEF = boost_coef
        self.zero_padding = zero_padding
        self._state = None

    def get_obs_space(self, agent: AgentID) -> Tuple[str, int]:
        if self.zero_padding is not None:
            return 'real', 52 + 20 * self.zero_padding * 2
        elif self._state is not None:
            return 'real', 52 + 20 * len(self._state.cars)
        else:
            return 'real', -1 # Without zero padding this depends on the current state, but we don't have one yet

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self._state = initial_state

    def build_obs(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        self._state = state
        obs = {}
        for agent in agents:
            obs[agent] = self._build_obs(agent, state, shared_info)

        return obs

    def _build_obs(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]) -> np.ndarray:
        car = state.cars[agent]
        if car.team_num == ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pad_timers
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pad_timers

        obs = [  # Global stuff
            ball.position * self.POS_COEF,
            ball.linear_velocity * self.LIN_VEL_COEF,
            ball.angular_velocity * self.ANG_VEL_COEF,
            pads * self.PAD_TIMER_COEF,
            [  # Partially observable variables
                car.is_holding_jump,
                car.handbrake,
                car.has_jumped,
                car.is_jumping,
                car.has_flipped,
                car.is_flipping,
                car.has_double_jumped,
                car.can_flip,
                car.air_time_since_jump
            ]
        ]

        car_obs = self._generate_car_obs(car, inverted)
        obs.append(car_obs)

        allies = []
        enemies = []

        for other, other_car in state.cars.items():
            if other == agent:
                continue

            if other_car.team_num == car.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            team_obs.append(self._generate_car_obs(other_car, inverted))

        if self.zero_padding is not None:
            # Padding for multi game mode
            while len(allies) < self.zero_padding - 1:
                allies.append(np.zeros_like(car_obs))
            while len(enemies) < self.zero_padding:
                enemies.append(np.zeros_like(car_obs))

        obs.extend(allies)
        obs.extend(enemies)
        return np.concatenate(obs)

    def _generate_car_obs(self, car: Car, inverted: bool) -> np.ndarray:
        if inverted:
            physics = car.inverted_physics
        else:
            physics = car.physics

        return np.concatenate([
            physics.position * self.POS_COEF,
            physics.forward,
            physics.up,
            physics.linear_velocity * self.LIN_VEL_COEF,
            physics.angular_velocity * self.ANG_VEL_COEF,
            [car.boost_amount * self.BOOST_COEF,
             car.demo_respawn_timer,
             int(car.on_ground),
             int(car.is_boosting),
             int(car.is_supersonic)]
        ])

class AdvancedObs(ObsBuilder[AgentID, np.ndarray, GameState, Tuple[str, int]]):
    """
    Observation builder matching GigaLearn's AdvancedObs.
    """
    POS_COEF = 1 / 5000.0
    VEL_COEF = 1 / 2300.0
    ANG_VEL_COEF = 1 / 3.0

    def __init__(self):
        super().__init__()
        self._state = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self._state = initial_state

    def build_obs(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        self._state = state
        obs = {}
        for agent in agents:
            obs[agent] = self._build_obs(agent, state, shared_info)
        return obs

    def _build_obs(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]) -> np.ndarray:
        car = state.cars[agent]
        if car.team_num == ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pad_timers
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pad_timers

        # 1. Ball (9)
        obs = [
            ball.position * self.POS_COEF,
            ball.linear_velocity * self.VEL_COEF,
            ball.angular_velocity * self.ANG_VEL_COEF
        ]

        # 2. Previous Action (8)
        prev_action = shared_info.get('prev_actions', {}).get(agent, np.zeros(8))
        obs.append(prev_action)

        # 3. Boost Pads (34)
        # GigaLearn logic: obs += 1.f if pads[i] else 1.f / (1.f + padTimers[i])
        pads_obs = np.where(pads <= 0, 1.0, 1.0 / (1.0 + pads))
        obs.append(pads_obs)

        # 4. Self (29)
        obs.append(self._add_player_to_obs(car, inverted, ball))

        # 5. Others (Teammates + Opponents) (29 each)
        teammates = []
        opponents = []
        for other_id, other_car in state.cars.items():
            if other_id == agent:
                continue
            
            player_obs = self._add_player_to_obs(other_car, inverted, ball)
            if other_car.team_num == car.team_num:
                teammates.append(player_obs)
            else:
                opponents.append(player_obs)
        
        obs.extend(teammates)
        obs.extend(opponents)

        return np.concatenate(obs)

    def _add_player_to_obs(self, car: Car, inverted: bool, ball: PhysicsObject) -> np.ndarray:
        phys = car.inverted_physics if inverted else car.physics
        
        # Rotation matrix [forward, right, up]
        rot_mat = np.array([phys.forward, phys.right, phys.up]) # 3x3
        
        local_ang_vel = np.dot(rot_mat, phys.angular_velocity)
        local_ball_pos = np.dot(rot_mat, ball.position - phys.position)
        local_ball_vel = np.dot(rot_mat, ball.linear_velocity - phys.linear_velocity)

        return np.concatenate([
            phys.position * self.POS_COEF,
            phys.forward,
            phys.up,
            phys.linear_velocity * self.VEL_COEF,
            phys.angular_velocity * self.ANG_VEL_COEF,
            local_ang_vel * self.ANG_VEL_COEF,
            local_ball_pos * self.POS_COEF,
            local_ball_vel * self.VEL_COEF,
            [
                car.boost_amount / 100.0,
                float(car.on_ground),
                float(car.has_flip),
                float(car.is_demoed),
                float(car.has_jumped)
            ]
        ])