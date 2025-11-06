# environment2.py
"""
MyEnv2: Gymnasium environment for the Leo rover on procedurally generated Mars-like terrain.

Key features in this version:
 - Automatically chooses a robust initial waypoint index after spawn (no hard-coded index).
 - Keeps LQR baseline + DRL residual architecture.
 - Observation expanded to include baseline/residual info and slip magnitude.
 - Debug printing controlled by self._debug.
 - Safe clipping of waypoint indices to avoid out-of-range accesses.
 - Clear docstrings and standardized comments for maintainability.
"""

import time
import math
import random
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium.spaces import Box

# Local imports (package layout expects these to be importable)
from leoroverpybullet.path_generation import QuinticPolynomial2dPlanner, display_path
from mars_terrain import create_terrain, add_rocks, get_height_at
from leoroverpybullet.lqr_baseline import LQRBaseline

p.setAdditionalSearchPath(pybullet_data.getDataPath())
import leoroverpybullet  # ensures package resources are available


class MyEnv2(gym.Env):
    """
    Gym environment compatible with SB3 PPO. The agent outputs residual corrections
    which are combined with a baseline LQR controller inside Controller2.

    Observation layout (16 dims):
      [pos_x, pos_y, pos_z,
       roll, pitch, yaw,
       waypoint_x, waypoint_y, waypoint_z, waypoint_yaw,
       forward_speed,
       baseline_velocity_command, baseline_omega_command,
       last_action_residual_velocity, last_action_residual_omega,
       slip_magnitude]
    """

    def __init__(self, display: bool = False, debug: bool = False):
        super().__init__()

        # Flags
        self._display = bool(display)
        self._debug = bool(debug)

        # Robot & path
        self._robot_id = None
        self._base_link_name = b"base_link"
        self._distance_threshold = 0.2

        # Controller + baseline
        self._controller = Controller2(debug=self._debug)
        self._simulation_time_limit = 10.0  # planner horizon T

        # -----------------------------
        # Random goal within forward half of 10x10 area (x positive so path is ahead)
        # ensure minimum distance from origin so first waypoint isn't too close
        # -----------------------------

        bounds = 10.0  # +/- bounds in y, gx sampled in [min_goal_dist, bounds]
        min_goal_dist = 2.0  # ensure first waypoint at least 2 m ahead (positive X)


        gx = random.uniform(min_goal_dist, bounds)  # FORCE gx in front (positive)
        gy = random.uniform(-bounds, bounds)


        # store goal (z = 0.0)
        self._goal_pos = np.array([gx, gy, 0.0], dtype=np.float32)

        # make goal yaw roughly point toward the goal from origin, plus small random perturbation
        base_gyaw = math.atan2(gy - 0.0, gx - 0.0)
        gyaw_perturb = random.uniform(-math.pi / 3.0, math.pi / 3.0)  # up to ±22.5° randomness
        gyaw = base_gyaw + gyaw_perturb

        # start conditions: origin facing +X and small start speed so yaw is well-defined
        sx, sy, syaw = 0.0, 0.0, 0.0
        sv = 0.05  # small positive start speed (m/s)
        sa = 0.0  # start accel

        # goal velocity/accel (tune if desired)
        gv = 0.12
        ga = 0.0

        # other metadata
        max_accel = 1.0
        max_jerk = 0.5
        dt = 0.1
        T = self._simulation_time_limit

        # create the quintic planner from origin to random goal
        self._curve = QuinticPolynomial2dPlanner(
            sx, sy, syaw, sv, sa,
            float(gx), float(gy), float(gyaw),
            gv, ga, max_accel, max_jerk, dt, T
        )

        # Placeholder for reward points (populated at reset)
        self._reward_points = None
        self._current_reward_point_index = 0
        self._previous_reward_point_index = 0

        # Action space: agent outputs residuals for [velocity, omega]
        self.action_space = Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        # Observation space
        self.observation_dim = 16
        low = np.array([-100.0] * 3 + [-np.pi] * 3 + [-100.0] * 4 + [-10.0] * 1 + [-5.0] * 2 + [-1.0] * 2 + [0.0])
        high = np.array([100.0] * 3 + [np.pi] * 3 + [100.0] * 4 + [10.0] * 1 + [5.0] * 2 + [1.0] * 2 + [100.0])
        self.observation_space = Box(low=low.astype(np.float32),
                                     high=high.astype(np.float32),
                                     shape=(self.observation_dim,),
                                     dtype=np.float32)

        # Internal state for logging / reward computation
        self._previous_distance = 1e20
        self._start_time = time.time()
        self._data_point = None
        self._path_debug_item = None
        self._heightfield_data = None

        # Last-known controller/action values (for observation)
        self._last_agent_action = np.array([0.0, 0.0], dtype=np.float32)
        self._last_baseline_command = np.array([0.0, 0.0], dtype=np.float32)
        self._last_total_command = np.array([0.0, 0.0], dtype=np.float32)
        self._last_slip_magnitude = 0.0

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def get_time(self):
        """Return elapsed time since environment reset (seconds)."""
        return time.time() - self._start_time

    # -------------------------------------------------------------------------
    # Gym API: reset
    # -------------------------------------------------------------------------
    def reset(self, **kwargs):
        """
        Reset the PyBullet simulation, create terrain, spawn robot, compute
        waypoints projected onto terrain heights, choose a robust initial waypoint,
        and return the initial observation.

        This version also creates a new random path every episode (recreates self._curve).
        It honors an optional seed kwarg passed by Gym for reproducibility.
        """
        # -------------------------
        # RNG / seeding: honor gym seed if provided, otherwise reseed from OS entropy
        # -------------------------
        seed = kwargs.get("seed", None)
        if seed is None:
            # Use system entropy to get different randomness each episode
            random.seed(None)
            np.random.seed(None)
        else:
            random.seed(seed)
            np.random.seed(seed)

        # Reset simulation state
        p.resetSimulation()

        # Create terrain (heightfield)
        self._walls, self._heightfield_data = create_terrain()
        # optional: self._rocks = add_rocks(self._heightfield_data)

        # -------------------------
        # Create a new random goal and rebuild the quintic planner here (per-episode)
        # Ensures the path is different each reset unless a seed is provided.
        # Constraints:
        #   - goal is inside +/-bounds area
        #   - goal X is in front (positive) so first waypoint is ahead of rover
        #   - ensure goal at least min_goal_dist away
        # -------------------------
        bounds = 10.0  # +/- bounds in y, gx sampled in [min_goal_dist, bounds]
        min_goal_dist = 2.0  # ensure first waypoint at least 2 m ahead
        # Force goal to be in front of rover (positive X) so path starts ahead
        gx = random.uniform(min_goal_dist, bounds)
        gy = random.uniform(-bounds, bounds)

        # store goal (z = 0.0)
        self._goal_pos = np.array([gx, gy, 0.0], dtype=np.float32)

        # goal yaw: generally point toward goal from origin, plus a small perturbation for curvature
        base_gyaw = math.atan2(gy - 0.0, gx - 0.0)
        gyaw_perturb = random.uniform(-math.pi / 3.0, math.pi / 3.0)  # ±22.5°
        gyaw = base_gyaw + gyaw_perturb

        # start conditions: origin facing +X and small start speed so yaw is well-defined
        sx, sy, syaw = 0.0, 0.0, 0.0
        sv = 0.05  # small start speed (m/s)
        sa = 0.0

        # goal velocity/accel (tune if desired)
        gv = 0.12
        ga = 0.0

        # planner metadata
        max_accel = 1.0
        max_jerk = 0.5
        dt = 0.1
        T = self._simulation_time_limit

        # Recreate the planner for this episode:
        try:
            self._curve = QuinticPolynomial2dPlanner(
                sx, sy, syaw, sv, sa,
                float(gx), float(gy), float(gyaw),
                gv, ga, max_accel, max_jerk, dt, T
            )
        except Exception as e:
            # fallback: simple straight-line planner if quintic construction fails
            if self._debug:
                print("[reset] QuinticPolynomial2dPlanner construction failed:", e)

            # Construct a trivial straight-line fallback by overriding the planner's get_waypoints_rewards
            class _StraightFallback:
                def __init__(self, sx, sy, gx, gy, dt, T):
                    self.sx, self.sy, self.gx, self.gy, self.dt, self.T = sx, sy, gx, gy, dt, T

                def get_waypoints_rewards(self):
                    n = max(3, int(self.T / self.dt))
                    xs = list(np.linspace(self.sx, self.gx, n))
                    ys = list(np.linspace(self.sy, self.gy, n))
                    yaws = [math.atan2(ys[i + 1] - ys[i], xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
                    yaws.append(yaws[-1] if yaws else 0.0)
                    return list(np.linspace(0.0, self.T, len(xs))), xs, ys, yaws, [], [], []

            self._curve = _StraightFallback(sx, sy, gx, gy, dt, T)

        # -------------------------
        # Project planner waypoints onto terrain heights
        # -------------------------
        _, x_list, y_list, yaw_list, _, _, _ = self._curve.get_waypoints_rewards()
        z_list = [get_height_at(xi, yi, self._heightfield_data) for xi, yi in zip(x_list, y_list)]
        self._reward_points = np.array([x_list, y_list, z_list, yaw_list], dtype=np.float32).T

        # Validate waypoints
        if not np.all(np.isfinite(self._reward_points)):
            raise ValueError("Invalid reward points after projection: contains non-finite values")

        if self._debug:
            print("[MyEnv2.reset] New random goal:", (gx, gy, gyaw))
            print("[MyEnv2.reset] Computed reward_points (first 8 shown):")
            for i, wp in enumerate(self._reward_points[:8]):
                print(f"  {i:02d}: x={wp[0]:.3f}, y={wp[1]:.3f}, z={wp[2]:.3f}, yaw={wp[3]:.3f}")
            print(f"[MyEnv2.reset] Total waypoints: {len(self._reward_points)}")

        # Optional visualization: small spheres at waypoints
        if self._display:
            for pt in self._reward_points:
                sphere_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1])
                p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_vis, basePosition=[pt[0], pt[1], pt[2]])

        # Spawn robot at origin height + small offset
        z0 = get_height_at(0.0, 0.0, self._heightfield_data) + 0.5
        self._robot_id = p.loadURDF("leo_robot_1_ros2_shared.urdf",
                                    basePosition=[0.0, 0.0, z0],
                                    useFixedBase=False)

        # Configure controller with robot id
        self._controller._set_robot_id(self._robot_id)
        self._controller._step_count = 0

        # -------------------------
        # Select the initial reward index: choose the first waypoint at least min_first_waypoint_dist away
        # This prevents the first waypoint being too close and avoids starting mid-path unexpectedly.
        # -------------------------
        min_first_waypoint_dist = 2.0
        found_idx = None
        for j in range(len(self._reward_points)):
            if math.hypot(self._reward_points[j][0], self._reward_points[j][1]) >= min_first_waypoint_dist:
                found_idx = j
                break

        if found_idx is None:
            # if nothing found, fall back to index 0
            found_idx = 0

        self._current_reward_point_index = int(found_idx)
        self._previous_reward_point_index = 0

        if self._debug:
            print(f"[MyEnv2.reset] Starting at reward index {self._current_reward_point_index}")

        # Gravity and sim options
        p.setGravity(0, 0, -9.71)

        # Visualize current target and path
        current_reward_point = self._reward_points[self._current_reward_point_index]
        if self._data_point is not None:
            try:
                p.removeUserDebugItem(self._data_point)
            except Exception:
                pass
            self._data_point = None

        self._data_point = p.addUserDebugLine(
            current_reward_point[:3],
            [current_reward_point[0] + 0.1, current_reward_point[1] + 0.1, current_reward_point[2]],
            lineColorRGB=[0, 1, 1],
            lineWidth=2.0,
            lifeTime=0
        )

        if self._path_debug_item is not None:
            try:
                p.removeUserDebugItem(self._path_debug_item)
            except Exception:
                pass
            self._path_debug_item = None

        # draw full path
        self._path_debug_item = display_path(*self._curve.get_waypoints(), self._heightfield_data)

        # Build initial observation
        observation = self._build_observation()

        # Reset timers and bookkeeping
        self._start_time = time.time()
        self._previous_distance = np.linalg.norm(
            np.array(p.getBasePositionAndOrientation(self._robot_id)[0])[:3] -
            self._reward_points[self._current_reward_point_index][:3]
        )

        # Reset last-known values
        self._last_agent_action[:] = 0.0
        self._last_baseline_command[:] = 0.0
        self._last_total_command[:] = 0.0
        self._last_slip_magnitude = 0.0

        return observation, {}

    # -------------------------------------------------------------------------
    # Gym API: step
    # -------------------------------------------------------------------------
    def step(self, action):
        """
        Accept agent action (residuals), combine with baseline via Controller2.forward,
        apply commands to robot, compute observation & reward, and check termination.
        """
        # Validate action size and store
        action = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        self._last_agent_action = action.copy()

        # Ensure index is in-range before reading reward points
        if self._reward_points is None or len(self._reward_points) == 0:
            raise RuntimeError("Reward waypoints not initialized. Call reset() before step().")

        self._current_reward_point_index = int(np.clip(self._current_reward_point_index, 0, len(self._reward_points) - 1))
        current_waypoint = self._reward_points[self._current_reward_point_index]  # [x,y,z,yaw]

        # Controller computes baseline and applies combined control (baseline + residual)
        velocity_command_total, omega_command_total, baseline_commands, residual_commands = \
            self._controller.forward(action, current_waypoint)

        # Save for observation & reward calculations
        self._last_baseline_command = np.array(baseline_commands, dtype=np.float32)
        self._last_total_command = np.array([velocity_command_total, omega_command_total], dtype=np.float32)

        # Observation
        observation = self._build_observation()

        # Reward
        reward = self._compute_reward(baseline_commands, residual_commands)

        # Termination checks
        done = self._is_done()
        truncated = False

        info = {
            "baseline_velocity_command": float(baseline_commands[0]),
            "baseline_omega_command": float(baseline_commands[1]),
            "velocity_command_total": float(velocity_command_total),
            "omega_command_total": float(omega_command_total),
            "slip_magnitude": float(self._last_slip_magnitude),
            "current_waypoint_index": int(self._current_reward_point_index)
        }

        if self._debug:
            print("=== Step debug ===")
            print("Observation:", observation)
            print("Agent action (residual):", action)
            print("Baseline commands (velocity, omega):", baseline_commands)
            print("Residual added:", residual_commands)
            print("Final commands (velocity, omega):", (velocity_command_total, omega_command_total))
            print("Reward:", reward, "Done:", done, "Truncated:", truncated)
            print("==================")

        return observation, float(reward), bool(done), bool(truncated), info

    # -------------------------------------------------------------------------
    # Observation builder
    # -------------------------------------------------------------------------
    def _build_observation(self):
        """
        Compose a 16D observation vector for the policy.
        """
        position, orientation = p.getBasePositionAndOrientation(self._robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)

        # Ensure index safe
        self._current_reward_point_index = int(np.clip(self._current_reward_point_index, 0, len(self._reward_points) - 1))
        current_waypoint = self._reward_points[self._current_reward_point_index]  # [x,y,z,yaw]

        # approximate forward speed by projecting world linear velocity into robot body x-axis
        linear_velocity, angular_velocity = p.getBaseVelocity(self._robot_id)
        cos_y = np.cos(-yaw)
        sin_y = np.sin(-yaw)
        vx_body = cos_y * linear_velocity[0] - sin_y * linear_velocity[1]
        forward_speed = float(vx_body)

        baseline_velocity_cmd = float(self._last_baseline_command[0])
        baseline_omega_cmd = float(self._last_baseline_command[1])
        last_action_residual_velocity = float(self._last_agent_action[0])
        last_action_residual_omega = float(self._last_agent_action[1])
        slip_magnitude = float(self._last_slip_magnitude)

        obs = np.array([
            float(position[0]), float(position[1]), float(position[2]),
            float(roll), float(pitch), float(yaw),
            float(current_waypoint[0]), float(current_waypoint[1]), float(current_waypoint[2]), float(current_waypoint[3]),
            forward_speed,
            baseline_velocity_cmd, baseline_omega_cmd,
            last_action_residual_velocity, last_action_residual_omega,
            slip_magnitude
        ], dtype=np.float32)

        if not np.all(np.isfinite(obs)):
            raise ValueError(f"Invalid observation (contains non-finite values): {obs}")

        return obs

    # -------------------------------------------------------------------------
    # Reward computation
    # -------------------------------------------------------------------------
    def _compute_reward(self, baseline_commands, residual_commands):
        """
        Reward is a combination of:
          - progress reward (distance to current waypoint decreased)
          - heading alignment reward
          - path proximity penalty
          - waypoint transition bonus
          - slip penalty (commanded vs actual forward speed)
          - residual magnitude penalty (discourage large corrections)
        """
        position, orientation = p.getBasePositionAndOrientation(self._robot_id)
        current_position = np.array(position, dtype=np.float32)
        yaw = p.getEulerFromQuaternion(orientation)[2]

        # Ensure safe indexing
        self._current_reward_point_index = int(np.clip(self._current_reward_point_index, 0, len(self._reward_points) - 1))
        target_x, target_y, target_z, target_yaw = self._reward_points[self._current_reward_point_index]

        dx = current_position[0] - target_x
        dy = current_position[1] - target_y
        dz = current_position[2] - target_z
        euclidean_distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # Progress reward (positive when distance decreases)
        progress_reward = 0.5 * (self._previous_distance - euclidean_distance)
        self._previous_distance = euclidean_distance

        # Yaw alignment reward
        heading_error = abs(((yaw - target_yaw + np.pi) % (2 * np.pi)) - np.pi)
        yaw_reward = 0.15 * np.cos(heading_error)

        # Path proximity penalty (XY)
        path_xy_error = np.hypot(dx, dy)
        proximity_penalty = -0.2 * path_xy_error

        # Waypoint transition & bonus
        waypoint_transition_bonus = 0.0
        if np.hypot(dx, dy) < 0.2 and abs(dz) <= 3.0:
            self._current_reward_point_index += 1
            # clamp to last index
            self._current_reward_point_index = int(min(self._current_reward_point_index, len(self._reward_points) - 1))
            waypoint_transition_bonus = 50.0
            if self._current_reward_point_index == len(self._reward_points) - 1:
                waypoint_transition_bonus = 200.0

        # Slip magnitude: positive difference between commanded forward and actual forward speed
        linear_velocity, _ = p.getBaseVelocity(self._robot_id)
        cos_y = np.cos(-yaw)
        sin_y = np.sin(-yaw)
        vx_body = cos_y * linear_velocity[0] - sin_y * linear_velocity[1]
        actual_forward_speed = float(vx_body)

        commanded_forward_speed = float(baseline_commands[0] + residual_commands[0])  # baseline + residual
        slip_raw = float(commanded_forward_speed - actual_forward_speed)
        slip_magnitude = float(max(0.0, slip_raw))  # only penalize positive slip
        self._last_slip_magnitude = slip_magnitude

        slip_penalty = -1.5 * slip_magnitude

        # Residual penalty
        resid_vec = np.array(residual_commands, dtype=np.float32)
        residual_penalty = -0.5 * float(np.linalg.norm(resid_vec) ** 2)

        # Distance penalty
        distance_penalty = -0.01 * euclidean_distance

        total_reward = (progress_reward + yaw_reward + proximity_penalty +
                        waypoint_transition_bonus + slip_penalty + residual_penalty + distance_penalty)

        # Numerical safety
        if not np.isfinite(total_reward):
            total_reward = -100.0

        return float(total_reward)

    # -------------------------------------------------------------------------
    # Done conditions
    # -------------------------------------------------------------------------
    def _is_done(self):
        """Return True if any termination condition is met."""
        return (self._is_state_invalid() or
                self._is_out_of_bounds() or
                self._is_goal_reached() or
                self._is_out_of_time() or
                self._is_out_of_steps())

    def _is_state_invalid(self):
        """Placeholder for checking NaNs / flipped robot or other invalid states."""
        return False

    def _is_out_of_bounds(self):
        """Return True if robot leaves the reasonable area around the goal."""
        x1 = -abs(self._goal_pos[0])
        x2 = abs(self._goal_pos[0])
        y1 = -abs(self._goal_pos[1])
        y2 = abs(self._goal_pos[1])
        position, _ = p.getBasePositionAndOrientation(self._robot_id)
        x, y = float(position[0]), float(position[1])
        tolerance = 5.0
        xmin, xmax = x1 - tolerance, x2 + tolerance
        ymin, ymax = y1 - tolerance, y2 + tolerance
        return not (xmin <= x <= xmax and ymin <= y <= ymax)

    def _is_out_of_time(self):
        """Return True if elapsed time exceeds allowed horizon."""
        time_tolerance = 120
        return self.get_time() > self._simulation_time_limit + time_tolerance

    def _is_out_of_steps(self):
        """Return True if controller step count exceeded a safety cap."""
        return self._controller._step_count > 100_000

    def _is_goal_reached(self):
        """Return True if last waypoint has been reached (index >= len-1)."""
        return self._current_reward_point_index >= (len(self._reward_points) - 1)


# -----------------------------------------------------------------------------
# Controller2 class (unchanged behavior except clear docstrings & debug)
# -----------------------------------------------------------------------------
class Controller2:
    """
    Controller that:
     - computes LQR baseline (LQRBaseline)
     - combines baseline with DRL residuals
     - maps resulting commands to wheel velocities and applies them via PyBullet
    """

    def __init__(self, wheel_base: float = 0.34, wheel_radius: float = 0.6 / 2.0, debug: bool = False):
        # Geometry/limits
        self._wheel_base = float(wheel_base)
        self._wheel_radius = float(wheel_radius)
        self._max_wheel_speed = 1.333
        self._max_wheel_accel = 1.333
        self._sim_timestep = 1.0 / 50.0

        # Bookkeeping
        self._step_count = 0
        self._previous_joint_velocities = [0.0, 0.0]
        self._robot_id = None

        # Baseline helper
        self._lqr_baseline = LQRBaseline(wheel_base=self._wheel_base, debug=debug)

        # Residual scaling (tune)
        self._residual_scale = np.array([0.6, 0.6], dtype=np.float32)  # [velocity_scale, omega_scale]

        # Last outputs
        self._last_baseline_command = np.array([0.0, 0.0], dtype=np.float32)
        self._last_total_command = np.array([0.0, 0.0], dtype=np.float32)
        self._last_residual = np.array([0.0, 0.0], dtype=np.float32)

        self._debug = bool(debug)

    # ----------------------------
    # Setup
    # ----------------------------
    def _set_robot_id(self, robot_id):
        """Register the robot id and adjust wheel dynamics."""
        self._robot_id = robot_id
        wheel_indices = [1, 2, 4, 5]
        for wheel_index in wheel_indices:
            p.changeDynamics(self._robot_id, wheel_index, lateralFriction=1.0)

    # ----------------------------
    # Baseline computation
    # ----------------------------
    def compute_baseline(self, reference_waypoint):
        """
        Compute LQR baseline commands for a reference waypoint.
        reference_waypoint: [x, y, z, yaw]
        Returns: (velocity_baseline, omega_baseline)
        """
        position, orientation = p.getBasePositionAndOrientation(self._robot_id)
        current_yaw = p.getEulerFromQuaternion(orientation)[2]
        linear_velocity, _ = p.getBaseVelocity(self._robot_id)

        # Project into body frame to get approximate forward speed
        cos_y = np.cos(-current_yaw)
        sin_y = np.sin(-current_yaw)
        vx_body = cos_y * linear_velocity[0] - sin_y * linear_velocity[1]
        forward_speed = float(vx_body)

        # Unpack reference waypoint
        x_ref, y_ref, z_ref, yaw_ref = reference_waypoint

        # Use a default velocity_reference; consider passing path-based v_ref later
        velocity_reference = 0.6

        # Compute baseline via LQRBaseline
        velocity_baseline, omega_baseline = self._lqr_baseline.baseline_control(
            current_position=position,
            current_yaw=current_yaw,
            current_forward_speed=forward_speed,
            reference=(x_ref, y_ref, z_ref, yaw_ref, velocity_reference)
        )

        self._last_baseline_command = np.array([velocity_baseline, omega_baseline], dtype=np.float32)

        if self._debug:
            print("[Controller2.compute_baseline] forward_speed=%.3f baseline=(vel=%.3f, omega=%.3f)" % (
                forward_speed, velocity_baseline, omega_baseline))

        return float(velocity_baseline), float(omega_baseline)

    # ----------------------------
    # Apply commands (baseline + residual)
    # ----------------------------
    def forward(self, agent_residual, reference_waypoint):
        """
        Combine baseline and residuals, send wheel velocity commands to the robot,
        and step the simulation a few micro-steps.
        Returns: (velocity_total, omega_total, baseline_tuple, residual_tuple)
        """
        # Compute baseline
        baseline_velocity, baseline_omega = self.compute_baseline(reference_waypoint)

        # Prepare residual
        residual = np.asarray(agent_residual, dtype=np.float32).reshape(-1)[:2]
        residual_scaled = self._residual_scale * residual

        # Compose total commands
        velocity_command_total = float(baseline_velocity + residual_scaled[0])
        omega_command_total = float(baseline_omega + residual_scaled[1])

        # Clip
        velocity_command_total = float(np.clip(velocity_command_total, -1.2, 1.2))
        omega_command_total = float(np.clip(omega_command_total, -4.0, 4.0))

        # Store
        self._last_baseline_command = np.array([baseline_velocity, baseline_omega], dtype=np.float32)
        self._last_total_command = np.array([velocity_command_total, omega_command_total], dtype=np.float32)
        self._last_residual = np.array(residual_scaled, dtype=np.float32)

        # Map to wheel velocities for differential drive (two wheel groups)
        joint_velocities = [0.0, 0.0]
        joint_velocities[0] = (velocity_command_total - (omega_command_total * self._wheel_base)) / (2.0 * self._wheel_radius)
        joint_velocities[1] = (velocity_command_total + (omega_command_total * self._wheel_base)) / (2.0 * self._wheel_radius)

        # Enforce acceleration limits
        for i in range(2):
            delta_v = joint_velocities[i] - self._previous_joint_velocities[i]
            max_delta_v = self._max_wheel_accel * self._sim_timestep * 50.0
            if abs(delta_v) > max_delta_v:
                joint_velocities[i] = self._previous_joint_velocities[i] + np.sign(delta_v) * max_delta_v

        # Clip per-wheel speeds
        joint_velocities = np.clip(joint_velocities,
                                   a_min=[-self._max_wheel_speed, -self._max_wheel_speed],
                                   a_max=[self._max_wheel_speed, self._max_wheel_speed])

        self._previous_joint_velocities = joint_velocities.copy()

        # Build per-joint velocity list for URDF with 6 joints
        my_joint_velocities = [0.0, joint_velocities[0], joint_velocities[0],
                               0.0, joint_velocities[1], joint_velocities[1]]

        for i in range(6):
            p.setJointMotorControl2(
                bodyUniqueId=self._robot_id,
                jointIndex=i,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=my_joint_velocities[i],
            )

        # Step simulation micro-steps for better numerical stability
        for _ in range(10):
            p.stepSimulation()

        # Update counter
        self._step_count += 10

        if self._debug:
            print("[Controller2.forward] residual_input=", residual, "residual_scaled=", residual_scaled)
            print("[Controller2.forward] baseline=(%.3f, %.3f) total=(%.3f, %.3f)" % (
                baseline_velocity, baseline_omega, velocity_command_total, omega_command_total))

        return (velocity_command_total,
                omega_command_total,
                (baseline_velocity, baseline_omega),
                (float(residual_scaled[0]), float(residual_scaled[1])))
