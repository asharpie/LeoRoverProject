# lqr_baseline.py
"""
LQR baseline controller helper.

Provides:
 - LQRBaseline: computes a baseline control (velocity_command, omega_command)
   given the current robot pose/velocity and a reference waypoint (x,y,yaw,velocity_ref).
 - Uses SciPy's continuous ARE solver if available; otherwise falls back to tuned PD-style gains.

This file intentionally exposes clear, fully spelled variable names for readability.
"""

import numpy as np

# Try to use SciPy's CARE solver for a proper LQR gain. If SciPy is not installed,
# fallback to a PD-style approximate gain matrix.
try:
    from scipy.linalg import solve_continuous_are
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


class LQRBaseline:
    """
    Compute baseline control (velocity_command, omega_command) for tracking a
    path reference. The state used is:
      e = [lateral_error, heading_error, velocity_error].

    Public method:
      baseline_control(current_position, current_yaw, current_forward_speed, reference_tuple)
    where reference_tuple = (x_ref, y_ref, yaw_ref, velocity_reference)
    """

    def __init__(self, wheel_base=0.34, debug: bool = False):
        # physical parameter (for reference; LQR itself does not use wheel base)
        self.wheel_base = float(wheel_base)
        self.debug = bool(debug)

        # LQR costs (tunable)
        # Q penalizes state error (lateral, heading, velocity)
        self.Q = np.diag([10.0, 6.0, 1.0])
        # R penalizes effort on inputs (delta_velocity, omega)
        self.R = np.diag([0.5, 0.8])

        # Fallback PD-like gains (only used if SciPy unavailable)
        self.fallback_kp_lateral = 1.5
        self.fallback_kp_heading = 2.0
        self.fallback_kp_velocity = 0.6

    # ----------------------------
    # Public API
    # ----------------------------
    def baseline_control(self, current_position, current_yaw, current_forward_speed, reference):
        """
        Compute baseline control commands.

        Inputs:
          - current_position: (x, y, z) world coordinates of robot base
          - current_yaw: robot yaw (radians)
          - current_forward_speed: scalar forward velocity (m/s) (approx)
          - reference: (x_ref, y_ref, z_ref, yaw_ref, velocity_reference) OR (x_ref, y_ref, yaw_ref, velocity_reference)
                       The function will accept either 4- or 5-length tuples; z_ref is ignored.

        Returns:
          (velocity_command_baseline, omega_command_baseline)
        """
        # Normalize reference tuple
        if len(reference) == 5:
            x_ref, y_ref, _, yaw_ref, velocity_reference = reference
        else:
            x_ref, y_ref, yaw_ref, velocity_reference = reference

        # Compute error vector in path-local frame (lateral, heading, velocity error)
        error_vector = self._compute_tracking_errors_in_local_frame(
            current_position, current_yaw, current_forward_speed,
            (x_ref, y_ref, yaw_ref, velocity_reference)
        )

        # Linearize model around velocity_reference and form A,B
        A, B = self._linearize_unicycle_model(velocity_reference)

        # Compute state-feedback gain K via LQR or fallback
        K = self._compute_lqr_gain(A, B)

        # Control law: delta_u = -K @ e  where u = [delta_v, delta_omega]
        delta_u = - K @ error_vector  # shape (2,)

        # Compose baseline absolute control: velocity_reference + delta_v, omega_reference + delta_omega
        velocity_baseline = float(velocity_reference + delta_u[0])
        omega_baseline = float(0.0 + delta_u[1])  # we use omega_ref = 0.0 by default

        if self.debug:
            print("[LQRBaseline] reference=(x:{:.3f}, y:{:.3f}, yaw:{:.3f}, vel_ref:{:.3f})".format(
                x_ref, y_ref, yaw_ref, velocity_reference))
            print("[LQRBaseline] error_vector (lat, head, vel):", error_vector)
            print("[LQRBaseline] A=", A)
            print("[LQRBaseline] B=", B)
            print("[LQRBaseline] K=", K)
            print("[LQRBaseline] delta_u (dv, domega)=", delta_u)
            print("[LQRBaseline] baseline commands (velocity, omega)=", (velocity_baseline, omega_baseline))

        return velocity_baseline, omega_baseline

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _compute_tracking_errors_in_local_frame(self, current_position, current_yaw, current_forward_speed, reference):
        """
        Convert world-frame error into a local/path frame tracking error vector:
          - lateral_error: signed lateral offset from path measured in path frame (left positive)
          - heading_error: wrapped angle (current_yaw - yaw_ref) in [-pi, pi]
          - velocity_error: current_forward_speed - velocity_reference

        This uses the reference yaw to define forward direction of the path. This choice
        aligns the lateral error with the path's normal vector and is robust for tracking.
        """
        if len(reference) == 5:
            x_ref, y_ref, _, yaw_ref, velocity_reference = reference
        else:
            x_ref, y_ref, yaw_ref, velocity_reference = reference

        # Position differences (robot - reference)
        dx = float(current_position[0]) - float(x_ref)
        dy = float(current_position[1]) - float(y_ref)

        # Rotate (dx,dy) into path frame defined by yaw_ref:
        cos_r = np.cos(yaw_ref)
        sin_r = np.sin(yaw_ref)
        # In path frame: forward coordinate = cos_r*dx + sin_r*dy
        #                lateral coordinate = -sin_r*dx + cos_r*dy  (left positive)
        forward_distance_along_path = cos_r * dx + sin_r * dy
        lateral_distance_to_path = -sin_r * dx + cos_r * dy

        # heading error: robot yaw minus path yaw, wrapped to [-pi, pi]
        raw_heading_error = current_yaw - yaw_ref
        heading_error = ((raw_heading_error + np.pi) % (2.0 * np.pi)) - np.pi

        velocity_error = float(current_forward_speed) - float(velocity_reference)

        # Assemble state error vector
        error_vector = np.array([lateral_distance_to_path, heading_error, velocity_error], dtype=np.float64)
        return error_vector

    def _linearize_unicycle_model(self, velocity_reference):
        """
        Linearize a unicycle model about the reference forward speed.
        State: [lateral_error, heading_error, velocity_error]
        Inputs: [delta_v, omega]

        Returns continuous-time A, B matrices (3x3, 3x2).
        """
        A = np.zeros((3, 3), dtype=np.float64)
        A[0, 1] = float(velocity_reference)  # d(lateral)/dt = v_ref * heading_error
        A[2, 2] = -1.0  # simple damping for velocity error dynamics

        B = np.zeros((3, 2), dtype=np.float64)
        B[1, 1] = 1.0  # heading_error_dot depends directly on omega input
        B[2, 0] = 1.0  # velocity_error_dot depends on delta_v input

        return A, B

    def _compute_lqr_gain(self, A, B):
        """
        Compute the state-feedback gain K (2x3) using continuous-time LQR:
          K = R^-1 B^T P, where P solves A^T P + P A - P B R^-1 B^T P + Q = 0

        If SciPy is not available, return a sensible fallback K (approximate).
        """
        if SCIPY_AVAILABLE:
            P = solve_continuous_are(A, B, self.Q, self.R)
            K = np.linalg.inv(self.R) @ (B.T @ P)
            return K
        else:
            # Fallback approximate gains (constructed to mimic PD behaviour)
            K = np.zeros((2, 3), dtype=np.float64)
            # delta_v = -kp_v * velocity_error
            K[0, 2] = self.fallback_kp_velocity
            # omega = -kp_lat * lateral_error - kp_head * heading_error
            K[1, 0] = self.fallback_kp_lateral
            K[1, 1] = self.fallback_kp_heading
            return K
