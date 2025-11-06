# path_generation.py
"""
Quintic polynomial 2D planner and display helpers.

This is largely the original code with cleaned-up comments and docstrings.
"""

import numpy as np
import math
import pybullet as p
from mars_terrain import get_height_at

MAX_T = 100.0
MIN_T = 5.0


class QuinticPolynomial:
    """
    1D quintic polynomial defined by boundary pos, vel, accel at start and end.
    Provides point and derivative evaluation utilities.
    """
    def __init__(self, xs, vxs, axs, xe, vxe, axe, time_):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time_ ** 3, time_ ** 4, time_ ** 5],
                      [3 * time_ ** 2, 4 * time_ ** 3, 5 * time_ ** 4],
                      [6 * time_, 12 * time_ ** 2, 20 * time_ ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time_ - self.a2 * time_ ** 2,
                      vxe - self.a1 - 2 * self.a2 * time_,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        return self.a0 + self.a1 * t + self.a2 * t ** 2 + \
               self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

    def calc_first_derivative(self, t):
        return self.a1 + 2 * self.a2 * t + \
               3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

    def calc_second_derivative(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

    def calc_third_derivative(self, t):
        return 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2


class QuinticPolynomial2dPlanner:
    """
    2D planner using two independent quintic polynomials (x and y).
    Provides methods to sample positions and yaw along the path.
    """
    def __init__(self, sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt, T):
        self.sx = sx
        self.sy = sy
        self.syaw = syaw
        self.sv = sv
        self.sa = sa
        self.gx = gx
        self.gy = gy
        self.gyaw = gyaw
        self.gv = gv
        self.ga = ga
        self.max_accel = max_accel
        self.max_jerk = max_jerk
        self.dt = dt
        self.T = T

        self.vxs = self.sv * math.cos(self.syaw)
        self.vys = self.sv * math.sin(self.syaw)
        self.vxg = self.gv * math.cos(self.gyaw)
        self.vyg = self.gv * math.sin(self.gyaw)

        self.axs = self.sa * math.cos(self.syaw)
        self.ays = self.sa * math.sin(self.syaw)
        self.axg = self.ga * math.cos(self.gyaw)
        self.ayg = self.ga * math.sin(self.gyaw)

        self.compute()

    def compute(self):
        """Construct quintic polynomials for x(t) and y(t)."""
        self.xqp = QuinticPolynomial(self.sx, self.vxs, self.axs, self.gx, self.vxg, self.axg, self.T)
        self.yqp = QuinticPolynomial(self.sy, self.vys, self.ays, self.gy, self.vyg, self.ayg, self.T)

    def get_pos(self, t):
        """Return (x,y,yaw) at time t on the polynomial path."""
        x = self.xqp.calc_point(t)
        y = self.yqp.calc_point(t)
        yaw = math.atan2(self.yqp.calc_first_derivative(t), self.xqp.calc_first_derivative(t))
        return x, y, yaw

    def get_waypoints(self):
        """Return finely sampled waypoints along the path using dt."""
        time_, rx, ry, ryaw, rv, ra, rj = [], [], [], [], [], [], []
        for t in np.arange(0.0, self.T + self.dt, self.dt):
            time_.append(t)
            rx.append(self.xqp.calc_point(t))
            ry.append(self.yqp.calc_point(t))
            vx = self.xqp.calc_first_derivative(t)
            vy = self.yqp.calc_first_derivative(t)
            yaw = math.atan2(vy, vx)
            ryaw.append(yaw)
        return time_, rx, ry, ryaw, rv, ra, rj

    def get_waypoints_rewards(self):
        """Return sparser waypoints (1s interval) for reward points / navigation references."""
        time_, rx, ry, ryaw, rv, ra, rj = [], [], [], [], [], [], []
        for t in np.arange(0.0, self.T + 1.0, 1.0):
            time_.append(t)
            rx.append(self.xqp.calc_point(t))
            ry.append(self.yqp.calc_point(t))
            vx = self.xqp.calc_first_derivative(t)
            vy = self.yqp.calc_first_derivative(t)
            yaw = math.atan2(vy, vx)
            ryaw.append(yaw)
        return time_, rx, ry, ryaw, rv, ra, rj


def display_path(time_, rx, ry, ryaw, rv, ra, rj, heightfieldData):
    """
    Draw the path in the PyBullet simulation by projecting path points onto the terrain.
    """
    rz = [get_height_at(x, y, heightfieldData) for x, y in zip(rx, ry)]
    waypoints = np.array([rx, ry, rz]).T.tolist()

    for i in range(len(waypoints) - 1):
        p.addUserDebugLine(
            waypoints[i], waypoints[i + 1],
            lineColorRGB=[1, 0, 0],
            lineWidth=2.0,
            lifeTime=0
        )
