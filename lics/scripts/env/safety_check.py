import numpy as np
from numba import jit
import laser_geometry.laser_geometry as lg
import sensor_msgs.point_cloud2 as pc2

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

# Costmap
COSTMAP_RESOLUTION = 0.05
ROBOT_REAR_PIXEL = 7  # =1(center)+6(int(0.21/0.05)) and 0.05 is costmap resolution
ROBOT_SIDE_PIXEL = 4  # int(0.155/0.05) and 0.05 is costmap resolution
CENTER_COSTMAP = 21  # It can be changed according to robot.costmap_range
COST_THRESHOLD = 70  # Can be increased to 80


def transform_points(points, x, y, psi):
    # Transform the global path to the robot's frame
    mat = np.array(
        [[np.cos(psi), np.sin(psi), -x * np.cos(psi) - y * np.sin(psi)],
         [-np.sin(psi), np.cos(psi), x * np.sin(psi) - y * np.cos(psi)],
         [0, 0, 1]],
    )

    pi = np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)
    pr = np.matmul(mat, pi.T)
    return pr[:2].T


@jit(nopython=True)
def safety_check_rotation(points, laser_offset, safe_radius):
    min_dist = np.inf
    min_point = None

    for x, y in points:
        x = x + laser_offset
        dist = np.sqrt(x ** 2 + y ** 2)
        if dist < safe_radius:
            min_dist = min(min_dist, dist)
            if min_dist == dist:
                min_point = (x, y)

    return False, min_dist / 10, min_point, 0


@jit(nopython=True)
def safety_check_linear(points, v, laser_offset, width, padding, laser_clip):
    unsafe_zone = False
    min_dist = np.inf
    min_point = None

    for x, y in points:
        x = x + laser_offset
        if v * x < 0:
            continue
        dist = np.sqrt(x ** 2 + y ** 2)
        if dist < laser_clip and abs(y) < (width / 2 + padding):
            unsafe_zone = True
            min_dist = min(min_dist, dist)
            if min_dist == dist:
                min_point = (x, y)

    safe = not unsafe_zone
    dt = min_dist / abs(v)
    return safe, dt, min_point, 1


@jit(nopython=True)
def safety_check_radial(points, v, w, laser_offset, width, padding, laser_clip):
    # w positive: turning left, w negative: turning right
    turning_radius = v / w

    outer_radius = abs(turning_radius) + width / 2 + padding
    inner_radius = max(0, abs(turning_radius) - width / 2 - padding)

    unsafe_zone = False
    min_angle = np.inf
    min_point = None

    for x, y in points:
        x = x + laser_offset
        if v * x < 0:
            continue
        dist = np.sqrt(x ** 2 + y ** 2)
        circle_dist = np.sqrt(x ** 2 + (y - turning_radius) ** 2)
        if dist < laser_clip and inner_radius < circle_dist < outer_radius:
            unsafe_zone = True
            gamma = np.arctan2(y - turning_radius, x)
            gamma_0 = np.arctan2(-turning_radius, 0)
            angle_dist = np.abs(gamma - gamma_0)
            min_angle = min(min_angle, angle_dist)
            if min_angle == angle_dist:
                min_point = (x, y)

    safe = not unsafe_zone
    dt = min_angle / abs(w)
    return safe, dt, min_point, 2


class SafetyCheck:
    """
    A class for performing safety checks for a mobile robot or autonomous vehicle using sensor data.

    Attributes:
        width (float): The length of the vehicle or robot (in meters).
        padding (float): Additional safety padding to consider around the vehicle (in meters).
        laser_clip (float): Maximum distance to consider for laser range finder readings (in meters).

    Methods:
        __call__(msg, v, w): Determines if the vehicle is in a safe state given sensor data and velocities.
        safety_check_linear(angles, dist, v): Checks for safety in a linear motion context.
        safety_check_radial(angles, dist, v, w): Checks for safety in a radial (turning) motion context.
    """

    def __init__(self, length=0.508, width=0.43, padding=0.03, laser_offset=0.025, laser_clip=2.0, use_costmap=True):
        # def __init__(self, length=0.18, width=0.204, padding=0.03, laser_offset=0.025, laser_clip=2.0, use_costmap=True):
        """
        Initializes the SafetyCheck with specified vehicle parameters.

        Args:
            width (float): The physical width of the vehicle, default is 0.33 meters.
            padding (float): Additional safety margin around the vehicle, default is 0.1 meters.
            laser_clip (float): The clipping distance for the laser range finder, default is 2.0 meters.
        """

        self.length = length
        self.width = width
        self.padding = padding
        self.laser_offset = laser_offset  # Front direction offset from the center of the robot
        self.cost_threshold = 100
        self.safe_radius = np.sqrt(self.length ** 2 + self.width ** 2) / 2
        self.laser_clip = laser_clip
        self.lp = lg.LaserProjection()

        self.laser_points = np.empty((0, 2))
        self.costmap_points = np.empty((0, 2))

        self.use_costmap = use_costmap

    def __call__(self, v, w):
        # Combine the laser and costmap points
        if self.use_costmap:
            try:
                points = np.concatenate([self.laser_points, self.costmap_points])
            except ValueError:
                points = self.laser_points
        else:
            points = self.laser_points

        if len(points) == 0:
            return False, np.inf, None, 1

        # Visualize the points
        # plt.figure()
        # Clear the plot
        # plt.clf()
        # plt.scatter(points[:, 0], points[:, 1], c='b', s=1)
        # plt.xlim(-5, 5)
        # plt.ylim(-5, 5)
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.pause(0.001)

        # if abs(v) < 0.3 and abs(w) >= 0.1:
        #     # Safety type 0: Rotating in place
        #     result = safety_check_rotation(points, self.laser_offset, self.safe_radius)
        #     if not result[0]:
        #         return result

        if abs(w) < 0.01:
            # Safety type 1: Linear motion
            return safety_check_linear(points, v, self.laser_offset, self.width, self.padding, self.laser_clip)
        else:
            # Safety type 2: Radial motion
            return safety_check_radial(points, v, w, self.laser_offset, self.width, self.padding, self.laser_clip)

    def update_laser(self, msg):
        pc2_msg = self.lp.projectLaser(msg)
        point_generator = pc2.read_points(pc2_msg)
        self.laser_points = np.array([point[:2] for point in point_generator])

    def update_local_costmap(self, msg, odom):
        resolution = msg.info.resolution
        width = msg.info.width
        height = msg.info.height
        origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])

        # Find cells where cost value is greater than threshold
        costmap = np.array(msg.data).reshape((height, width)).T
        obstacle_cells = np.argwhere(costmap >= self.cost_threshold)
        points = (obstacle_cells + 0.5) * resolution + origin

        # Transform points to the robot's frame
        points = transform_points(points, *odom)

        # Update the costmap points only when x is negative
        self.costmap_points = points[points[:, 0] < 0]

    # def back_linear_check(self, v, costmap, psi):
    #     unsafe_zone = False
    #     min_dist = np.inf
    #
    #     points = []
    #     unsafe_points = []
    #
    #     sin = np.sin(psi)
    #     cos = np.cos(psi)
    #
    #     for row in range(costmap.shape[0]):  # y
    #         for col in range(costmap.shape[1]):  # x
    #             # Robot's rear position (Robot's rear part is behind : 7 pixels and each side : 4 pixels
    #             if sin * col + cos * row - CENTER_COSTMAP * (
    #                     sin + cos) >= 0 and cos * col - sin * row - CENTER_COSTMAP * (cos - sin) - ROBOT_SIDE_PIXEL <= 0 \
    #                     and cos * col + sin * row - CENTER_COSTMAP * (
    #                     cos + sin) + ROBOT_REAR_PIXEL >= 0 and cos * col - sin * row - CENTER_COSTMAP * (
    #                     cos - sin) + ROBOT_SIDE_PIXEL >= 0:
    #                 continue
    #             if costmap[row][col] >= COST_THRESHOLD and cos * col - sin * row - CENTER_COSTMAP * (
    #                     cos - sin) - ROBOT_SIDE_PIXEL <= 0 and cos * col - sin * row - CENTER_COSTMAP * (
    #                     cos - sin) + ROBOT_SIDE_PIXEL >= 0:
    #                 unsafe_zone = True
    #                 min_dist = min(min_dist, abs(cos * col + sin * row - CENTER_COSTMAP * (cos + sin) + 7) * 0.05)
    #
    #     safe = not unsafe_zone
    #     dt = min_dist / abs(v)
    #     return safe, dt
    #
    # def back_radial_check(self, v, w, costmap, psi):
    #     # w positive: turning left, w negative: turning right
    #     sin = np.sin(psi)
    #     cos = np.cos(psi)
    #     rot_loc2glob = np.array([[cos, sin], [-sin, cos]])
    #     turning_radius = v / (w * COSTMAP_RESOLUTION)
    #     turning_center = np.matmul(rot_loc2glob, np.array([turning_radius, 0]).T) + CENTER_COSTMAP
    #     outer_radius = abs(turning_radius) + ROBOT_SIDE_PIXEL + self.padding / COSTMAP_RESOLUTION
    #     inner_radius = max(0, abs(turning_radius) - ROBOT_SIDE_PIXEL - self.padding / COSTMAP_RESOLUTION)
    #
    #     unsafe_zone = False
    #     min_angle = np.inf
    #
    #     points = []
    #     unsafe_points = []
    #
    #     for row in range(costmap.shape[0]):  # y
    #         for col in range(costmap.shape[1]):  # x
    #             if sin * col + cos * row - CENTER_COSTMAP * (
    #                     sin + cos) >= 0 and cos * col - sin * row - CENTER_COSTMAP * (cos - sin) - ROBOT_SIDE_PIXEL <= 0 \
    #                     and cos * col + sin * row - CENTER_COSTMAP * (
    #                     cos + sin) + ROBOT_REAR_PIXEL >= 0 and cos * col - sin * row - CENTER_COSTMAP * (
    #                     cos - sin) + ROBOT_SIDE_PIXEL >= 0:
    #                 continue
    #             circle_dist = np.sqrt((turning_center[0] - col) ** 2 + (turning_center[1] - row) ** 2)
    #             if costmap[row][
    #                 col] >= COST_THRESHOLD and sin * col + cos * row - sin * CENTER_COSTMAP - cos * CENTER_COSTMAP <= 0 and inner_radius < circle_dist < outer_radius \
    #                     and (cos * (col - turning_center[0]) - sin * (row - turning_center[1])) * turning_radius >= 0:
    #                 unsafe_zone = True
    #                 gamma = (np.arctan2(abs(sin * col + cos * row - CENTER_COSTMAP * (sin + cos)), np.sqrt(
    #                     abs(circle_dist ** 2 - (sin * col + cos * row - CENTER_COSTMAP * (sin + cos)) ** 2)))
    #                          angle_dist = np.abs(gamma))
    #                 min_angle = min(min_angle, angle_dist)
    #
    #                 safe = not unsafe_zone
    #                 dt = min_angle / abs(w)
    #     return safe, dt
