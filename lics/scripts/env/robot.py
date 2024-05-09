import os
import yaml
import rospy
import rospkg
import subprocess
import numpy as np
from scipy.signal import savgol_filter
from dynamic_reconfigure.client import Client
from scipy.spatial.transform import Rotation as R

from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from robot_localization.srv import SetPose
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped


STATE_NORMAL = 0
STATE_WAIT = 1


class Robot:
    def __init__(
            self,
    ):
        # self.lp_client = None
        self.inflater_client = None

        # ROS initialization and parameter setting
        rospy.init_node('robot', anonymous=True)

        # Get params from ROS parameter server
        config_path = rospy.get_param('~config_path', None)

        if config_path is None:
            rospack = rospkg.RosPack()
            base_path = rospack.get_path('lics')
            config_path = os.path.join(base_path, 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['robot']

        rospy.set_param('/use_sim_time', config['use_sim_time'])

        # Subscribers
        rospy.Subscriber(config['laser_topic'], LaserScan, self.update_laser, queue_size=1)
        rospy.Subscriber('/odometry/filtered', Odometry, self.update_state, queue_size=1)
        # rospy.Subscriber('/move_base/cmd_vel', Twist, self.get_lp_velocity, queue_size=1)
        rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, self.update_path, queue_size=1)

        # Publishers and service proxies
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.pose_srv = rospy.ServiceProxy('/set_pose', SetPose)
        self.clear_costmap_srv = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)

        # Parameters in float32
        self.los = np.float32(config['los'])
        self.v_multiplier = np.float32(1.0)
        self.max_v = np.float32(config['max_v'])
        self.min_v = np.float32(config['min_v'])
        self.max_w = np.float32(config['max_w'])
        self.min_w = np.float32(config['min_w'])
        self.laser_dist = np.float32(config['laser_clip'])
        self.laser_scale = np.float32(config['laser_scale'])
        self.threshold_dist = np.float32(config['threshold_dist'])
        self.threshold_v = np.float32(config['threshold_v'])
        self.min_inflation_radius = np.float32(config['min_inflation_radius'])
        self.max_inflation_radius = np.float32(config['max_inflation_radius'])

        # Initial state in float32
        self.state = STATE_NORMAL
        self.inflated = False
        self.interpolate_laser = config['interpolate_laser']
        self.odom = np.zeros((3,), dtype=np.float32)
        self.lp_vel = np.zeros((2,), dtype=np.float32)
        self.local_goal = np.zeros((2,), dtype=np.float32)
        self.global_path = np.zeros((0, 2), dtype=np.float32)
        self.laser = np.full((720,), self.laser_dist, dtype=np.float32)

        self.reset()
        self.inflater_client = Client('/move_base/global_costmap/inflater_layer')

    def update_state(self, msg):
        pose = msg.pose.pose
        twist = msg.twist.twist
        r = R.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        self.odom[:] = (pose.position.x, pose.position.y, r.as_euler('zyx')[0])

        v = twist.linear.x / self.max_v
        if self.inflater_client is not None:
            if v > self.threshold_v:
                self.set_inflation_radius(
                    self.min_inflation_radius + (v - self.threshold_v) / (self.max_v - self.threshold_v) * (
                                self.max_inflation_radius - self.min_inflation_radius))
            else:
                self.set_inflation_radius(self.min_inflation_radius)

    def update_path(self, msg):
        if len(msg.poses) == 0:
            # If the global path is empty, reduce the velocity
            self.v_multiplier = np.float32(0.3)
            return
        gp = np.array([[pose.pose.position.x, pose.pose.position.y] for pose in msg.poses], dtype=np.float32)
        x, y = gp[:, 0], gp[:, 1]
        try:
            x_hat = savgol_filter(x, 19, 3).astype(np.float32)
            y_hat = savgol_filter(y, 19, 3).astype(np.float32)
        except ValueError:
            x_hat, y_hat = x, y
        self.global_path = self.transform_path(np.stack([x_hat, y_hat], axis=-1), *self.odom)
        self.local_goal = self.get_local_goal(self.global_path)

        # Reset the velocity multiplier if the global path is not empty
        self.v_multiplier = np.float32(1.0)

    def update_laser(self, msg):
        if self.interpolate_laser:
            new_ranges = []
            new_intensities = []

            for i in range(len(msg.ranges)):
                if i % 3 == 0:
                    new_ranges.append(msg.ranges[i])
                    new_intensities.append(msg.intensities[i])
                new_ranges.append(msg.ranges[i])
                new_intensities.append(msg.intensities[i])

            msg.ranges = new_ranges
            msg.intensities = new_intensities
            msg.angle_increment *= 540 / 720

        self.laser[:] = np.clip(np.array(msg.ranges, dtype=np.float32) * self.laser_scale, 0, self.laser_dist)
        self.laser[self.laser == 0] = self.laser_dist

        min_dist_front = np.min(self.laser[253:467])
        if min_dist_front < self.threshold_dist:
            self.v_multiplier = min_dist_front / self.threshold_dist
        else:
            self.v_multiplier = np.float32(1.0)

    def get_local_goal(self, gp):
        local_goal = np.zeros(2, dtype=np.float32)
        odom = np.zeros(2, dtype=np.float32)
        if len(gp) > 0:
            if np.linalg.norm(gp[0] - odom) > 0.05:
                odom = gp[0]
            for wp in gp:
                dist = np.linalg.norm(wp - odom)
                if dist > self.los:
                    break
            local_goal = wp - odom
            local_goal /= np.linalg.norm(local_goal)

        return local_goal

    def set_velocity(self, v, w):
        # Scale the velocity to the maximum and minimum values and clip
        if self.state == STATE_NORMAL:
            twist = Twist()
            twist.linear.x = np.clip(v * self.max_v * self.v_multiplier, self.min_v, self.max_v)
            twist.angular.z = np.clip(w * self.max_w, self.min_w, self.max_w)
            self.vel_pub.publish(twist)

    def set_inflation_radius(self, inflation_radius):
        self.inflater_client.update_configuration({
            'inflation_radius': inflation_radius
        })

    def get_lp_velocity(self, msg):
        v = msg.linear.x / self.max_v
        w = msg.angular.z / self.max_w
        self.lp_vel[:] = (v, w)

    def clear_costmap(self):
        for _ in range(3):
            rospy.wait_for_service('/move_base/clear_costmaps')
            try:
                self.clear_costmap_srv()
            except rospy.ServiceException:
                print("/clear_costmaps service call failed")

            rospy.sleep(0.1)

    def reset(self):
        # Stop the robot
        self.set_velocity(0, 0)

        # Reset the robot's pose
        rospy.wait_for_service('/set_pose')
        reset_pose = PoseWithCovarianceStamped()
        reset_pose.header.frame_id = 'odom'
        self.pose_srv(reset_pose)

        self.clear_costmap()  # Clear the costmap
        self.odom[:] = (0, 0, 0)  # Reset the state

    @staticmethod
    def transform_path(gp, x, y, psi):
        # Transform the global path to the robot's frame
        mat = np.array(
            [[np.cos(psi), np.sin(psi), -x * np.cos(psi) - y * np.sin(psi)],
             [-np.sin(psi), np.cos(psi), x * np.sin(psi) - y * np.cos(psi)],
             [0, 0, 1]],
        )

        pi = np.concatenate([gp, np.ones_like(gp[:, :1])], axis=-1)
        pr = np.matmul(mat, pi.T)
        return pr[:2].T
