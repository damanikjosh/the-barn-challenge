#!/usr/bin/env python3

import rospy
import numpy as np

from sensor_msgs.msg import LaserScan

laser_pub = None


def callback(msg):
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

    laser_pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('laser_interpolation')
    laser_pub = rospy.Publisher('/scan/interpolated', LaserScan, queue_size=1)
    rospy.Subscriber('/scan', LaserScan, callback)
    rospy.spin()
