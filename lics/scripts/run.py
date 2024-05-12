#!/usr/bin/env python3
import os
import torch
import rospy
import rospkg
import easydict
import numpy as np

from env import Robot
from models import Transformer
from utils import load_config

DEVICE = torch.device('cpu')
MODEL_FOLDER = 'models'  # Change the model path here
MODEL_FILE = 'transformer_99.pth'  # Change the model file here

config_dict = easydict.EasyDict({
    "input_dim": 32,
    "num_patch": 36,
    "model_dim": 32,
    "ffn_dim": 256,
    "attention_heads": 4,
    "attention_dropout": 0.0,
    "dropout": 0.2,
    "encoder_layers": 3,
    "decoder_layers": 3,
    "device": DEVICE,
})

if __name__ == '__main__':
    rospack = rospkg.RosPack()
    lics_path = rospack.get_path('lics')
    model_path = os.path.join(lics_path, MODEL_FOLDER, MODEL_FILE)

    print('Initializing robot...')
    robot = Robot()
    print('Robot initialized')

    print('Initializing model...')
    model = Transformer(config_dict).to(DEVICE)
    print('Model initialized')
    print('Loading model weights...')
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print('Model weights loaded')

    # Get parameters from ROS parameter server
    control_freq = rospy.get_param('~control_freq', 20)

    rate = rospy.Rate(control_freq)

    # Initial warm-up
    for _ in range(10):
        action, _, _ = model(torch.rand(1, 720, 1).to(DEVICE), torch.rand(1, 2, 1).to(DEVICE))

    print('Starting control loop...')
    while not rospy.is_shutdown():
        obs = torch.tensor(robot.laser, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
        local_goal = torch.tensor(robot.local_goal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)

        with torch.no_grad():
            action, _, _ = model(obs, local_goal)

        action = action.squeeze().cpu().numpy()

        # odom_safe, odom_dt, odom_min_point, odom_safety_type = robot.safety_check(robot.odom[3], robot.odom[4])
        # model_safe, model_dt, model_min_point, _ = robot.safety_check(action[0] * robot.max_v, action[1] * robot.max_w)
        # lp_safe, lp_dt, lp_min_point, _ = robot.safety_check(robot.lp_vel[0], robot.lp_vel[1])

        # Angle between action and local goal
        # angle = np.arccos(
        #     np.dot(action, robot.local_goal) / (np.linalg.norm(action) * np.linalg.norm(robot.local_goal)))

        # if odom_dt < 1.0 or (robot.odom[3] < 0.3 and model_dt < 1.0):
        #     min_point = odom_min_point or model_min_point
        #     print(f'Collision is imminent. Time to collision: {odom_dt}. Action: {action}')
        #     # Check model safety
        #     if action[0] > 0.3 and model_dt >= 1.0:
        #         print(f'Model is safe. Time to collision: {model_dt}. Action: {action}')
        #         pass
        #     else:
        #         straight_safe, straight_dt, straight_min_point, _ = robot.safety_check(0.3, 0)
        #         if straight_dt >= 1.0:
        #             action = [0.3, 0]
        #             print(f'Straight is safe. Time to collision: {straight_dt}. Action: {action}')
        #         else:
        #             if odom_safety_type == 0:
        #
        #                 print(f'Safety type 0 alert. Moving backwards. Time to collision: {odom_dt}', end='. ')
        #                 if (min_point[0] > 0 and min_point[1] > 0) or \
        #                         (min_point[0] < 0 and min_point[1] < 0):
        #                     if min_point[0] > 0:
        #                         action = [-0.3, 0.3]
        #                         print('Case 3')
        #                     else:
        #                         action = [0.3, 0]
        #                         print('Case 4')
        #                 else:
        #                     if min_point[0] > 0:
        #                         action = [-0.3, -0.3]
        #                         print('Case 1')
        #                     else:
        #                         action = [0.3, 0]
        #                         print('Case 2')
        #             else:
        #                 direction = robot.local_goal / abs(robot.local_goal)
        #                 if min_point[0] > 0:
        #                     action = [-0.3, 0.15 * np.sign(direction[1])]
        #                 else:
        #                     action = [0.3, 0]
        #                 print(f'Other safety type alert. Rotating in place: {action}')

        # else:
        #     if odom_dt < 0.25: # If robot movement is unsafe
        #         if odom_safety_type == 0:
        #             min_point = odom_min_point or model_min_point
        #
        #             print(f'Safety type 0 alert. Moving backwards. Time to collision: {odom_dt}', end='. ')
        #             if (min_point[0] > 0 and min_point[1] > 0) or \
        #                     (min_point[0] < 0 and min_point[1] < 0):
        #                 if min_point[0] > 0:
        #                     action = [-0.3, -0.5]
        #                     print('Case 3')
        #                 else:
        #                     action = [0.3, 0]
        #                     print('Case 4')
        #             else:
        #                 if min_point[0] > 0:
        #                     action = [-0.3, 0.5]
        #                     print('Case 1')
        #                 else:
        #                     action = [0.3, 0]
        #                     print('Case 2')
        #         else:
        #             direction = robot.local_goal / abs(robot.local_goal)
        #             action = [-0.3, 0.5 * direction[1]]
        #             print(f'Other safety type alert. Rotating in place: {action}')
        #     else:
        #         print(f'Model is safe. Time to collision: {model_dt}. Action: {action}')
        #         pass

        robot.set_velocity(action[0], action[1])
        rate.sleep()
