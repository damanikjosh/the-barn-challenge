#!/usr/bin/env python3
import os
import torch
import rospy
import rospkg
import easydict

from env import Robot
from models import Transformer
from utils import load_config

DEVICE = torch.device('cpu')
MODEL_FOLDER = 'models'  # Change the model path here
MODEL_FILE = 'transformer_99.pth'  # Change the model file here
CONTROL_FREQ = 20  # Hz. # Change the control frequency here

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

    config = load_config()
    rospack = rospkg.RosPack()
    lics_path = rospack.get_path('lics')
    model_path = os.path.join(lics_path, MODEL_FOLDER, MODEL_FILE)

    print('Initializing robot...')
    robot = Robot(**config['robot'])
    print('Robot initialized')

    print('Initializing model...')
    model = Transformer(config_dict).to(DEVICE)
    print('Model initialized')
    print('Loading model weights...')
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print('Model weights loaded')

    goal_x = rospy.get_param('~goal_x', 0)
    goal_y = rospy.get_param('~goal_y', 10)
    goal_psi = rospy.get_param('~goal_psi', 1.57)

    robot.set_goal(goal_x, goal_y, goal_psi)

    rate = rospy.Rate(CONTROL_FREQ)

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
        robot.set_velocity(action[0], action[1])
        rate.sleep()
