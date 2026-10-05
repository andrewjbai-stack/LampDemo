"""Simulated lamp body: robot_state_publisher + MuJoCo sim (opens its viewer).

RViz is only used to watch the webcam, face-tracking and sim head camera feeds
(rviz:=false to skip it); the lamp itself is shown in MuJoCo.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('lelamp_sim')
    with open(os.path.join(share, 'urdf', 'lelamp.urdf')) as f:
        robot_description = f.read()

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Open RViz with the camera feeds'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(
            package='lelamp_sim',
            executable='mujoco_sim',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', os.path.join(share, 'rviz', 'cameras.rviz')],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ])
