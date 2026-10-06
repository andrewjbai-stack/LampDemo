"""Top-level LeLamp launch: camera + simulated body (MuJoCo) + logic.

    ros2 launch ~/lelamp_ws/launch/lelamp.launch.py
    ros2 launch ~/lelamp_ws/launch/lelamp.launch.py rviz:=false device:=/dev/video2

RViz shows only the webcam, face-tracking and sim head camera feeds.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sim_launch = os.path.join(get_package_share_directory('lelamp_sim'),
                              'launch', 'sim.launch.py')

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Open RViz with the camera feeds'),
        DeclareLaunchArgument('device', default_value='/dev/video0',
                              description='Webcam device path or index'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(sim_launch),
            launch_arguments={'rviz': LaunchConfiguration('rviz')}.items(),
        ),
        Node(
            package='lelamp_camera',
            executable='camera_node',
            parameters=[{'device': LaunchConfiguration('device')}],
        ),
        Node(
            package='lelamp_logic',
            executable='face_node',
        ),
        Node(
            package='lelamp_logic',
            executable='object_node',
        ),
        Node(
            package='lelamp_logic',
            executable='voice_node',
        ),
        Node(
            package='lelamp_logic',
            executable='logic_node',
        ),
    ])
