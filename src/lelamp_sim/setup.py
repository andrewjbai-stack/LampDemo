from glob import glob

from setuptools import setup

package_name = 'lelamp_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/urdf', glob('urdf/*.urdf')),
        ('share/' + package_name + '/meshes', glob('meshes/*.stl')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Andrew',
    maintainer_email='thecircularectangle@gmail.com',
    description='Simulated LeLamp body: URDF, kinematic joint sim, and MuJoCo viewer.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'joint_sim = lelamp_sim.joint_sim_node:main',
            'mujoco_sim = lelamp_sim.mujoco_sim_node:main',
        ],
    },
)
