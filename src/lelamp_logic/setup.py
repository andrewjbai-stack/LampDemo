from glob import glob

from setuptools import setup

package_name = 'lelamp_logic'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, package_name + '.logic_node'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', glob('models/*.onnx')),
    ] + [
        # one entry per sound (assets/sounds/<name>/<name>_1..5.wav); data_files isn't recursive
        ('share/' + package_name + '/' + d.rstrip('/'), glob(d + '*.wav'))
        for d in sorted(glob('assets/sounds/*/'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Andrew',
    maintainer_email='thecircularectangle@gmail.com',
    description='LeLamp behavior/thinking logic.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'logic_node = lelamp_logic.logic_node.node:main',
            'motion_node = lelamp_logic.motion_node:main',
            'face_node = lelamp_logic.face_node:main',
            'object_node = lelamp_logic.object_node:main',
            'voice_node = lelamp_logic.voice_node:main',
            'llm_node = lelamp_logic.llm_node:main',
            'sound_node = lelamp_logic.sound_node:main',
        ],
    },
)
