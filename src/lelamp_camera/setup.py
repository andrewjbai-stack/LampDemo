from setuptools import setup

package_name = 'lelamp_camera'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Andrew',
    maintainer_email='thecircularectangle@gmail.com',
    description="Publishes the laptop webcam as the LeLamp's eye.",
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_node = lelamp_camera.camera_node:main',
        ],
    },
)
