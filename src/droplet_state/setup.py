from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'droplet_state'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.json')),
        (os.path.join('share', package_name, 'calib'),
            glob('calib/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rajneesh',
    maintainer_email='rajneesh.1491981@gmail.com',
    description='State estimation and data collection for DropletRunner',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'estimator = droplet_state.estimator:main',
            'data_collector = droplet_state.data_collector:main',
            'policy_runner = droplet_state.policy_runner:main',          #for trained weights by RL testing
            'pid_controller = droplet_state.pid_controller:main',        #for PID controller testing
            'open_loop_replay = droplet_state.open_loop_replay:main',    #for open loop testing    
        ],
    },
)