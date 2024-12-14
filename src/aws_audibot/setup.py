from setuptools import find_packages, setup

package_name = 'aws_audibot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
#    data_files=[
#        ('share/ament_index/resource_index/packages',
#            ['resource/' + package_name]),
#        ('share/' + package_name, ['package.xml']),
#    ],
	data_files=[
	    ('share/ament_index/resource_index/packages',
	        ['resource/aws_audibot']),
	    ('share/aws_audibot', ['package.xml']),
	    ('share/aws_audibot/launch', [
        'launch/aws_audibot.launch.py',
        'launch/train_ddqn_with_track.launch.py'
	    ]),
	],



    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='azizi_',
    maintainer_email='azizi_@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)

# from setuptools import find_packages, setup
# from glob import glob

# package_name = 'aws_audibot'

# setup(
#     name=package_name,
#     version='0.0.0',
#     packages=find_packages(include=[package_name]),  # Adjust as needed
#     data_files=[
#         ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
#         ('share/' + package_name, ['package.xml']),
#         ('share/' + package_name + '/launch', ['launch/aws_audibot.launch.py']),
#     ],
#     install_requires=['setuptools'],
#     zip_safe=True,
#     maintainer='azizi_',
#     maintainer_email='azizi_@todo.todo',
#     description='TODO: Package description',  # Update as needed
#     license='TODO: License declaration',  # Update as needed
#     tests_require=['pytest'],
#     entry_points={
#         'console_scripts': [
#             'keyboard_control = audibot.keyboard_control:main',
#         ],
#     },
# )
