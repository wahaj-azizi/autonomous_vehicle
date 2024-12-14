# from setuptools import find_packages, setup

# package_name = 'ddqn_audibot'

# setup(
#     name=package_name,
#     version='0.0.0',
#     packages=find_packages(exclude=['test']),
#     data_files=[
#         ('share/ament_index/resource_index/packages',
#             ['resource/' + package_name]),
#         ('share/' + package_name, ['package.xml']),
#     ],
#     install_requires=['setuptools'],
#     zip_safe=True,
#     maintainer='azizi_',
#     maintainer_email='azizi_@todo.todo',
#     description='TODO: Package description',
#     license='TODO: License declaration',
#     tests_require=['pytest'],
#     entry_points={
#         'console_scripts': [
#         ],
#     },
# )
# from setuptools import setup

# package_name = 'ddqn_audibot'

# setup(
#     name=package_name,
#     version='0.0.0',
#     packages=[package_name],
#     install_requires=['setuptools'],
#     zip_safe=False,
#     data_files=[
#         ('share/ament_index/resource/index/packages', ['resource/' + package_name]),
#         ('share/' + package_name, ['launch/train_ddqn_launch.py']),
#     ],
#     package_data={
#         package_name: ['launch/*.py'],
#     },
# )
# from setuptools import setup
# from glob import glob
# import os

# package_name = 'ddqn_audibot'

# setup(
#     name=package_name,
#     version='0.0.0',
#     packages=[package_name],
#     data_files=[
#         ('share/ament_index/resource_index/packages',
#          ['resource/' + package_name]),
#         ('share/' + package_name, ['package.xml']),
#         ('share/' + package_name + '/launch', glob('launch/*.py')),
#     ],
#     install_requires=['setuptools'],
#     zip_safe=True,
#     maintainer='azizi_',
#     maintainer_email='azizi_@todo.todo',
#     description='TODO: Package description',
#     license='TODO: License declaration',
#     tests_require=['pytest'],
#     entry_points={
#         'console_scripts': [],
#     },
# )
from setuptools import setup
import os
from glob import glob

package_name = 'ddqn_audibot'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Wahaj Azizi',
    maintainer_email='wahaj.azizi@example.com',
    description='DDQN agent for controlling audibot',
    license='MIT',
    tests_require=['pytest'],
    # entry_points={
    #     'console_scripts': [
    #         'ddqn_agent = ddqn_audibot.ddqn_agent:main',
    #     ],
    entry_points={
    'console_scripts': [
        'ddqn_agent = ddqn_audibot.ddqn_agent:main',
    ],
},
)

