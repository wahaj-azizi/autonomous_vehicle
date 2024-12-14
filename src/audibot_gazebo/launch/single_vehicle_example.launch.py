import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # gazebo_simulator = IncludeLaunchDescription(   #LAUNCHING GAZEBO TWICE, ALSO COMMENTED AT LINE 41
    #     PythonLaunchDescriptionSource([
    #         os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
    #     ])
    # )

    audibot_options = dict(
        start_x = '0.0',
        start_y = '-15.0',
        start_z = '0',
        start_yaw = '0',
        pub_tf = 'true',
        tf_freq = '100.0',
        blue = 'false'
    )

    spawn_audibot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('audibot_gazebo'), 'launch', 'audibot_robot.launch.py')
        ]),
        launch_arguments=audibot_options.items()
    )

    threshold = Node(
        package='ros2_img_threshold',
        executable='img_threshold_node'
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='single_vehicle_viz',
        arguments=['-d', os.path.join(get_package_share_directory('audibot_gazebo'), 'rviz', 'single_vehicle_example.rviz')]
    )

    return LaunchDescription([
       # gazebo_simulator,
        spawn_audibot,
        threshold,
        rviz
    ])
