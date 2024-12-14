from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Paths to other launch files
    aws_audibot_launch_path = os.path.join(
        get_package_share_directory('aws_audibot'),
        'launch',
        'aws_audibot.launch.py'
    )

    return LaunchDescription([
        # Launch the Gazebo track with the 'audibot'
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(aws_audibot_launch_path),
            launch_arguments={}.items()
        ),

        # Run the keyboard control
        ExecuteProcess(
            cmd=['ros2', 'run', 'aws_audibot_controller', 'keyboard_control'],
            output='screen'
        ),

        # Define arguments for DDQN training
        DeclareLaunchArgument('vehicle_name', default_value='audibot', description='Name of the vehicle'),
        DeclareLaunchArgument('learning_rate', default_value='0.001', description='Learning rate for the DDQN agent'),
        DeclareLaunchArgument('gamma', default_value='0.99', description='Discount factor for future rewards'),
        DeclareLaunchArgument('epsilon', default_value='1.0', description='Initial exploration rate'),
        DeclareLaunchArgument('epsilon_min', default_value='0.01', description='Minimum exploration rate'),
        DeclareLaunchArgument('epsilon_decay', default_value='0.995', description='Decay rate for epsilon'),
        DeclareLaunchArgument('batch_size', default_value='32', description='Batch size for training'),
        DeclareLaunchArgument('replay_buffer_size', default_value='10000', description='Size of the replay buffer'),
        DeclareLaunchArgument('target_update_interval', default_value='100', description='Frequency of target network updates'),
        DeclareLaunchArgument('num_episodes', default_value='1000', description='Number of training episodes'),

        # Node to launch the DDQN training script
        Node(
            package='ddqn_audibot',
            executable='train_ddqn',
            name='train_ddqn_node',
            output='screen',
            parameters=[
                {'vehicle_name': LaunchConfiguration('vehicle_name')},
                {'learning_rate': LaunchConfiguration('learning_rate')},
                {'gamma': LaunchConfiguration('gamma')},
                {'epsilon': LaunchConfiguration('epsilon')},
                {'epsilon_min': LaunchConfiguration('epsilon_min')},
                {'epsilon_decay': LaunchConfiguration('epsilon_decay')},
                {'batch_size': LaunchConfiguration('batch_size')},
                {'replay_buffer_size': LaunchConfiguration('replay_buffer_size')},
                {'target_update_interval': LaunchConfiguration('target_update_interval')},
                {'num_episodes': LaunchConfiguration('num_episodes')}
            ],
            remappings=[
                # Add any topic remappings if needed
            ]
        ),
    ])
