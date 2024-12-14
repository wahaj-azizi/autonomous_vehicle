#from launch import LaunchDescription
#from launch_ros.actions import Node
#from launch.actions import IncludeLaunchDescription
#from launch.launch_description_sources import PythonLaunchDescriptionSource
#import os
#
#def generate_launch_description():
#    # Paths to the RoboMaker and Audibot launch files
#    robomaker_launch = os.path.join(
#        os.getenv('COLCON_PREFIX_PATH').split(':')[0],
#        'aws_robomaker_track', 'launch', '/home/azizi_/aws_audibot_ws/src/aws-robomaker-racetrack-world/launch/racetrack.launch.py', 'racetrack.launch.py'
#    )
#
#    audibot_launch = os.path.join(
#        os.getenv('COLCON_PREFIX_PATH').split(':')[0],
#        'audibot_gazebo', 'launch', 'audibot_robot.launch.py'
#    )
#
#    return LaunchDescription([
#        IncludeLaunchDescription(
#            PythonLaunchDescriptionSource(robomaker_launch)
#        ),
#        IncludeLaunchDescription(
#            PythonLaunchDescriptionSource(audibot_launch)
#        ),
#    ])

#from launch import LaunchDescription
#from launch_ros.actions import Node
#from launch.actions import IncludeLaunchDescription
#from launch.launch_description_sources import PythonLaunchDescriptionSource
#import os
#
#def generate_launch_description():
#    # Paths to the RoboMaker and Audibot launch files
#    robomaker_launch = os.path.join(
#        os.getenv('COLCON_PREFIX_PATH').split(':')[0],
#        'aws_robomaker_racetrack_world', 'launch', 'racetrack.launch.py'
#    )
#
#    audibot_launch = os.path.join(
#        os.getenv('COLCON_PREFIX_PATH').split(':')[0],
#        'audibot_gazebo', 'launch', 'audibot_robot.launch.py'
#    )
#
#    return LaunchDescription([
#        IncludeLaunchDescription(
#            PythonLaunchDescriptionSource(robomaker_launch)
#        ),
#        IncludeLaunchDescription(
#            PythonLaunchDescriptionSource(audibot_launch)
#        ),
#    ])
#
#from launch import LaunchDescription
#from launch_ros.actions import Node
#from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
#from launch.launch_description_sources import PythonLaunchDescriptionSource
#from ament_index_python.packages import get_package_share_directory
#import os
#
#def generate_launch_description():
#    # Paths to the necessary launch files
#    racetrack_launch = os.path.join(
#        get_package_share_directory('aws-robomaker-racetrack-world'),
#        'launch',
#        'racetrack.launch.py'
#    )
#
#    audibot_gazebo_launch = os.path.join(
#        get_package_share_directory('audibot_gazebo'),
#        'launch',
#        'gazebo.launch.py'
#    )
#
#    return LaunchDescription([
#        DeclareLaunchArgument('mode', default_value='default', description='Mode of operation'),
#        IncludeLaunchDescription(
#            PythonLaunchDescriptionSource(racetrack_launch),
#        ),
#        IncludeLaunchDescription(
##            PythonLaunchDescriptionSource(audibot_gazebo_launch),
#            PathJoinSubstitution(
#                [FindPackageShare('audibot_gazebo'), 'launch', 'single_vehicle_example.launch.py']
#    )
#)        ,
#    ])
#
#
#from launch import LaunchDescription
#from launch_ros.actions import Node
#from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
#from launch.launch_description_sources import PythonLaunchDescriptionSource
#from ament_index_python.packages import get_package_share_directory
#from launch.substitutions import PathJoinSubstitution
#import os
#
#def generate_launch_description():
#    # Paths to the necessary launch files
#    racetrack_launch = os.path.join(
#        get_package_share_directory('aws-robomaker-racetrack-world'),
#        'launch',
#        'view_racetrack.launch.py'
#        #'racetrack_day_empty.world'
#    )
#
#    audibot_gazebo_launch = PathJoinSubstitution(
#        [get_package_share_directory('audibot_gazebo'), 'launch', 'single_vehicle_example.launch.py']
#    )
#
#    return LaunchDescription([
#        DeclareLaunchArgument('mode', default_value='default', description='Mode of operation'),
#        IncludeLaunchDescription(
#            PythonLaunchDescriptionSource(racetrack_launch),
#        ),
#        IncludeLaunchDescription(
#            PythonLaunchDescriptionSource(audibot_gazebo_launch),
#        ),
#    ])
#
# from launch import LaunchDescription
# from launch_ros.actions import Node
# from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from ament_index_python.packages import get_package_share_directory
# from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
# import os

# def generate_launch_description():
#     # Paths to the necessary launch files
#     racetrack_launch = os.path.join(
#         get_package_share_directory('aws-robomaker-racetrack-world'),
#         'launch',
#         'view_racetrack.launch.py'
#     )

#     audibot_gazebo_launch = PathJoinSubstitution(
#         [get_package_share_directory('audibot_gazebo'), 'launch', 'single_vehicle_example.launch.py']
#     )

#     # Declare arguments for the spawning position
#     declare_x_arg = DeclareLaunchArgument('x', default_value='2.75', description='Initial x position')
#     declare_y_arg = DeclareLaunchArgument('y', default_value='-14.00', description='Initial y position')
#     declare_z_arg = DeclareLaunchArgument('z', default_value='0.0', description='Initial z position')

#     return LaunchDescription([
#         declare_x_arg,
#         declare_y_arg,
#         declare_z_arg,
#         DeclareLaunchArgument('mode', default_value='default', description='Mode of operation'),
#         IncludeLaunchDescription(
#             PythonLaunchDescriptionSource(racetrack_launch),
#         ),
#         IncludeLaunchDescription(
#             PythonLaunchDescriptionSource(audibot_gazebo_launch),
#             launch_arguments={
#                 'x': LaunchConfiguration('x'),
#                 'y': LaunchConfiguration('y'),
#                 'z': LaunchConfiguration('z')
#             }.items(),
#         ),
#     ])


# from launch import LaunchDescription
# from launch_ros.actions import Node
# from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from ament_index_python.packages import get_package_share_directory
# from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
# import os

# def generate_launch_description():
#    # Paths to the necessary launch files
#    racetrack_launch = os.path.join(
#        get_package_share_directory('aws-robomaker-racetrack-world'),
#        'launch',
#        'view_racetrack.launch.py'
#    )

#    audibot_gazebo_launch = PathJoinSubstitution(
#        [get_package_share_directory('audibot_gazebo'), 'launch', 'single_vehicle_example.launch.py']
#    )

#    # Declare arguments for the spawning position
#    declare_x_arg = DeclareLaunchArgument('x', default_value='0.0', description='Initial x position')
#    declare_y_arg = DeclareLaunchArgument('y', default_value='0.0', description='Initial y position')
#    declare_z_arg = DeclareLaunchArgument('z', default_value='0.0', description='Initial z position')

#    return LaunchDescription([
#        declare_x_arg,
#        declare_y_arg,
#        declare_z_arg,
#        DeclareLaunchArgument('mode', default_value='default', description='Mode of operation'),
#        IncludeLaunchDescription(
#            PythonLaunchDescriptionSource(racetrack_launch),
#        ),
#        IncludeLaunchDescription(
#            PythonLaunchDescriptionSource(audibot_gazebo_launch),
#            launch_arguments={
#                'x': LaunchConfiguration('x'),
#                'y': LaunchConfiguration('y'),
#                'z': LaunchConfiguration('z')
#            }.items(),
#        ),
#    ])
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
import os

def generate_launch_description():
    # Paths to the necessary launch files
    racetrack_launch = os.path.join(
        get_package_share_directory('aws-robomaker-racetrack-world'),
        'launch',
        'view_racetrack.launch.py'
    )

    audibot_gazebo_launch = PathJoinSubstitution(
        [get_package_share_directory('audibot_gazebo'), 'launch', 'single_vehicle_example.launch.py']
    )

    # Declare arguments for the spawning position
    declare_x_arg = DeclareLaunchArgument('x', default_value='-5.5', description='Initial x position')
    declare_y_arg = DeclareLaunchArgument('y', default_value='-15.0', description='Initial y position')
    declare_z_arg = DeclareLaunchArgument('z', default_value='0.0', description='Initial z position')

    return LaunchDescription([
        declare_x_arg,
        declare_y_arg,
        declare_z_arg,
        DeclareLaunchArgument('mode', default_value='default', description='Mode of operation'),

        # Include the racetrack launch to start Gazebo and load the racetrack
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(racetrack_launch),
        ),
        
        # Include the audibot launch to spawn the model into the already running Gazebo instance
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(audibot_gazebo_launch),
            launch_arguments={
                'x': LaunchConfiguration('x'),
                'y': LaunchConfiguration('y'),
                'z': LaunchConfiguration('z')
            }.items(),
        ),
    ])
