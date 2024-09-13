import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, Shutdown, RegisterEventHandler, OpaqueFunction
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def ensure_output_directory_exists(context, output_folder):
    input_bag_name = LaunchConfiguration('input_bag_name').perform(context)
    output_directory_name = os.path.join(output_folder, input_bag_name)
    if not os.path.exists(output_directory_name):
        os.makedirs(output_directory_name)
        print(f"Created output directory: {output_directory_name}")
    else:
        print(f"Output directory already exists: {output_directory_name}")
    return []


def generate_launch_description():
    # Declare the input bag argument
    input_bag_name_arg = DeclareLaunchArgument(
        'input_bag_name',
        default_value='ec_hallways_run0',
        description='Name of the input ROS2 bag file without extension.'
    )

    # Directories
    bag_folder = os.path.join(os.environ['HOME'], 'coloradar/bags2')
    output_bag_folder = os.path.join(os.environ['HOME'], 'coloradar/octomap_bags2')
    os.makedirs(output_bag_folder, exist_ok=True)
    input_bag_name = LaunchConfiguration('input_bag_name')
    output_file_name = PathJoinSubstitution([output_bag_folder, input_bag_name, 'octomap.bt'])

    launch_file_dir = os.path.dirname(__file__)
    qos_profile_path = os.path.join(launch_file_dir, 'reliability_override.yaml')

    return LaunchDescription([
        input_bag_name_arg,
        OpaqueFunction(function=ensure_output_directory_exists, args=[output_bag_folder]),

        # Play the input bag using ExecuteProcess with Transient QoS and QoS profile override
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', PathJoinSubstitution([bag_folder, input_bag_name]), '--clock', '--qos-profile-overrides-path', qos_profile_path],
            output='screen',
            on_exit=[Shutdown()]
        ),

        # Static transform publisher with Transient QoS
        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='os1_link_joint',
        #     arguments=['0.0', '0.0', '0.03618', '0', '0', '0', '1', 'os1_sensor', 'os1_lidar'],
        #     parameters=[{
        #         'qos_overrides./tf_static': {
        #             'durability': 'transient_local',
        #             'reliability': 'reliable',
        #             'depth': 1
        #         }
        #     }],
        #     on_exit=[Shutdown()]
        # ),

        # Odom to TF node with Transient QoS
        Node(
            package='mapless_navigation',
            executable='odom2tf',
            name='odom2tf',
            on_exit=[Shutdown()]
        ),

        # Octomap server node with Transient QoS
        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            parameters=[
                {'resolution': 0.25},
                {'frame_id': 'world'},
                {'base_frame_id': 'world'},
                {'sensor_model/max_range': 7.0},
                {'sensor_model/min_range': 0.0},
                {'sensor_model/hit': 0.95},
                {'sensor_model/miss': 0.45}
            ],
            remappings=[
                ('cloud_in', '/os1_cloud_node/points')
            ],
            on_exit=[Shutdown()]
        ),

        # Save the octomap to a file after the octomap server shuts down
        RegisterEventHandler(
            OnProcessExit(
                target_action=Node(package='octomap_server', executable='octomap_server_node', name='octomap_server'),
                on_exit=[ExecuteProcess(
                    cmd=['ros2', 'service', 'call', '/octomap_server/save_map', 'octomap_msgs/srv/SaveMap',
                         f'{{filename: "{output_file_name}"}}'],
                    output='screen'
                )]
            )
        )
    ])
