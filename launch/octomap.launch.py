import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

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

    # Ensure the output directory exists
    os.makedirs(output_bag_folder, exist_ok=True)

    # Define LaunchConfigurations
    input_bag_name = LaunchConfiguration('input_bag_name')

    # Derive the correct output directory and bag file name
    output_directory_name = PathJoinSubstitution([
        output_bag_folder,
        input_bag_name
    ])

    # Define a QoS profile with VOLATILE durability
    # qos_volatile = {
    #     'durability': 'volatile',
    #     'reliability': 'reliable',
    #     'depth': 10,
    # }

    return LaunchDescription([
        # Declare the input bag name argument
        input_bag_name_arg,

        # Play the input bag using ExecuteProcess
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', PathJoinSubstitution([bag_folder, input_bag_name]), '--clock'],
            output='screen',
            on_exit=[Shutdown()]
        ),

        # Static transform publisher with VOLATILE QoS
        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='os1_link_joint',
        #     arguments=['0.0', '0.0', '0.03618', '3.14159', '0', '0', 'os1_sensor', 'os1_lidar'],
        #     parameters=[{'qos_overrides./tf_static': qos_volatile}],
        #     on_exit=[Shutdown()]
        # ),

        # Odom to TF node with VOLATILE QoS
        Node(
            package='mapless_navigation',
            executable='odom2tf',
            name='odom2tf',
            # parameters=[{'qos_overrides./lidar_ground_truth': qos_volatile}],
            on_exit=[Shutdown()]
        ),

        # Octomap server node with VOLATILE QoS
        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            parameters=[
                {'resolution': 0.25},
                {'frame_id': 'world'},
                {'base_frame_id': 'imu_viz_link'},
                {'sensor_model/max_range': 7.0},
                {'sensor_model/min_range': 0.0},
                {'sensor_model/hit': 0.95},
                {'sensor_model/miss': 0.45},
                # {'qos_overrides./cloud_in': qos_volatile}
            ],
            remappings=[
                ('cloud_in', '/os1_cloud_node/points')
            ],
            on_exit=[Shutdown()]
        ),

        # Record the rosbag using ExecuteProcess
        ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-o', output_directory_name,
                 '/tf', '/tf_static', '/lidar_ground_truth',
                 '/lidar_filtered/occupied_cells_vis_array', '/lidar_filtered/octomap_full',
                 '/lidar_filtered/octomap_point_cloud_centers', '/lidar_filtered/octomap_server/parameter_descriptions',
                 '/lidar_filtered/octomap_server/parameter_updates', '/lidar_filtered/projected_map'],
            output='screen',
            on_exit=[Shutdown()]
        ),
    ])
