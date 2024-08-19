from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            namespace='lidar',
            parameters=[
                {'resolution': 0.25},
                {'frame_id': 'world'},
                {'base_frame_id': 'world'},
                {'sensor_model/max_range': 7.0},
                {'sensor_model/min_range': 0.0},
                {'sensor_model/hit': 0.95},
                {'sensor_model/miss': 0.48}
            ],
            remappings=[
                ('cloud_in', '/os1_cloud_node/points')
            ]
        ),

        Node(
            package='isrr_analysis',
            executable='point_cloud_angle_filter',
            name='point_cloud_filter',
            output='screen',
            remappings=[
                ('in_cloud', '/os1_cloud_node/points'),
                ('filterd_cloud', '/os1_cloud_node/points/filtered')
            ]
        ),

        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server_filtered',
            namespace='lidar_filtered',
            parameters=[
                {'resolution': 0.25},
                {'frame_id': 'world'},
                {'base_frame_id': 'world'},
                {'sensor_model/max_range': 7.0},
                {'sensor_model/min_range': 0.0},
                {'sensor_model/hit': 0.95},
                {'sensor_model/miss': 0.48}
            ],
            remappings=[
                ('cloud_in', '/os1_cloud_node/points/filtered')
            ]
        )
    ])
