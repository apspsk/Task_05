#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            name='debug_mode',
            default_value='true',
            description='Enable debug image publishing'
        ),
        
        DeclareLaunchArgument(
            name='confidence_threshold',
            default_value='0.5',
            description='Confidence threshold for armor detection'
        ),
        
        # 装甲板检测节点
        Node(
            package='armor_detector',
            executable='armor_detector_node',
            name='armor_detector',
            output='screen',
            parameters=[{
                'use_camera': False,
                'video_path': '/path/to/your/test/video.mp4',  # 修改为你的测试视频路径
                'debug_mode': LaunchConfiguration('debug_mode'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'publish_rate': 30.0
            }]
        )
    ])
