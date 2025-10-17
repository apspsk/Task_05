from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, EqualsSubstitution
from launch.conditions import IfCondition  # Humble中IfCondition的正确导入路径
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取包的路径
    package_dir = get_package_share_directory('armor_detector')
    
    # 定义RViz配置文件路径
    rviz_config_path = os.path.join(package_dir, 'config', 'armor_detector.rviz')
    
    # 声明启动参数
    use_camera_arg = DeclareLaunchArgument(
        'use_camera',
        default_value='true',
        description='Whether to use camera or video file'
    )
    video_path_arg = DeclareLaunchArgument(
        'video_path',
        default_value=os.path.join(package_dir, 'videos', 'test.mp4'),
        description='Path to video file if not using camera'
    )
    debug_mode_arg = DeclareLaunchArgument(
        'debug_mode',
        default_value='true',
        description='Enable debug mode with image visualization'
    )
    enable_rviz_arg = DeclareLaunchArgument(
        'enable_rviz',
        default_value='true',
        description='Whether to launch RViz'
    )
    
    # 相机节点
    hik_camera_node = Node(
        package='armor_detector',
        executable='hik_camera_node',
        name='hik_camera_node',
        output='screen',
        parameters=[os.path.join(package_dir, 'config', 'armor_detector.yaml')]
    )
    
    # 装甲板检测节点
    armor_detector_node = Node(
        package='armor_detector',
        executable='armor_detector_node',
        name='armor_detector_node',
        output='screen',
        parameters=[
            {
                'use_camera': LaunchConfiguration('use_camera'),
                'video_path': LaunchConfiguration('video_path'),
                'debug_mode': LaunchConfiguration('debug_mode'),
                'enable_rviz': LaunchConfiguration('enable_rviz'),
                'confidence_threshold': 0.5
            }
        ]
    )
    
    # RViz节点（适配Humble的条件判断）
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path],
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration('enable_rviz'), 'true'))
    )
    
    return LaunchDescription([
        use_camera_arg,
        video_path_arg,
        debug_mode_arg,
        enable_rviz_arg,
        hik_camera_node,
        armor_detector_node,
        rviz_node
    ])

