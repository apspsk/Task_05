#include "armor_detector/armor_detector.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <iomanip>

class ArmorDetectorNode : public rclcpp::Node
{
public:
    ArmorDetectorNode() : Node("armor_detector_node")
    {
        // 参数声明
        this->declare_parameter("use_camera", true);
        this->declare_parameter("video_path", "");
        this->declare_parameter("camera_topic", "/image_raw");
        this->declare_parameter("publish_rate", 30.0);
        this->declare_parameter("debug_mode", true);
        this->declare_parameter("confidence_threshold", 0.5);
        this->declare_parameter("enable_rviz", true);

        // 获取参数
        use_camera_ = this->get_parameter("use_camera").as_bool();
        video_path_ = this->get_parameter("video_path").as_string();
        camera_topic_ = this->get_parameter("camera_topic").as_string();
        publish_rate_ = this->get_parameter("publish_rate").as_double();
        debug_mode_ = this->get_parameter("debug_mode").as_bool();
        confidence_threshold_ = this->get_parameter("confidence_threshold").as_double();
        enable_rviz_ = this->get_parameter("enable_rviz").as_bool();

        // 初始化检测器
        detector_ = std::make_unique<ArmorDetector>();
        detector_->setConfidenceThreshold(confidence_threshold_);

        // 发布器
        if (enable_rviz_) {
            bbox_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("armor_bbox", 10);
            centers_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>("armor_center", 10);
        }
        
        // 统计信息发布器
        status_pub_ = this->create_publisher<std_msgs::msg::String>("armor_status", 10);

        // 根据模式选择输入源
        if (use_camera_) {
            // 订阅相机话题
            image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                camera_topic_, 10,
                std::bind(&ArmorDetectorNode::imageCallback, this, std::placeholders::_1));
            RCLCPP_INFO(this->get_logger(), "Using camera topic: %s", camera_topic_.c_str());
        } else {
            // 使用本地视频文件
            if (video_path_.empty()) {
                RCLCPP_ERROR(this->get_logger(), "Video path is empty when not using camera!");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Using video file: %s", video_path_.c_str());
            // 创建定时器处理视频帧
            video_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate_)),
                std::bind(&ArmorDetectorNode::videoTimerCallback, this));
            
            // 打开视频文件
            video_cap_.open(video_path_);
            if (!video_cap_.isOpened()) {
                RCLCPP_ERROR(this->get_logger(), "Cannot open video file: %s", video_path_.c_str());
                return;
            }
        }

        // 性能统计定时器
        stats_timer_ = this->create_wall_timer(
            std::chrono::seconds(2),
            std::bind(&ArmorDetectorNode::publishStats, this));

        RCLCPP_INFO(this->get_logger(), "Armor Detector Node Started");
        RCLCPP_INFO(this->get_logger(), "Parameters: use_camera=%s, debug_mode=%s, confidence_threshold=%.2f",
                   use_camera_ ? "true" : "false", debug_mode_ ? "true" : "false", confidence_threshold_);
    }

    // 初始化方法，在对象构造完成后调用
    void initialize()
    {
        // 安全使用 shared_from_this()
        image_transport_ = std::make_unique<image_transport::ImageTransport>(shared_from_this());
        if (debug_mode_) {
            debug_image_pub_ = image_transport_->advertise("debug_image", 1);
            RCLCPP_INFO(this->get_logger(), "Debug image publishing enabled");
        }
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            // 检查图像消息有效性
            if (msg->width <= 0 || msg->height <= 0) {
                RCLCPP_WARN(this->get_logger(), "Invalid image dimensions: %dx%d", msg->width, msg->height);
                return;
            }
            
            RCLCPP_DEBUG(this->get_logger(), "Received image: %dx%d, encoding: %s", 
                        msg->width, msg->height, msg->encoding.c_str());
            
            // 转换ROS图像消息为OpenCV格式（修复编码判断错误）
            cv_bridge::CvImagePtr cv_ptr;
            cv::Mat input_image;
            
            try {
                // 不强制编码格式，使用原始格式
                cv_ptr = cv_bridge::toCvCopy(msg);
                
                // 关键修复：使用字符串比较判断编码（避免依赖未定义的常量）
                if (msg->encoding == "bayer_rggb8") {
                    // 标准BAYER_RGGB8编码
                    cv::cvtColor(cv_ptr->image, input_image, cv::COLOR_BayerRGGB2BGR);
                } else if (msg->encoding == "bayer_rg8") {  // 相机实际发布的编码字符串
                    // 针对海康相机BAYER_RG8编码的转换
                    cv::cvtColor(cv_ptr->image, input_image, cv::COLOR_BayerRG2BGR);
                } else if (msg->encoding == sensor_msgs::image_encodings::MONO8) {
                    // 灰度图转BGR
                    cv::cvtColor(cv_ptr->image, input_image, cv::COLOR_GRAY2BGR);
                } else if (msg->encoding == sensor_msgs::image_encodings::RGB8) {
                    // 视频流RGB转BGR（避免红蓝颠倒）
                    cv::cvtColor(cv_ptr->image, input_image, cv::COLOR_RGB2BGR);
                } else if (msg->encoding == sensor_msgs::image_encodings::BGR8) {
                    input_image = cv_ptr->image.clone();
                } else {
                    RCLCPP_WARN(this->get_logger(), "Unsupported encoding: %s, trying BayerRG->BGR", msg->encoding.c_str());
                    cv::cvtColor(cv_ptr->image, input_image, cv::COLOR_BayerRG2BGR);
                }
            } catch (const cv_bridge::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge conversion failed: %s", e.what());
                return;
            }
            
            // 检查转换后的图像
            if (input_image.empty()) {
                RCLCPP_WARN(this->get_logger(), "Converted OpenCV image is empty");
                return;
            }
            
            // 缩放图像以减少处理负载
            cv::Mat processed_image;
            double scale_factor = 1.0;
            
            if (input_image.cols > 1500) {
                scale_factor = 0.5;  // 大图缩放到50%
            } else if (input_image.cols > 1000) {
                scale_factor = 0.75; // 中等图缩放到75%
            }
            
            cv::resize(input_image, processed_image, cv::Size(), scale_factor, scale_factor);
            
            RCLCPP_DEBUG(this->get_logger(), "Image scaled from %dx%d to %dx%d", 
                        input_image.cols, input_image.rows,
                        processed_image.cols, processed_image.rows);
            
            // 处理缩放后的图像
            processImage(processed_image, msg->header, scale_factor);
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in imageCallback: %s", e.what());
        }
    }

    void videoTimerCallback()
    {
        if (!video_cap_.isOpened()) return;

        cv::Mat frame;
        video_cap_ >> frame;
        
        if (frame.empty()) {
            RCLCPP_WARN(this->get_logger(), "Video frame is empty, video might have ended");
            video_timer_->cancel();
            return;
        }

        std_msgs::msg::Header header;
        header.stamp = this->now();
        header.frame_id = "video_frame";
        
        processImage(frame, header, 1.0);
    }

    void processImage(const cv::Mat& image, const std_msgs::msg::Header& header, double scale_factor = 1.0)
    {
        auto start_time = this->now();
        
        // 调用装甲板识别代码
        std::vector<ArmorPlate> detected_armors = detector_->detect(image);
        
        // 将检测结果坐标缩放回原图尺寸
        double inverse_scale = 1.0 / scale_factor;
        for (auto& armor : detected_armors) {
            armor.bounding_box.x *= inverse_scale;
            armor.bounding_box.y *= inverse_scale;
            armor.bounding_box.width *= inverse_scale;
            armor.bounding_box.height *= inverse_scale;
            armor.center.x *= inverse_scale;
            armor.center.y *= inverse_scale;
        }
        
        // 过滤低置信度的检测结果
        std::vector<ArmorPlate> filtered_armors;
        for (const auto& armor : detected_armors) {
            if (armor.confidence >= confidence_threshold_) {
                filtered_armors.push_back(armor);
            }
        }
        
        auto end_time = this->now();
        auto processing_time = (end_time - start_time).seconds() * 1000.0;
        
        // 更新统计信息
        total_frames_++;
        total_processing_time_ += processing_time;
        if (!filtered_armors.empty()) {
            frames_with_detections_++;
        }

        // 发布检测结果
        publishResults(filtered_armors, header);
        
        // 调试信息
        if (debug_mode_) {
            RCLCPP_DEBUG(this->get_logger(), "Detected %zu armors in %.2f ms", 
                        filtered_armors.size(), processing_time);
            
            // 发布调试图像
            publishDebugImage(image, filtered_armors, header, scale_factor);
        }

        // 定期输出检测状态
        static int frame_count = 0;
        if (++frame_count % 100 == 0) {
            RCLCPP_INFO(this->get_logger(), "Processed %d frames, Current FPS: %.2f, Detected armors: %zu", 
                       frame_count, 1000.0 / processing_time, filtered_armors.size());
        }
    }

    void publishResults(const std::vector<ArmorPlate>& armors, const std_msgs::msg::Header& header)
    {
        if (enable_rviz_) {
            // 发布边界框
            auto bbox_markers = createBoundingBoxMarkers(armors, header);
            bbox_pub_->publish(bbox_markers);

            // 发布所有检测到的装甲板中心点
            for (size_t i = 0; i < armors.size(); ++i) {
                auto center_msg = createCenterMessage(armors[i], header, i);
                centers_pub_->publish(center_msg);
            }
        }
    }

    visualization_msgs::msg::MarkerArray createBoundingBoxMarkers(
        const std::vector<ArmorPlate>& armors, const std_msgs::msg::Header& header)
    {
        visualization_msgs::msg::MarkerArray markers;
        
        // 先清除之前的标记
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.header = header;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        markers.markers.push_back(clear_marker);

        for (size_t i = 0; i < armors.size(); ++i) {
            const auto& armor = armors[i];
            
            // 边界框标记
            visualization_msgs::msg::Marker bbox_marker;
            bbox_marker.header = header;
            bbox_marker.ns = "armor_bbox";
            bbox_marker.id = i;
            bbox_marker.type = visualization_msgs::msg::Marker::CUBE;
            bbox_marker.action = visualization_msgs::msg::Marker::ADD;

            // 设置边界框位置和尺寸
            bbox_marker.pose.position.x = armor.center.x;
            bbox_marker.pose.position.y = armor.center.y;
            bbox_marker.pose.position.z = 0;
            bbox_marker.pose.orientation.w = 1.0;

            bbox_marker.scale.x = armor.bounding_box.width;
            bbox_marker.scale.y = armor.bounding_box.height;
            bbox_marker.scale.z = 1.0;

            // 设置颜色（根据置信度和顺序）
            if (i == 0) {
                bbox_marker.color.r = 0.0;  // 第一顺位：绿色
                bbox_marker.color.g = 1.0;
                bbox_marker.color.b = 0.0;
            } else if (i == 1) {
                bbox_marker.color.r = 1.0;  // 第二顺位：黄色
                bbox_marker.color.g = 1.0;
                bbox_marker.color.b = 0.0;
            } else if (i == 2) {
                bbox_marker.color.r = 1.0;  // 第三顺位：橙色
                bbox_marker.color.g = 0.5;
                bbox_marker.color.b = 0.0;
            } else {
                bbox_marker.color.r = 1.0;  // 其他：红色
                bbox_marker.color.g = 0.0;
                bbox_marker.color.b = 0.0;
            }
            bbox_marker.color.a = 0.3; // 半透明

            bbox_marker.lifetime = rclcpp::Duration::from_seconds(1.0 / publish_rate_ * 2);
            markers.markers.push_back(bbox_marker);

            // 添加文本标记显示标签和置信度
            visualization_msgs::msg::Marker text_marker;
            text_marker.header = header;
            text_marker.ns = "armor_labels";
            text_marker.id = i;
            text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::msg::Marker::ADD;
            
            text_marker.pose.position.x = armor.center.x;
            text_marker.pose.position.y = armor.center.y - armor.bounding_box.height / 2 - 10;
            text_marker.pose.position.z = 0;
            text_marker.pose.orientation.w = 1.0;
            
            text_marker.scale.z = 12.0; // 文字大小
            
            // 文本颜色与边界框一致
            text_marker.color.r = bbox_marker.color.r;
            text_marker.color.g = bbox_marker.color.g;
            text_marker.color.b = bbox_marker.color.b;
            text_marker.color.a = 1.0;
            
            std::stringstream ss;
            ss << armor.label << " Score:" << std::fixed << std::setprecision(2) << armor.confidence;
            text_marker.text = ss.str();
            
            text_marker.lifetime = rclcpp::Duration::from_seconds(1.0 / publish_rate_ * 2);
            markers.markers.push_back(text_marker);
        }

        return markers;
    }

    geometry_msgs::msg::PointStamped createCenterMessage(
        const ArmorPlate& armor, const std_msgs::msg::Header& header, int id)
    {
        geometry_msgs::msg::PointStamped center_msg;
        center_msg.header = header;
        center_msg.header.frame_id = "armor_" + std::to_string(id);
        center_msg.point.x = armor.center.x;
        center_msg.point.y = armor.center.y;
        center_msg.point.z = 0.0;
        return center_msg;
    }

    void publishDebugImage(const cv::Mat& image, const std::vector<ArmorPlate>& armors, 
                          const std_msgs::msg::Header& header, double scale_factor = 1.0)
    {
        // 确保 debug_image_pub_ 已经初始化
        if (!debug_image_pub_.getTopic().empty()) {
            cv::Mat debug_image = image.clone();
            
            // 在图像上绘制检测结果（修复：删除错误缩放）
            for (size_t i = 0; i < armors.size(); i++) {
                const auto& armor = armors[i];
                
                // 直接使用检测坐标（已基于缩放后图像计算）
                cv::Rect debug_bbox(
                    armor.bounding_box.x,
                    armor.bounding_box.y,
                    armor.bounding_box.width,
                    armor.bounding_box.height
                );
                
                cv::Point2f debug_center(
                    armor.center.x,
                    armor.center.y
                );
                
                cv::Scalar armorColor;
                
                // 根据锁定顺序使用不同颜色
                if (i == 0) {
                    armorColor = cv::Scalar(0, 255, 0); // 第一顺位：绿色
                } else if (i == 1) {
                    armorColor = cv::Scalar(0, 255, 255); // 第二顺位：黄色
                } else if (i == 2) {
                    armorColor = cv::Scalar(0, 165, 255); // 第三顺位：橙色
                } else {
                    armorColor = cv::Scalar(0, 0, 255); // 其他：红色
                }
                
                // 绘制装甲板矩形
                cv::rectangle(debug_image, debug_bbox, armorColor, 3);
                
                // 绘制中心点
                cv::circle(debug_image, debug_center, 5, cv::Scalar(0, 0, 255), -1);
                
                // 绘制十字准星在中心点
                cv::drawMarker(debug_image, debug_center, cv::Scalar(255, 0, 0), 
                              cv::MARKER_CROSS, 20, 2);
                
                // 显示装甲板信息
                std::string orderText = "Order:" + std::to_string(i + 1);
                std::string scoreText = "Score:" + std::to_string(armor.confidence);
                
                cv::Point textPos(debug_bbox.x, debug_bbox.y - 10);
                if (textPos.y < 20) textPos.y = debug_bbox.y + 20;
                
                cv::putText(debug_image, orderText, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.4, armorColor, 1);
                cv::putText(debug_image, scoreText, cv::Point(textPos.x, textPos.y + 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, armorColor, 1);
            }
            
            // 添加处理信息
            std::stringstream info_ss;
            info_ss << "Armors: " << armors.size();
            cv::putText(debug_image, info_ss.str(), 
                       cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            
            // 添加FPS信息
            if (total_frames_ > 0) {
                double avg_fps = 1000.0 / (total_processing_time_ / total_frames_);
                std::stringstream fps_ss;
                fps_ss << "FPS: " << std::fixed << std::setprecision(1) << avg_fps;
                cv::putText(debug_image, fps_ss.str(), 
                           cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            }

            // 转换为ROS消息并发布
            auto debug_msg = cv_bridge::CvImage(header, "bgr8", debug_image).toImageMsg();
            debug_image_pub_.publish(debug_msg);
        }
    }

    void publishStats()
    {
        if (total_frames_ > 0) {
            double avg_processing_time = total_processing_time_ / total_frames_;
            double avg_fps = 1000.0 / avg_processing_time;
            double detection_rate = (double)frames_with_detections_ / total_frames_ * 100.0;
            
            std::stringstream ss;
            ss << "Frames: " << total_frames_ 
               << " | Avg FPS: " << std::fixed << std::setprecision(1) << avg_fps
               << " | Avg Proc Time: " << std::fixed << std::setprecision(1) << avg_processing_time << "ms"
               << " | Detection Rate: " << std::fixed << std::setprecision(1) << detection_rate << "%";
            
            std_msgs::msg::String status_msg;
            status_msg.data = ss.str();
            status_pub_->publish(status_msg);
            
            RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
        }
    }

    // 成员变量
    std::unique_ptr<ArmorDetector> detector_;
    std::unique_ptr<image_transport::ImageTransport> image_transport_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr bbox_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr centers_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
    image_transport::Publisher debug_image_pub_;
    rclcpp::TimerBase::SharedPtr video_timer_;
    rclcpp::TimerBase::SharedPtr stats_timer_;
    
    cv::VideoCapture video_cap_;
    
    bool use_camera_;
    std::string video_path_;
    std::string camera_topic_;
    double publish_rate_;
    bool debug_mode_;
    bool enable_rviz_;
    double confidence_threshold_;
    
    // 性能统计
    int total_frames_ = 0;
    int frames_with_detections_ = 0;
    double total_processing_time_ = 0.0;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ArmorDetectorNode>();
    // 在对象完全构造后调用初始化
    node->initialize();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

