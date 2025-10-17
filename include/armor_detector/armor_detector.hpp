#ifndef ARMOR_DETECTOR_HPP
#define ARMOR_DETECTOR_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// 装甲板检测结果结构体
struct ArmorPlate
{
    cv::Rect bounding_box;
    cv::Point2f center;
    float confidence;
    std::string label;

    ArmorPlate() : confidence(0.0f), label("unknown") {}
    ArmorPlate(const cv::Rect& bbox, float conf, const std::string& lbl = "armor")
        : bounding_box(bbox), center(bbox.x + bbox.width/2, bbox.y + bbox.height/2), 
          confidence(conf), label(lbl) {}
};

class ArmorDetector
{
public:
    ArmorDetector();
    ~ArmorDetector();

    // 装甲板检测主函数 - 直接使用你的算法
    std::vector<ArmorPlate> detect(const cv::Mat& image);
    
    // 设置参数
    void setConfidenceThreshold(float threshold) { confidence_threshold_ = threshold; }

private:
    float confidence_threshold_ = 0.5f;
};

#endif // ARMOR_DETECTOR_HPP


