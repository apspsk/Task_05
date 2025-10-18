#ifndef ARMOR_DETECTOR_HPP
#define ARMOR_DETECTOR_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// 装甲板检测结果结构体
struct ArmorPlate
{
    cv::Rect bounding_box;       // 轴对齐矩形框
    cv::Point2f center;          // 中心点坐标
    float confidence;            // 匹配置信度
    std::string label;           // 装甲板标签
    cv::Point2f vertices[4];     // 新增：装甲板4个顶点（用于RViz显示旋转矩形）

    // 构造函数补充顶点初始化
    ArmorPlate() : confidence(0.0f), label("unknown") {
        for (int i = 0; i < 4; ++i) {
            vertices[i] = cv::Point2f(0, 0);
        }
    }
    ArmorPlate(const cv::Rect& bbox, float conf, const std::string& lbl = "armor")
        : bounding_box(bbox), center(bbox.x + bbox.width/2, bbox.y + bbox.height/2), 
          confidence(conf), label(lbl) {
        for (int i = 0; i < 4; ++i) {
            vertices[i] = cv::Point2f(0, 0);
        }
    }
};

class ArmorDetector {
private:
    // 新增：置信度阈值成员变量
    float confidence_threshold_ = 0.5f;  // 默认值0.5

    // 其他已有成员（几何变换参数等）
	float tx_ = 224.0f;  // x轴右移224
	float ty_ = 142.0f;  // y轴下移142
	float sx_ = 1.0f;    // 无缩放
	float sy_ = 1.0f;    // 无缩放
	float rot_ = 0.0f;   // 无旋转


public:
    // 其他已有函数声明
    ArmorDetector();
    ~ArmorDetector();
    std::vector<ArmorPlate> detect(const cv::Mat& image);
    void setParams(float tx, float ty, float sx, float sy, float rot);

    // 新增：设置置信度阈值的函数声明
    void setConfidenceThreshold(float threshold) {
        confidence_threshold_ = threshold;
    }
};



#endif // ARMOR_DETECTOR_HPP


