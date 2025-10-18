#include "armor_detector/armor_detector.hpp"
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <set>
#include <map>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/camera_info.hpp>

using namespace cv;
using namespace std;

// 灯条结构体
struct LightBar {
    RotatedRect rect;
    float area;
    float length;
    float angle;
    float brightness;
    string color;
    float colorScore; // 颜色得分
    Point2f topCenter;    // 灯条上端中心
    Point2f bottomCenter; // 灯条下端中心
    Point2f verticalCenter; // 灯条竖直中线
    bool isMatched; // 是否已被匹配
    int id; // 灯条唯一标识
};

// 装甲板结构体
struct ArmorPlateInternal {
    Point2f vertices[4];
    Rect boundingRect;
    vector<LightBar> matchedLightBars; // 匹配的灯条
    bool isSimilarParallel; // 是否为相似平行灯条
    bool isSmallArmor; // 是否为小装甲板
    float matchScore; // 匹配得分
    float rectangleSimilarity; // 矩形相似度得分
    int armorId; // 装甲板唯一标识
    float aspectRatio; // 基于中线端点的长宽比
};

// 计算灯条的竖直中线
void calculateLightBarGeometry(LightBar& lightBar) {
    Point2f points[4];
    lightBar.rect.points(points);
    
    // 找到灯条的上下端点
    vector<Point2f> sortedPoints(points, points + 4);
    sort(sortedPoints.begin(), sortedPoints.end(), [](const Point2f& a, const Point2f& b) {
        return a.y < b.y;
    });
    
    // 上端中心点（两个较高的点取平均）
    Point2f top1 = sortedPoints[0];
    Point2f top2 = sortedPoints[1];
    lightBar.topCenter = (top1 + top2) * 0.5;
    
    // 下端中心点（两个较低的点取平均）
    Point2f bottom1 = sortedPoints[2];
    Point2f bottom2 = sortedPoints[3];
    lightBar.bottomCenter = (bottom1 + bottom2) * 0.5;
    
    // 竖直中线（上下端点的中点）
    lightBar.verticalCenter = (lightBar.topCenter + lightBar.bottomCenter) * 0.5;
}

// 预处理函数 - 优化二值化参数
Mat preprocessImage(const Mat& input) {
    // 检查输入图像有效性
    if (input.empty()) {
        RCLCPP_ERROR(rclcpp::get_logger("ArmorDetector"), "Input image is empty!");
        return Mat();
    }
    
    if (input.cols <= 0 || input.rows <= 0) {
        RCLCPP_ERROR(rclcpp::get_logger("ArmorDetector"), "Invalid image dimensions: %dx%d", input.cols, input.rows);
        return Mat();
    }
    
    RCLCPP_DEBUG(rclcpp::get_logger("ArmorDetector"), "Preprocessing image: %dx%d", input.cols, input.rows);
    
    Mat gray, blurred, binary;
    
    vector<Mat> channels;
    split(input, channels);
    
    // 增强红色和蓝色通道
    Mat enhanced_gray = 0.5 * channels[2] + 0.4 * channels[0] + 0.1 * channels[1];
    
    // 中值滤波去噪 - 根据图像大小调整滤波大小
    int filterSize = 5;
    if (input.cols < 400 || input.rows < 300) {
        filterSize = 3; // 小图像使用较小的滤波器
    }
    medianBlur(enhanced_gray, blurred, filterSize);

    // 多种阈值方法结合 - 调整阈值参数
    Mat binary_low;
    threshold(blurred, binary_low, 190, 255, THRESH_BINARY); // 降低低阈值
    
    Mat binary_high;
    threshold(blurred, binary_high, 240, 255, THRESH_BINARY); // 稍微降低高阈值
    
    Mat binary_otsu;
    threshold(blurred, binary_otsu, 0, 255, THRESH_BINARY + THRESH_OTSU);
    
    double otsu_threshold = threshold(blurred, binary_otsu, 0, 255, THRESH_BINARY + THRESH_OTSU);
    // 修改OTSU阈值后处理逻辑
    if (otsu_threshold < 200) { // 降低OTSU阈值下限
        threshold(blurred, binary_otsu, max(150.0, otsu_threshold), 255, THRESH_BINARY);
    }
    
    // 合并二值化结果
    Mat binary_combined;
    bitwise_or(binary_otsu, binary_high, binary_combined);
    bitwise_or(binary_combined, binary_low, binary_combined);
    
    // 形态学操作 - 调整参数以适应正方形灯条
    Mat kernel_erode = getStructuringElement(MORPH_RECT, Size(1, 1)); // 减小腐蚀核
    erode(binary_combined, binary, kernel_erode);
    
    // 增加膨胀操作连接相近区域
    Mat kernel_dilate = getStructuringElement(MORPH_RECT, Size(2, 2));
    dilate(binary, binary, kernel_dilate);
    
    return binary;
}

// 计算白色比例（在二值图下）
float calculateWhiteRatio(const Mat& binaryRoi) {
    if (binaryRoi.empty() || binaryRoi.rows < 3 || binaryRoi.cols < 3) return 0;
    
    // 在二值图中，白色像素值为255
    int white_pixels = countNonZero(binaryRoi);
    int total_pixels = binaryRoi.rows * binaryRoi.cols;
    
    if (total_pixels == 0) return 0;
    
    return static_cast<float>(white_pixels) / total_pixels;
}

// 计算颜色得分
float calculateColorScore(const Mat& roi, const Point2f& center, const Mat& original) {
    if (roi.empty()) return 0;
    
    int x = max(0, min(original.cols-1, (int)center.x));
    int y = max(0, min(original.rows-1, (int)center.y));
    
    Rect sampleRect(max(0, x-1), max(0, y-1), 3, 3);
    sampleRect &= Rect(0, 0, original.cols, original.rows);
    
    if (sampleRect.width < 3 || sampleRect.height < 3) return 0;
    
    Mat sample = original(sampleRect);
    Scalar meanColor = mean(sample);
    double blue = meanColor[0];
    double green = meanColor[1];
    double red = meanColor[2];
    
    double whiteScore = min(red, min(green, blue)) / 255.0;
    
    double blueRedDominance = 0;
    if (blue > green + 30 && blue > red + 20) {
        blueRedDominance = (blue - max(green, red)) / 255.0;
    } else if (red > green + 30 && red > blue + 20) {
        blueRedDominance = (red - max(green, blue)) / 255.0;
    }
    
    float colorScore = whiteScore * 0.6 + blueRedDominance * 0.4;
    return colorScore;
}

// 计算矩形相似度得分
float calculateRectangleSimilarity(const vector<Point2f>& vertices) {
    if (vertices.size() != 4) return 0.0f;
    
    // 计算四边形的面积
    vector<Point2f> hull;
    convexHull(vertices, hull);
    float area = contourArea(hull);
    
    // 计算最小外接矩形面积
    RotatedRect minRect = minAreaRect(vertices);
    float rectArea = minRect.size.width * minRect.size.height;
    
    if (rectArea < 1e-5) return 0.0f;
    
    // 面积比越接近1，越接近矩形
    float areaRatio = area / rectArea;
    
    // 计算角度接近直角的程度
    vector<float> angles;
    for (int i = 0; i < 4; i++) {
        Point2f v1 = vertices[i] - vertices[(i+3)%4];
        Point2f v2 = vertices[(i+1)%4] - vertices[i];
        float dot = v1.x * v2.x + v1.y * v2.y;
        float len1 = sqrt(v1.x*v1.x + v1.y*v1.y);
        float len2 = sqrt(v2.x*v2.x + v2.y*v2.y);
        if (len1 < 1e-5 || len2 < 1e-5) continue;
        float angle = acos(dot / (len1 * len2)) * 180 / CV_PI;
        angles.push_back(angle);
    }
    
    float angleScore = 0.0f;
    if (!angles.empty()) {
        float angleDiffSum = 0.0f;
        for (float angle : angles) {
            float diff = abs(angle - 90.0f); // 与90度直角的差异
            angleDiffSum += max(0.0f, 1.0f - diff / 45.0f); // 差异小于45度给分
        }
        angleScore = angleDiffSum / angles.size();
    }
    
    // 综合得分：面积比占60%，角度得分占40%
    float similarity = areaRatio * 0.6f + angleScore * 0.4f;
    return similarity;
}

// 灯条检测函数 - 修改长宽比和形状筛选条件
vector<LightBar> detectLightBarsByBrightness(const Mat& binary, const Mat& original) {
    vector<LightBar> lightBars;
    
    // 检查输入图像有效性
    if (binary.empty() || original.empty()) {
        RCLCPP_ERROR(rclcpp::get_logger("ArmorDetector"), "Input images are empty!");
        return lightBars;
    }
    
    if (binary.cols <= 0 || binary.rows <= 0 || original.cols <= 0 || original.rows <= 0) {
        RCLCPP_ERROR(rclcpp::get_logger("ArmorDetector"), "Invalid image dimensions: binary=%dx%d, original=%dx%d", 
                    binary.cols, binary.rows, original.cols, original.rows);
        return lightBars;
    }
    
    vector<vector<Point>> contours;
    findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    Mat hsv;
    cvtColor(original, hsv, COLOR_BGR2HSV);
    vector<Mat> hsv_channels;
    split(hsv, hsv_channels);
    Mat value_channel = hsv_channels[2];
    
    static int lightBarId = 0; // 灯条ID计数器
    
    for (const auto& contour : contours) {
        // 检查轮廓有效性
        if (contour.empty()) continue;
        
        double area = contourArea(contour);
        if (area < 8) continue;
        
        RotatedRect rect = minAreaRect(contour);
        float width = rect.size.width;
        float height = rect.size.height;
        
        if (min(width, height) < 1e-5) continue;
        
        float ratio = max(width, height) / min(width, height);
        float length = max(width, height);
        float short_side = min(width, height);
        
        // 放宽长宽比限制，允许正方形和近似正方形
        if (ratio < 0.8f) {
            // 过于扁平的形状，可能是噪声
            continue;
        }
        
        // 根据形状特征调整筛选条件
        if (ratio <= 1.5f) {
            // 正方形或近似正方形灯条
            // 要求面积足够大，避免小噪声
            if (area < 15.0f) continue;
            if (length < 8.0f) continue; // 基础尺寸要求
        } 
        else if (ratio <= 3.0f) {
            // 略有扁平的矩形灯条
            if (length < 10.0f) continue;
        }
        else {
            // 细长灯条
            if (length < 15.0f) continue;
            if (ratio > 25.0f) continue;
        }
        
        // 增加面积与周长比的筛选，排除过于不规则的形状
        double perimeter = arcLength(contour, true);
        if (perimeter < 1e-5) continue;
        
        double circularity = (4 * CV_PI * area) / (perimeter * perimeter);
        // 圆形度太低说明形状不规则
        if (circularity < 0.3f) continue;
        
        // 计算亮度均值
        Mat mask = Mat::zeros(binary.size(), CV_8UC1);
        drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(255), FILLED);
        Scalar mean_brightness = mean(value_channel, mask);
        
        // 亮度阈值，只保留最亮的区域
        if (mean_brightness[0] < 220) continue; // 从180提高到220
        
        // 对于正方形灯条，可以稍微放宽亮度要求，但保持较高标准
        if (ratio <= 2.0f) {
            // 正方形灯条的最小亮度要求
            if (mean_brightness[0] < 200) continue; // 正方形灯条最低200
        } else {
            // 细长灯条保持更高的亮度标准
            if (mean_brightness[0] < 220) continue; // 细长灯条最低220
        }
        
        // 放宽角度筛选条件，适应不同方向的灯条
        float angle = rect.angle;
        bool is_vertical = false;
        
        if (width < height) {
            angle = 90 - angle;
            is_vertical = true;
        } else {
            angle = -angle;
        }
        
        // 放宽角度限制，允许更多方向的灯条
        if (is_vertical) {
            if (abs(angle) < 30 && ratio > 2.0f) continue; // 对于细长灯条保持较严格角度
        } else {
            if (abs(angle) < 30 && ratio > 2.0f) continue;
        }
        
        // 根据形状调整颜色检测的采样区域
        float colorScore = 0.7;
        string color = "unknown";
        
        // 对于近似正方形的灯条，使用中心区域采样
        Point2f samplePoint = rect.center;
        if (ratio <= 2.0f) {
            // 正方形灯条，在中心区域采样更准确
            int sampleSize = min(rect.size.width, rect.size.height) * 0.3;
            if (sampleSize > 3) {
                // 在中心区域取多个点平均
                vector<Point2f> samplePoints = {
                    rect.center,
                    Point2f(rect.center.x + sampleSize, rect.center.y),
                    Point2f(rect.center.x - sampleSize, rect.center.y),
                    Point2f(rect.center.x, rect.center.y + sampleSize),
                    Point2f(rect.center.x, rect.center.y - sampleSize)
                };
                
                int validSamples = 0;
                Scalar totalColor(0, 0, 0);
                
                for (const auto& point : samplePoints) {
                    int x = max(0, min(original.cols-1, (int)point.x));
                    int y = max(0, min(original.rows-1, (int)point.y));
                    
                    if (x >= 0 && x < original.cols && y >= 0 && y < original.rows) {
                        Vec3b pixel = original.at<Vec3b>(y, x);
                        totalColor[0] += pixel[0];
                        totalColor[1] += pixel[1];
                        totalColor[2] += pixel[2];
                        validSamples++;
                    }
                }
                
                if (validSamples > 0) {
                    Scalar meanColor = totalColor / validSamples;
                    double blue = meanColor[0];
                    double green = meanColor[1];
                    double red = meanColor[2];
                    
                    if (red > blue + 30 && red > green + 30) {
                        color = "red";
                        colorScore = 0.8;
                    } else if (blue > red + 30 && blue > green + 30) {
                        color = "blue";
                        colorScore = 0.8;
                    } else if (red > 200 && green > 200 && blue > 200) {
                        color = "white";
                        colorScore = 0.9;
                    }
                }
            }
        } else {
            // 细长灯条使用单点采样
            int x = max(0, min(original.cols-1, (int)rect.center.x));
            int y = max(0, min(original.rows-1, (int)rect.center.y));
            
            Vec3b pixel = original.at<Vec3b>(y, x);
            int blue = pixel[0];
            int green = pixel[1];
            int red = pixel[2];
            
            if (red > blue + 30 && red > green + 30) {
                color = "red";
                colorScore = 0.8;
            } else if (blue > red + 30 && blue > green + 30) {
                color = "blue";
                colorScore = 0.8;
            } else if (red > 200 && green > 200 && blue > 200) {
                color = "white";
                colorScore = 0.9;
            }
        }
        
        // 颜色得分阈值
        if (colorScore < 0.3) continue; 
        
        LightBar lightBar;
        lightBar.rect = rect;
        lightBar.area = area;
        lightBar.length = length;
        lightBar.angle = angle;
        lightBar.brightness = mean_brightness[0];
        lightBar.color = color;
        lightBar.colorScore = colorScore;
        lightBar.isMatched = false;
        lightBar.id = lightBarId++;
        
        calculateLightBarGeometry(lightBar);
        
        lightBars.push_back(lightBar);
        
        RCLCPP_DEBUG(rclcpp::get_logger("ArmorDetector"), 
                    "检测到灯条%d: 长度=%.1f, 长宽比=%.1f, 面积=%.1f, 颜色=%s, 角度=%.1f, 亮度=%.1f, 位置=(%.1f,%.1f)", 
                    lightBar.id, lightBar.length, ratio, area, color.c_str(), angle, 
                    mean_brightness[0], rect.center.x, rect.center.y);
    }
    
    sort(lightBars.begin(), lightBars.end(), [](const LightBar& a, const LightBar& b) {
        return a.length > b.length;
    });
    
    return lightBars;
}

// 计算两个灯条的匹配得分 
float calculateMatchScore(const LightBar& leftBar, const LightBar& rightBar, const vector<Point2f>& armorVertices) {
    float score = 0.0f;
    
    // 计算灯条宽度（短边长度）
    float leftWidth = min(leftBar.rect.size.width, leftBar.rect.size.height);
    float rightWidth = min(rightBar.rect.size.width, leftBar.rect.size.height);
    
    // 计算灯条长宽比
    float leftRatio = max(leftBar.rect.size.width, leftBar.rect.size.height) / 
                     min(leftBar.rect.size.width, leftBar.rect.size.height);
    float rightRatio = max(rightBar.rect.size.width, rightBar.rect.size.height) / 
                      min(rightBar.rect.size.width, rightBar.rect.size.height);
    
    // 根据灯条形状调整宽度相似性得分权重
    float widthDiffRatio = abs(leftWidth - rightWidth) / max(leftWidth, rightWidth);
    float widthScore = 0.0f;
    
    if (leftRatio <= 2.0f && rightRatio <= 2.0f) {
        // 两个都是近似正方形灯条，放宽宽度差异要求
        if (widthDiffRatio < 0.7f) {
            widthScore = 0.25f * (1.0f - widthDiffRatio / 0.7f);
        }
    } else {
        // 至少有一个是细长灯条，保持原有严格条件
        if (widthDiffRatio < 0.5f) {
            widthScore = 0.25f * (1.0f - widthDiffRatio / 0.5f);
        }
    }
    score += widthScore;
    
    // 长度相似性得分 - 根据形状调整
    float lengthDiffRatio = abs(leftBar.length - rightBar.length) / max(leftBar.length, rightBar.length);
    float lengthScore = 0.0f;
    
    if (leftRatio <= 2.0f && rightRatio <= 2.0f) {
        // 两个都是近似正方形，放宽长度差异要求
        if (lengthDiffRatio < 0.6f) {
            lengthScore = 0.30f * (1.0f - lengthDiffRatio / 0.6f);
        }
    } else {
        // 细长灯条保持严格条件
        if (lengthDiffRatio < 0.4f) {
            lengthScore = 0.30f * (1.0f - lengthDiffRatio / 0.4f);
        }
    }
    score += lengthScore;
    
    // 角度关联性检测 - 考虑平行和垂直两种情况
    float angleDiff = abs(leftBar.angle - rightBar.angle);
    float angleScore = 0.0f;
    
    // 处理角度周期性（0-180度范围）
    if (angleDiff > 90.0f) {
        angleDiff = 180.0f - angleDiff; // 取互补角
    }
    
    if (leftRatio <= 2.0f && rightRatio <= 2.0f) {
        // 近似正方形灯条，放宽角度要求，考虑平行和垂直
        if (angleDiff < 25.0f) {
            // 接近平行（0度）或接近垂直（90度）都给分
            float parallelScore = 0.20f * (1.0f - angleDiff / 25.0f);
            
            // 对于接近垂直的情况也给予一定分数
            float verticalScore = 0.0f;
            if (angleDiff > 65.0f && angleDiff <= 90.0f) {
                verticalScore = 0.15f * (1.0f - (90.0f - angleDiff) / 25.0f);
            }
            
            angleScore = max(parallelScore, verticalScore);
        } else if (angleDiff > 65.0f && angleDiff <= 90.0f) {
            // 接近垂直的情况（65-90度）
            angleScore = 0.15f * (1.0f - (90.0f - angleDiff) / 25.0f);
        }
    } else {
        // 细长灯条主要考虑平行关系
        if (angleDiff < 15.0f) {
            angleScore = 0.20f * (1.0f - angleDiff / 15.0f);
        } else if (angleDiff > 75.0f && angleDiff <= 90.0f) {
            // 细长灯条也可能形成垂直关系，但权重较低
            angleScore = 0.10f * (1.0f - (90.0f - angleDiff) / 15.0f);
        }
    }
    score += angleScore;
    
    // 4. 高度对齐得分
    float yAlignment = abs(leftBar.verticalCenter.y - rightBar.verticalCenter.y);
    float avgLength = (leftBar.length + rightBar.length) * 0.5;
    float yAlignmentRatio = yAlignment / avgLength;
    
    float alignmentScore = 0.0f;
    if (yAlignmentRatio < 0.5f) {
        alignmentScore = 0.05f * (1.0f - yAlignmentRatio / 0.5f);
    }
    score += alignmentScore;
    
    // 5. 装甲板矩形相似度得分
    float rectangleSimilarity = calculateRectangleSimilarity(armorVertices);
    float rectangleScore = 0.20f * rectangleSimilarity;
    score += rectangleScore;
    
    return score;
}

// 计算基于中线端点的装甲板长宽比
float calculateAspectRatioFromMidlines(const LightBar& leftBar, const LightBar& rightBar) {
    // 计算两个灯条中线端点之间的距离作为装甲板的宽度
    float width = norm(leftBar.verticalCenter - rightBar.verticalCenter);
    
    // 计算两个灯条的平均长度作为装甲板的高度
    float height = (leftBar.length + rightBar.length) * 0.5f;
    
    if (height < 1e-5) return 0.0f;
    
    return width / height;
}

// 创建装甲板的辅助函数 - 基于中线端点构建矩形
ArmorPlateInternal createArmorFromLightBars(const LightBar& leftBar, const LightBar& rightBar, 
                                   const Mat& original, bool isSmall) {
    ArmorPlateInternal armor;
    armor.isSimilarParallel = false;
    armor.isSmallArmor = isSmall;
    
    // 确定左右灯条（基于x坐标）
    const LightBar* left = &leftBar;
    const LightBar* right = &rightBar;
    if (leftBar.verticalCenter.x > rightBar.verticalCenter.x) {
        swap(left, right);
    }
    
    // 基于中线端点构建装甲板四边形
    armor.vertices[0] = left->topCenter;     // 左上
    armor.vertices[1] = right->topCenter;    // 右上
    armor.vertices[2] = right->bottomCenter; // 右下
    armor.vertices[3] = left->bottomCenter;  // 左下
    
    vector<Point2f> armorPointsVec;
    for (int i = 0; i < 4; i++) {
        armorPointsVec.push_back(armor.vertices[i]);
    }
    
    armor.boundingRect = boundingRect(armorPointsVec);
    armor.boundingRect &= Rect(0, 0, original.cols, original.rows);
    
    // 计算基于中线端点的长宽比
    armor.aspectRatio = calculateAspectRatioFromMidlines(*left, *right);
    
    return armor;
}

// 修复装甲板验证函数
bool validateArmor(const ArmorPlateInternal& armor, const Mat& original, const Mat& binary) {
    // 检查边界矩形有效性
    if (armor.boundingRect.width < 3 || armor.boundingRect.height < 3) {
        RCLCPP_DEBUG(rclcpp::get_logger("ArmorDetector"), 
                    "装甲板%d 边界矩形太小: %dx%d", 
                    armor.armorId, armor.boundingRect.width, armor.boundingRect.height);
        return false;
    }
    
    // 检查边界矩形是否在图像范围内
    if (armor.boundingRect.x < 0 || armor.boundingRect.y < 0 ||
        armor.boundingRect.x + armor.boundingRect.width > original.cols ||
        armor.boundingRect.y + armor.boundingRect.height > original.rows) {
        RCLCPP_DEBUG(rclcpp::get_logger("ArmorDetector"), 
                    "装甲板%d 边界矩形超出图像范围", armor.armorId);
        return false;
    }
    
    // 检查matchedLightBars是否有效
    if (armor.matchedLightBars.size() < 2) {
        RCLCPP_DEBUG(rclcpp::get_logger("ArmorDetector"), 
                    "装甲板%d 匹配灯条数量不足: %d", 
                    armor.armorId, (int)armor.matchedLightBars.size());
        return false;
    }
    
    // 获取灯条信息
    const LightBar& leftBar = armor.matchedLightBars[0];
    const LightBar& rightBar = armor.matchedLightBars[1];
    
    float leftRatio = max(leftBar.rect.size.width, leftBar.rect.size.height) / 
                     min(leftBar.rect.size.width, leftBar.rect.size.height);
    float rightRatio = max(rightBar.rect.size.width, rightBar.rect.size.height) / 
                      min(rightBar.rect.size.width, rightBar.rect.size.height);
    
    // 根据灯条形状调整长宽比限制
    bool aspectRatioValid = false;
    if (leftRatio <= 2.0f && rightRatio <= 2.0f) {
        // 正方形灯条：放宽长宽比限制
        aspectRatioValid = (armor.aspectRatio >= 0.5f && armor.aspectRatio <= 4.5f);
        if (!aspectRatioValid) {
            RCLCPP_DEBUG(rclcpp::get_logger("ArmorDetector"), 
                        "正方形灯条装甲板长宽比验证失败: %.2f (允许范围: 0.5-4.5)", armor.aspectRatio);
        }
    } else {
        // 细长灯条：保持原有条件
        aspectRatioValid = (armor.aspectRatio >= 0.7f && armor.aspectRatio <= 5.0f);
        if (!aspectRatioValid) {
            RCLCPP_DEBUG(rclcpp::get_logger("ArmorDetector"), 
                        "装甲板长宽比验证失败: %.2f (允许范围: 0.7-5.0)", armor.aspectRatio);
        }
    }
    
    if (!aspectRatioValid) {
        return false;
    }
    
    // 检查白色比例（在二值图下）
    float whiteRatio = 0;
    if (armor.boundingRect.width > 5 && armor.boundingRect.height > 5) {
        // 确保ROI在图像范围内
        Rect safeRect = armor.boundingRect & Rect(0, 0, binary.cols, binary.rows);
        if (safeRect.width > 2 && safeRect.height > 2) {
            Mat binaryRoi = binary(safeRect);
            whiteRatio = calculateWhiteRatio(binaryRoi);
        }
    }
    
    // 根据灯条形状调整白色比例阈值
    bool whiteRatioValid = true;
    if (leftRatio <= 2.0f && rightRatio <= 2.0f) {
        // 正方形灯条：放宽白色比例限制
        whiteRatioValid = (whiteRatio <= 0.90f); // 提高到90%
        if (!whiteRatioValid) {
            RCLCPP_DEBUG(rclcpp::get_logger("ArmorDetector"), 
                        "正方形灯条装甲板%d 白色占比过高: %.2f > 0.90", armor.armorId, whiteRatio);
        }
    } else {
        whiteRatioValid = (whiteRatio <= 0.85f); // 细长灯条保持85%
        if (!whiteRatioValid) {
            RCLCPP_DEBUG(rclcpp::get_logger("ArmorDetector"), 
                        "装甲板%d 白色占比过高: %.2f > 0.85", armor.armorId, whiteRatio);
        }
    }
    
    if (!whiteRatioValid) {
        return false;
    }
    
    // 检查装甲板面积
    int armorArea = armor.boundingRect.width * armor.boundingRect.height;
    bool areaValid = false;
    if (leftRatio <= 2.0f && rightRatio <= 2.0f) {
        areaValid = (armorArea >= 40); // 正方形灯条降低面积要求
        if (!areaValid) {
            RCLCPP_DEBUG(rclcpp::get_logger("ArmorDetector"), 
                        "正方形灯条装甲板面积太小: %d < 40", armorArea);
        }
    } else {
        areaValid = (armorArea >= 60); // 细长灯条保持60
        if (!areaValid) {
            RCLCPP_DEBUG(rclcpp::get_logger("ArmorDetector"), 
                        "装甲板面积太小: %d < 60", armorArea);
        }
    }
    
    if (!areaValid) {
        return false;
    }
    
    RCLCPP_DEBUG(rclcpp::get_logger("ArmorDetector"), "装甲板%d 验证通过!", armor.armorId);
    return true;
}

// 装甲板匹配算法
vector<ArmorPlateInternal> matchArmorPlates(vector<LightBar>& lightBars, const Mat& original, const Mat& binary) {
    vector<ArmorPlateInternal> armors;
    
    if (lightBars.size() < 2) return armors;
    
    // 按x坐标排序
    sort(lightBars.begin(), lightBars.end(), [](const LightBar& a, const LightBar& b) {
        return a.verticalCenter.x < b.verticalCenter.x;
    });
    
    RCLCPP_DEBUG(rclcpp::get_logger("ArmorDetector"), "开始计算所有灯条组合的匹配得分");
    
    // 存储所有可能的匹配对和得分
    struct MatchCandidate {
        int leftIndex;
        int rightIndex;
        float score;
        ArmorPlateInternal armor;
        bool isValid;
    };
    
    vector<MatchCandidate> allCandidates;
    
    // 第一遍：计算所有灯条组合的得分
    for (size_t i = 0; i < lightBars.size(); i++) {
        for (size_t j = i + 1; j < lightBars.size(); j++) {
            LightBar& leftBar = lightBars[i];
            LightBar& rightBar = lightBars[j];
            
            // 创建临时装甲板用于计算矩形相似度
            bool isSmall = (leftBar.length < 30 && rightBar.length < 30);
            ArmorPlateInternal tempArmor = createArmorFromLightBars(leftBar, rightBar, original, isSmall);
            
            // 构建装甲板顶点向量
            vector<Point2f> armorVertices;
            for (int k = 0; k < 4; k++) {
                armorVertices.push_back(tempArmor.vertices[k]);
            }
            
            float matchScore = calculateMatchScore(leftBar, rightBar, armorVertices);
            
            // 创建完整的装甲板对象进行验证
            ArmorPlateInternal armor = createArmorFromLightBars(leftBar, rightBar, original, isSmall);
            armor.matchScore = matchScore;
            armor.isSimilarParallel = (abs(leftBar.angle - rightBar.angle) < 15.0f);
            armor.matchedLightBars = {leftBar, rightBar};
            armor.rectangleSimilarity = calculateRectangleSimilarity(armorVertices);
            
            MatchCandidate candidate;
            candidate.leftIndex = i;
            candidate.rightIndex = j;
            candidate.score = matchScore;
            candidate.armor = armor;
            candidate.isValid = (matchScore > 0.75f) && validateArmor(armor, original, binary);
            
            allCandidates.push_back(candidate);
            
            if (candidate.isValid) {
                RCLCPP_DEBUG(rclcpp::get_logger("ArmorDetector"), 
                           "有效候选装甲板: 灯条%d & %d, 得分=%.2f, 长宽比=%.2f, 矩形相似度=%.2f", 
                           leftBar.id, rightBar.id, matchScore, armor.aspectRatio, armor.rectangleSimilarity);
            }
        }
    }
    
    // 筛选有效候选
    vector<MatchCandidate> potentialMatches;
    for (const auto& candidate : allCandidates) {
        if (candidate.isValid) {
            potentialMatches.push_back(candidate);
        }
    }
    
    RCLCPP_DEBUG(rclcpp::get_logger("ArmorDetector"), 
                "有效装甲板候选数量: %d / %d", 
                (int)potentialMatches.size(), (int)(lightBars.size() * (lightBars.size() - 1) / 2));
    
    // 按匹配得分排序（从高到低）
    sort(potentialMatches.begin(), potentialMatches.end(), 
         [](const MatchCandidate& a, const MatchCandidate& b) {
             return a.score > b.score;
         });
    
    // 第二遍：按得分顺序选择装甲板
    set<pair<int, int>> usedPairs; // 记录已经使用的灯条对
    set<int> usedLightBars; // 记录已经使用的灯条ID
    
    static int armorIdCounter = 0; // 装甲板ID计数器
    
    for (const auto& candidate : potentialMatches) {
        int i = candidate.leftIndex;
        int j = candidate.rightIndex;
        float score = candidate.score;
        
        LightBar& leftBar = lightBars[i];
        LightBar& rightBar = lightBars[j];
        
        // 检查灯条是否已经被使用
        if (usedLightBars.find(leftBar.id) != usedLightBars.end() || 
            usedLightBars.find(rightBar.id) != usedLightBars.end()) {
            RCLCPP_DEBUG(rclcpp::get_logger("ArmorDetector"), 
                        "跳过装甲板: 灯条%d 或 %d 已被使用", leftBar.id, rightBar.id);
            continue;
        }
        
        // 检查是否是完全相同的灯条对
        pair<int, int> lightBarPair = make_pair(min(i, j), max(i, j));
        if (usedPairs.find(lightBarPair) != usedPairs.end()) {
            RCLCPP_DEBUG(rclcpp::get_logger("ArmorDetector"), 
                        "跳过重复装甲板: 灯条对%d-%d", leftBar.id, rightBar.id);
            continue;
        }
        
        // 创建装甲板
        ArmorPlateInternal armor = candidate.armor;
        armor.armorId = armorIdCounter++;
        
        armors.push_back(armor);
        usedPairs.insert(lightBarPair);
        usedLightBars.insert(leftBar.id);
        usedLightBars.insert(rightBar.id);
        
        RCLCPP_DEBUG(rclcpp::get_logger("ArmorDetector"), 
                   "成功匹配装甲板%d！灯条%d & %d, 得分=%.2f, 长宽比=%.2f, 矩形相似度=%.2f", 
                   armor.armorId, leftBar.id, rightBar.id, score, armor.aspectRatio, armor.rectangleSimilarity);
    }
    
    // 输出匹配统计
    RCLCPP_DEBUG(rclcpp::get_logger("ArmorDetector"), 
                "匹配统计: 总灯条数=%d, 装甲板数=%d", 
                (int)lightBars.size(), (int)armors.size());
    
    // 按得分对最终装甲板排序
    sort(armors.begin(), armors.end(), [](const ArmorPlateInternal& a, const ArmorPlateInternal& b) {
        return a.matchScore > b.matchScore;
    });
    
    return armors;
}

// ArmorDetector 类实现
ArmorDetector::ArmorDetector()
{
    RCLCPP_INFO(rclcpp::get_logger("ArmorDetector"), "装甲板检测器初始化完成");
}

ArmorDetector::~ArmorDetector()
{
    // 清理资源
}





std::vector<ArmorPlate> ArmorDetector::detect(const cv::Mat& image)
{
    std::vector<ArmorPlate> results;
    
    if (image.empty()) {
        RCLCPP_ERROR(rclcpp::get_logger("ArmorDetector"), "输入图像为空！");
        return results;
    }
    
    int img_cols = image.cols;
    int img_rows = image.rows;
    const int rviz_img_width = img_cols;
    const int rviz_img_height = img_rows;

    // 1. 第一次对称中心：(2/3W, 1/2H)
    float Cx1 = rviz_img_width * (2.0f / 3.0f);  // 第一对称中心x
    float Cy1 = rviz_img_height * 0.5f;          // 第一对称中心y
    
    // 2. 第二次对称轴：右往左1/8直线（x=7/8W）
    float Cx2 = rviz_img_width * (7.0f / 8.0f);  // 第二对称轴x（仅x轴）

    RCLCPP_INFO(rclcpp::get_logger("ArmorDetector"), 
                "第一次对称中心: (%.1f, %.1f) | 第二次对称轴x: %.1f", 
                Cx1, Cy1, Cx2);
    
    try {
        cv::Mat binary = preprocessImage(image);
        if (binary.empty()) {
            RCLCPP_WARN(rclcpp::get_logger("ArmorDetector"), "预处理后二值图为空");
            return results;
        }
        
        std::vector<LightBar> lightBars = detectLightBarsByBrightness(binary, image);
        std::vector<ArmorPlateInternal> internalArmors = matchArmorPlates(lightBars, image, binary);
        
        for (const auto& internalArmor : internalArmors) {
            ArmorPlate armor;
            cv::Rect original_rect = internalArmor.boundingRect;
            
            // 3. 装甲板原始中心（OpenCV坐标）
            cv::Point2f original_center(
                original_rect.x + original_rect.width / 2.0f,
                original_rect.y + original_rect.height / 2.0f
            );
            
            // 4. 转换到RViz原始坐标（y轴反转）
            float Ox_prime = original_center.x;
            float Oy_prime = rviz_img_height - original_center.y;
            
            // 5. 第一次对称：以(Cx1, Cy1)为中心
            float box_center_x1 = 2 * Cx1 - Ox_prime;  // 第一次x对称
            float box_center_y1 = 2 * Cy1 - Oy_prime;  // 第一次y对称
            
            // 6. 第二次对称：沿x=Cx2直线（仅x轴，y轴保持第一次结果）
            float box_center_x2 = 2 * Cx2 - box_center_x1;  // 第二次x对称
            float box_center_y2 = box_center_y1;            // y轴不变
            
            // 7. 计算最终矩形框四角
            float w = original_rect.width;
            float h = original_rect.height;
            cv::Point2f rect_tl(box_center_x2 - w/2, box_center_y2 - h/2);
            cv::Point2f rect_br(box_center_x2 + w/2, box_center_y2 + h/2);
            
            // 8. 转换为RViz矩形框
            cv::Rect rviz_rect;
            rviz_rect.x = 0.5*static_cast<int>(rect_tl.x)-335;
            rviz_rect.y = 0.5*static_cast<int>(rect_tl.y)-15;  
            rviz_rect.width = static_cast<int>(w);
            rviz_rect.height = static_cast<int>(h);
            
            // 9. 两次对称变换四个顶点
            for (int i = 0; i < 4; ++i) {
                cv::Point2f original_vertex = internalArmor.vertices[i];
                // 顶点转换到RViz原始坐标
                float vx_prime = original_vertex.x;
                float vy_prime = rviz_img_height - original_vertex.y;
                // 第一次对称
                float vx1 = 2 * Cx1 - vx_prime;
                float vy1 = 2 * Cy1 - vy_prime;
                // 第二次对称（仅x轴）
                armor.vertices[i].x = 2 * Cx2 - vx1;
                armor.vertices[i].y = vy1;  // y轴保持第一次结果
            }
            
            // 10. 填充装甲板信息
            armor.center = cv::Point2f(box_center_x2, box_center_y2);
            armor.bounding_box = rviz_rect;
            armor.confidence = internalArmor.matchScore;
            armor.label = "armor_" + std::to_string(internalArmor.armorId);
            
            // 调试信息：验证两次对称
            RCLCPP_INFO(rclcpp::get_logger("ArmorDetector"), 
                        "\n===== 装甲板 %s 两次对称 =====", armor.label.c_str());
            RCLCPP_INFO(rclcpp::get_logger("ArmorDetector"), 
                        "原始中心（RViz）: (%.1f, %.1f)", Ox_prime, Oy_prime);
            RCLCPP_INFO(rclcpp::get_logger("ArmorDetector"), 
                        "第一次对称后: (%.1f, %.1f)", box_center_x1, box_center_y1);
            RCLCPP_INFO(rclcpp::get_logger("ArmorDetector"), 
                        "第二次对称后: (%.1f, %.1f)", box_center_x2, box_center_y2);
            RCLCPP_INFO(rclcpp::get_logger("ArmorDetector"), 
                        "==============================\n");
            
            results.push_back(armor);
        }
        
    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("ArmorDetector"), "OpenCV错误: %s", e.what());
    }
    
    return results;
}





