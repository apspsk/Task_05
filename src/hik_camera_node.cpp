#include "MvCameraControl.h"
#include "PixelType.h"
#include "cv_bridge/cv_bridge.h"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <rclcpp/rclcpp.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

using namespace std::chrono_literals;

struct CameraDescriptor {
    std::string model;     
    std::string serial;  
    std::string ip;     
    unsigned int type;   
};

struct ImageData {
    unsigned char *data;       
    unsigned int width;        
    unsigned int height;       
    unsigned int pixelFormat;  
    unsigned int dataSize;     
    
    ImageData() : data(nullptr), width(0), height(0), pixelFormat(0), dataSize(0) {}
};

class HikCameraController {
public:
    HikCameraController() : handle_(nullptr), is_open_(false), is_grabbing_(false), 
                           convert_buffer_(nullptr), buffer_size_(0), device_index_(0),
                           saved_exposure_(0.0f), saved_gain_(0.0f), saved_trigger_(false),
                           saved_frame_rate_(0.0f), saved_pixel_format_(0) {}

    ~HikCameraController() {
        if (is_grabbing_) stop_grabbing();
        if (is_open_) close();
        if (convert_buffer_) delete[] convert_buffer_;
    }

    std::string get_last_error() const {
        return last_error_;
    }

    bool reconnect(unsigned int index, int max_retries = 5, int retry_delay_ms = 1000) {
        device_index_ = index;
        
        for (int attempt = 1; attempt <= max_retries && !is_open_; ++attempt) {
            if (is_grabbing_) stop_grabbing();
            if (is_open_) close();
            
            std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms * attempt));
            
            if (!open(index)) continue;
            
            if (saved_exposure_ > 0.0f) set_exposure(saved_exposure_);
            if (saved_gain_ > 0.0f) set_gain(saved_gain_);
            if (saved_frame_rate_ > 0.0f) set_frame_rate(saved_frame_rate_);
            if (saved_pixel_format_ > 0) set_pixel_format(saved_pixel_format_);
            set_trigger_mode(saved_trigger_);
            
            if (!start_grabbing()) {
                close();
                continue;
            }
            
            return true;
        }
        return false;
    }

    bool grab_image(ImageData &data, unsigned int timeout = 1000, unsigned int desired_format = 0) {
        if (!is_grabbing_) {
            last_error_ = "Camera not grabbing";
            return false;
        }

        MV_FRAME_OUT frame;
        memset(&frame, 0, sizeof(MV_FRAME_OUT));

        int ret = MV_CC_GetImageBuffer(handle_, &frame, timeout);
        if (ret != MV_OK) {
            set_error("Get image buffer failed", ret);
            return false;
        }

        if (!frame.pBufAddr || frame.stFrameInfo.nWidth == 0 || frame.stFrameInfo.nHeight == 0) {
            MV_CC_FreeImageBuffer(handle_, &frame);
            last_error_ = "Invalid image data";
            return false;
        }

        unsigned int width = frame.stFrameInfo.nWidth;
        unsigned int height = frame.stFrameInfo.nHeight;
        unsigned int src_format = frame.stFrameInfo.enPixelType;
        unsigned int target_format = desired_format ? desired_format : src_format;
        bool need_convert = (target_format != src_format);

        if (need_convert && !is_pixel_format_supported(target_format)) {
            RCLCPP_WARN(rclcpp::get_logger("camera_driver"), 
                       "Target pixel format 0x%08X not supported, using original", target_format);
            target_format = src_format;
            need_convert = false;
        }

        size_t required_size = need_convert ? estimate_buffer_size(target_format, width, height) 
                                           : frame.stFrameInfo.nFrameLen;
        if (required_size == 0) {
            MV_CC_FreeImageBuffer(handle_, &frame);
            last_error_ = "Cannot calculate buffer size";
            return false;
        }

        if (buffer_size_ < required_size) {
            if (convert_buffer_) {
                delete[] convert_buffer_;
                convert_buffer_ = nullptr;
            }
            try {
                convert_buffer_ = new unsigned char[required_size];
                buffer_size_ = required_size;
            } catch (const std::bad_alloc&) {
                MV_CC_FreeImageBuffer(handle_, &frame);
                last_error_ = "Memory allocation failed: buffer size " + std::to_string(required_size);
                return false;
            }
        }

        if (need_convert) {
            MV_CC_PIXEL_CONVERT_PARAM convert_param;
            memset(&convert_param, 0, sizeof(MV_CC_PIXEL_CONVERT_PARAM));
            convert_param.nWidth = width;
            convert_param.nHeight = height;
            convert_param.pSrcData = frame.pBufAddr;
            convert_param.nSrcDataLen = frame.stFrameInfo.nFrameLen;
            convert_param.enSrcPixelType = static_cast<MvGvspPixelType>(src_format);
            convert_param.enDstPixelType = static_cast<MvGvspPixelType>(target_format);
            convert_param.pDstBuffer = convert_buffer_;
            convert_param.nDstBufferSize = buffer_size_;

            ret = MV_CC_ConvertPixelType(handle_, &convert_param);
            if (ret == MV_OK) {
                data.pixelFormat = target_format;
                data.data = convert_buffer_;
                data.dataSize = convert_param.nDstLen;
            } else {
                set_error("Pixel format conversion failed", ret);
                MV_CC_FreeImageBuffer(handle_, &frame);
                return false;
            }
        } else {
            if (buffer_size_ < frame.stFrameInfo.nFrameLen) {
                delete[] convert_buffer_;
                try {
                    convert_buffer_ = new unsigned char[frame.stFrameInfo.nFrameLen];
                    buffer_size_ = frame.stFrameInfo.nFrameLen;
                } catch (const std::bad_alloc&) {
                    MV_CC_FreeImageBuffer(handle_, &frame);
                    last_error_ = "Memory allocation failed";
                    return false;
                }
            }
            memcpy(convert_buffer_, frame.pBufAddr, frame.stFrameInfo.nFrameLen);
            data.pixelFormat = src_format;
            data.data = convert_buffer_;
            data.dataSize = frame.stFrameInfo.nFrameLen;
        }

        data.width = width;
        data.height = height;
        MV_CC_FreeImageBuffer(handle_, &frame);
        
        return true;
    }

    bool set_exposure(float exposure) {
        if (!is_open_) {
            last_error_ = "Camera is not open";
            return false;
        }
        int ret = MV_CC_SetFloatValue(handle_, "ExposureTime", exposure);
        if (ret != MV_OK) {
            set_error("Set exposure time failed", ret);
            return false;
        }
        saved_exposure_ = exposure;
        return true;
    }

    bool set_gain(float gain) {
        if (!is_open_) return false;
        int ret = MV_CC_SetFloatValue(handle_, "Gain", gain);
        if (ret != MV_OK) return false;
        saved_gain_ = gain;
        return true;
    }

    bool set_trigger_mode(bool enable) {
        if (!is_open_) return false;
        int ret = MV_CC_SetEnumValue(handle_, "TriggerMode", enable ? 1 : 0);
        if (ret != MV_OK) return false;
        saved_trigger_ = enable;
        return true;
    }

    bool set_frame_rate(float fps) {
        if (!is_open_) return false;
        if (fps <= 0.0f) return false;

        MV_CC_SetBoolValue(handle_, "AcquisitionFrameRateEnable", true);
        MV_CC_SetEnumValue(handle_, "AcquisitionFrameRateAuto", 0);
        
        int ret = MV_CC_SetFloatValue(handle_, "AcquisitionFrameRate", fps);
        if (ret != MV_OK) {
            MVCC_FLOATVALUE limits;
            if (MV_CC_GetFloatValue(handle_, "AcquisitionFrameRate", &limits) == MV_OK) {
                float clamped = std::max(limits.fMin, std::min(fps, limits.fMax));
                ret = MV_CC_SetFloatValue(handle_, "AcquisitionFrameRate", clamped);
            }
        }
        
        if (ret == MV_OK) {
            saved_frame_rate_ = fps;
            return true;
        }
        return false;
    }

    // 获取相机实际帧率
    float get_frame_rate() {
        if (!is_open_) return 0.0f;
        MVCC_FLOATVALUE value;
        if (MV_CC_GetFloatValue(handle_, "ResultingFrameRate", &value) == MV_OK) 
            return value.fCurValue;
        return 0.0f;
    }

    bool set_pixel_format(unsigned int format) {
        if (!is_open_) return false;
        if (saved_pixel_format_ == format && format != 0) return true;

        if (!is_pixel_format_supported(format)) {
            last_error_ = "Unsupported pixel format: 0x" + std::to_string(format);
            return false;
        }

        bool was_grabbing = is_grabbing_;
        if (was_grabbing) stop_grabbing();

        int ret = MV_CC_SetEnumValue(handle_, "PixelFormat", format);
        if (ret != MV_OK) {
            set_error("Set pixel format failed", ret);
            if (was_grabbing) start_grabbing();
            return false;
        }

        saved_pixel_format_ = format;
        if (was_grabbing) start_grabbing();
        
        RCLCPP_INFO(rclcpp::get_logger("camera_driver"), "Pixel format set to: 0x%08X", format);
        return true;
    }
    
    bool is_pixel_format_supported(unsigned int format) const {
        static const std::vector<unsigned int> supported_formats = {
            PixelType_Gvsp_Mono8, PixelType_Gvsp_Mono10, PixelType_Gvsp_Mono12,
            PixelType_Gvsp_Mono14, PixelType_Gvsp_Mono16,
            PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_BGR8_Packed,
            PixelType_Gvsp_RGB10_Packed, PixelType_Gvsp_BGR10_Packed,
            PixelType_Gvsp_RGB12_Packed, PixelType_Gvsp_BGR12_Packed,
            PixelType_Gvsp_RGB16_Packed, PixelType_Gvsp_BGR16_Packed,
            PixelType_Gvsp_RGBA8_Packed, PixelType_Gvsp_BGRA8_Packed,
            PixelType_Gvsp_YUV422_Packed, PixelType_Gvsp_YUV422_YUYV_Packed,
            PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerRG8,
            PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerBG8
        };
        
        return std::find(supported_formats.begin(), supported_formats.end(), format) != supported_formats.end();
    }

    bool open(unsigned int index = 0) {
        if (is_open_) {
            last_error_ = "Camera is already open";
            return false;
        }

        MV_CC_DEVICE_INFO_LIST device_list;
        memset(&device_list, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

        int ret = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list);
        if (ret != MV_OK) {
            set_error("Enumerate devices failed", ret);
            return false;
        }
        if (index >= device_list.nDeviceNum) {
            last_error_ = "Invalid device index";
            return false;
        }

        ret = MV_CC_CreateHandle(&handle_, device_list.pDeviceInfo[index]);
        if (ret != MV_OK) {
            set_error("Create handle failed", ret);
            return false;
        }

        ret = MV_CC_OpenDevice(handle_);
        if (ret != MV_OK) {
            set_error("Open device failed", ret);
            MV_CC_DestroyHandle(handle_);
            handle_ = nullptr;
            return false;
        }

        ret = MV_CC_SetEnumValue(handle_, "TriggerMode", 0);
        if (ret != MV_OK) {
            set_error("Set trigger mode failed", ret);
        }
        
        is_open_ = true;
        device_index_ = index;
        return true;
    }

    bool close() {
        if (!is_open_) return true;
        if (is_grabbing_) stop_grabbing();
        
        int ret = MV_CC_CloseDevice(handle_);
        if (ret != MV_OK) {
            set_error("Close device failed", ret);
        }
        
        ret = MV_CC_DestroyHandle(handle_);
        if (ret != MV_OK) {
            set_error("Destroy handle failed", ret);
        }
        
        handle_ = nullptr;
        is_open_ = false;
        return true;
    }

    bool start_grabbing() {
        if (!is_open_) {
            last_error_ = "Camera is not open";
            return false;
        }
        if (is_grabbing_) return true;
        
        int ret = MV_CC_StartGrabbing(handle_);
        if (ret != MV_OK) {
            set_error("Start grabbing failed", ret);
            return false;
        }
        is_grabbing_ = true;
        return true;
    }

    bool stop_grabbing() {
        if (!is_grabbing_) return true;
        int ret = MV_CC_StopGrabbing(handle_);
        if (ret != MV_OK) {
            set_error("Stop grabbing failed", ret);
            return false;
        }
        is_grabbing_ = false;
        return true;
    }

    bool is_open() const { return is_open_; }
    bool is_grabbing() const { return is_grabbing_; }

private:
    void *handle_;
    bool is_open_;
    bool is_grabbing_;
    unsigned char *convert_buffer_;
    unsigned int buffer_size_;
    unsigned int device_index_;
    float saved_exposure_;
    float saved_gain_;
    bool saved_trigger_;
    float saved_frame_rate_;
    unsigned int saved_pixel_format_;
    std::string last_error_;

    size_t estimate_buffer_size(unsigned int format, unsigned int w, unsigned int h) {
        switch (format) {
        case PixelType_Gvsp_Mono8: return w * h;
        case PixelType_Gvsp_Mono10:
        case PixelType_Gvsp_Mono12:
        case PixelType_Gvsp_Mono14:
        case PixelType_Gvsp_Mono16: return w * h * 2;
        case PixelType_Gvsp_RGB8_Packed:
        case PixelType_Gvsp_BGR8_Packed: return w * h * 3;
        case PixelType_Gvsp_RGB10_Packed:
        case PixelType_Gvsp_BGR10_Packed:
        case PixelType_Gvsp_RGB12_Packed:
        case PixelType_Gvsp_BGR12_Packed:
        case PixelType_Gvsp_RGB16_Packed:
        case PixelType_Gvsp_BGR16_Packed: return w * h * 6;
        case PixelType_Gvsp_RGBA8_Packed:
        case PixelType_Gvsp_BGRA8_Packed: return w * h * 4;
        case PixelType_Gvsp_YUV422_Packed:
        case PixelType_Gvsp_YUV422_YUYV_Packed: return w * h * 2;
        case PixelType_Gvsp_BayerGR8:
        case PixelType_Gvsp_BayerRG8:
        case PixelType_Gvsp_BayerGB8:
        case PixelType_Gvsp_BayerBG8: return w * h;
        default: return w * h * 3;
        }
    }

    void set_error(const std::string &error, int error_code) {
        std::ostringstream oss;
        oss << error << " (Error code: 0x" << std::hex << error_code << ")";
        last_error_ = oss.str();
    }
};

class HikCameraNode : public rclcpp::Node {
public:
    explicit HikCameraNode() : Node("hik_camera_node") {
        declare_parameter("exposure", 4000.0);
        declare_parameter("image_gain", 16.9807);
        declare_parameter("use_trigger", false);
        declare_parameter("fps", 165.0);
        declare_parameter("pixel_format_code", static_cast<int>(PixelType_Gvsp_BayerRG8));
        declare_parameter("camera_frame", "camera_optical_frame");
        declare_parameter("serial_number", "");

        exposure_ = get_parameter("exposure").as_double();
        gain_ = get_parameter("image_gain").as_double();
        trigger_ = get_parameter("use_trigger").as_bool();
        frame_rate_ = get_parameter("fps").as_double();
        pixel_format_ = get_parameter("pixel_format_code").as_int();
        frame_id_ = get_parameter("camera_frame").as_string();
        serial_number_ = get_parameter("serial_number").as_string();

        param_callback_ = add_on_set_parameters_callback([this](const auto &params) {
            rcl_interfaces::msg::SetParametersResult result;
            result.successful = true;
            std::ostringstream reason;
            
            for (const auto &p : params) {
                if (p.get_name() == "exposure") {
                    double v = p.as_double();
                    double previous = exposure_;
                    if (!camera_.set_exposure(v)) {
                        result.successful = false;
                        exposure_ = previous;
                        if (!reason.str().empty()) reason << "; ";
                        reason << "Exposure setting failed: " << camera_.get_last_error();
                    } else {
                        exposure_ = v;
                    }
                } else if (p.get_name() == "image_gain") {
                    double v = p.as_double();
                    double previous = gain_;
                    if (!camera_.set_gain(v)) {
                        result.successful = false;
                        gain_ = previous;
                        if (!reason.str().empty()) reason << "; ";
                        reason << "Gain setting failed: " << camera_.get_last_error();
                    } else {
                        gain_ = v;
                    }
                } else if (p.get_name() == "use_trigger") {
                    bool v = p.as_bool();
                    bool previous = trigger_;
                    if (!camera_.set_trigger_mode(v)) {
                        result.successful = false;
                        trigger_ = previous;
                        if (!reason.str().empty()) reason << "; ";
                        reason << "Trigger mode setting failed: " << camera_.get_last_error();
                    } else {
                        trigger_ = v;
                    }
                } else if (p.get_name() == "fps") {
                    double v = p.as_double();
                    if (v <= 0.0) {
                        result.successful = false;
                        if (!reason.str().empty()) reason << "; ";
                        reason << "Frame rate must be greater than 0";
                        continue;
                    }
                    double previous = frame_rate_;
                    if (!camera_.set_frame_rate(v)) {
                        result.successful = false;
                        frame_rate_ = previous;
                        if (!reason.str().empty()) reason << "; ";
                        reason << "Frame rate setting failed: " << camera_.get_last_error();
                    } else {
                        frame_rate_ = v;
                    }
                } else if (p.get_name() == "pixel_format_code") {
                    int v = p.as_int();
                    unsigned int previous = pixel_format_;
                    
                    if (!camera_.is_pixel_format_supported(v)) {
                        result.successful = false;
                        if (!reason.str().empty()) reason << "; ";
                        reason << "Pixel format 0x" << std::hex << v << " not supported";
                        continue;
                    }
                    
                    if (!camera_.set_pixel_format(v)) {
                        result.successful = false;
                        pixel_format_ = previous;
                        if (!reason.str().empty()) reason << "; ";
                        reason << "Pixel format setting failed: " << camera_.get_last_error();
                    } else {
                        pixel_format_ = v;
                        RCLCPP_INFO(get_logger(), "Pixel format updated to: 0x%08X", v);
                    }
                }
            }
            result.reason = reason.str();
            return result;
        });

        publisher_ = create_publisher<sensor_msgs::msg::Image>("/image_raw", 10);
        timer_ = create_wall_timer(2ms, std::bind(&HikCameraNode::publish_image, this));
        last_fps_time_ = now();
        
        RCLCPP_INFO(get_logger(), "HikCameraNode initialized");
        RCLCPP_INFO(get_logger(), "Parameters: exposure=%.1f, gain=%.1f, fps=%.1f, trigger=%s", 
                   exposure_, gain_, frame_rate_, trigger_ ? "true" : "false");
    }

    ~HikCameraNode() {
        camera_.close();
        RCLCPP_INFO(get_logger(), "HikCameraNode shutdown");
    }

private:
    void publish_image() {
        if (!camera_.is_open()) {
            if (!settings_applied_ || (now() - last_attempt_).seconds() >= 1.0) {
                last_attempt_ = now();
                bool success = camera_.reconnect(0, 3, 500);
                
                if (success) {
                    RCLCPP_INFO(get_logger(), "Camera connected successfully");
                    apply_settings();
                    camera_.start_grabbing();
                } else {
                    RCLCPP_WARN(get_logger(), "Camera connection failed: %s", camera_.get_last_error().c_str());
                }
            }
            return;
        }

        if (!settings_applied_) apply_settings();
        
        if (!camera_.is_grabbing()) {
            if (!camera_.start_grabbing()) {
                RCLCPP_WARN(get_logger(), "Start grabbing failed: %s", camera_.get_last_error().c_str());
                camera_.close();
                settings_applied_ = false;
                return;
            }
        }

        ImageData img;
        if (!camera_.grab_image(img, 1000, pixel_format_)) {
            consecutive_failures_++;
            if (consecutive_failures_ >= 5) {
                RCLCPP_WARN(get_logger(), "Continuous capture failed %u times, triggering reconnect", consecutive_failures_);
                bool success = camera_.reconnect(0, 3, 500);
                if (success) {
                    RCLCPP_INFO(get_logger(), "Reconnect successful");
                    apply_settings();
                    camera_.start_grabbing();
                } else {
                    RCLCPP_ERROR(get_logger(), "Reconnect failed: %s", camera_.get_last_error().c_str());
                }
                consecutive_failures_ = 0;
            }
            return;
        }
        consecutive_failures_ = 0;

        if (!img.data || !img.width || !img.height) return;

        // 帧率统计
        auto current_time = now();
        frame_count_++;
        
        // 每2秒计算并显示一次帧率
        if ((current_time - last_fps_time_).seconds() >= 2.0) {
            double fps = frame_count_ / (current_time - last_fps_time_).seconds();
            frame_count_ = 0;
            last_fps_time_ = current_time;
            
            // 获取相机报告的帧率
            float camera_fps = camera_.get_frame_rate();
            
            RCLCPP_INFO(get_logger(), 
                       "FPS: %.2f (Camera: %.2f) | Size: %ux%u | Format: %s", 
                       fps, camera_fps, img.width, img.height, to_encoding(img.pixelFormat).c_str());
        }

        std::string encoding = to_encoding(img.pixelFormat);
        
        auto image_msg = std::make_unique<sensor_msgs::msg::Image>();
        image_msg->header.stamp = current_time;
        image_msg->header.frame_id = frame_id_;
        image_msg->height = img.height;
        image_msg->width = img.width;
        image_msg->encoding = encoding;
        
        // 设置步长
        if (encoding == sensor_msgs::image_encodings::MONO8 || 
            encoding == sensor_msgs::image_encodings::BAYER_RGGB8) {
            image_msg->step = img.width;
        } else if (encoding == sensor_msgs::image_encodings::RGB8 ||
                  encoding == sensor_msgs::image_encodings::BGR8) {
            image_msg->step = img.width * 3;
        } else if (encoding == sensor_msgs::image_encodings::MONO16) {
            image_msg->step = img.width * 2;
        } else {
            image_msg->step = img.width;
        }
        
        image_msg->data.resize(img.dataSize);
        memcpy(image_msg->data.data(), img.data, img.dataSize);

        publisher_->publish(std::move(image_msg));
    }

    void apply_settings() {
        if (!camera_.is_open()) return;
        
        bool ok = true;
        auto try_set = [&](const std::string &name, bool success) {
            if (!success) {
                ok = false;
                RCLCPP_WARN(get_logger(), "%s setting failed: %s", name.c_str(), camera_.get_last_error().c_str());
            }
        };

        try_set("Exposure", camera_.set_exposure(exposure_));
        try_set("Gain", camera_.set_gain(gain_));
        try_set("Trigger mode", camera_.set_trigger_mode(trigger_));
        if (frame_rate_ > 0.0) {
            try_set("Frame rate", camera_.set_frame_rate(frame_rate_));
        }
        if (camera_.is_pixel_format_supported(pixel_format_)) {
            try_set("Pixel format", camera_.set_pixel_format(pixel_format_));
        } else {
            RCLCPP_WARN(get_logger(), "Pixel format 0x%08X not supported, using default", pixel_format_);
            unsigned int default_format = PixelType_Gvsp_BayerRG8;
            if (camera_.set_pixel_format(default_format)) {
                pixel_format_ = default_format;
                RCLCPP_INFO(get_logger(), "Using default pixel format: 0x%08X", default_format);
            }
        }
        
        settings_applied_ = ok;
        if (ok) {
            RCLCPP_INFO(get_logger(), "Camera settings applied successfully");
        }
    }

    std::string to_encoding(unsigned int format) const {
        switch (format) {
        case PixelType_Gvsp_Mono8: 
            RCLCPP_INFO_ONCE(get_logger(), "Camera publishing MONO8 format");
            return sensor_msgs::image_encodings::MONO8;
        case PixelType_Gvsp_Mono10:
        case PixelType_Gvsp_Mono12:
        case PixelType_Gvsp_Mono14:
        case PixelType_Gvsp_Mono16: 
            RCLCPP_INFO_ONCE(get_logger(), "Camera publishing MONO16 format");
            return sensor_msgs::image_encodings::MONO16;
        case PixelType_Gvsp_RGB8_Packed: 
            RCLCPP_INFO_ONCE(get_logger(), "Camera publishing RGB8 format");
            return sensor_msgs::image_encodings::RGB8;
        case PixelType_Gvsp_BGR8_Packed: 
            RCLCPP_INFO_ONCE(get_logger(), "Camera publishing BGR8 format");
            return sensor_msgs::image_encodings::BGR8;
        case PixelType_Gvsp_RGB10_Packed:
        case PixelType_Gvsp_BGR10_Packed:
        case PixelType_Gvsp_RGB12_Packed:
        case PixelType_Gvsp_BGR12_Packed:
        case PixelType_Gvsp_RGB16_Packed:
        case PixelType_Gvsp_BGR16_Packed: 
            RCLCPP_INFO_ONCE(get_logger(), "Camera publishing RGB16 format");
            return sensor_msgs::image_encodings::RGB16;
        case PixelType_Gvsp_BayerGR8: 
            RCLCPP_INFO_ONCE(get_logger(), "Camera publishing BAYER_GRBG8 format");
            return sensor_msgs::image_encodings::BAYER_GRBG8;
        case PixelType_Gvsp_BayerRG8: 
            RCLCPP_INFO_ONCE(get_logger(), "Camera publishing BAYER_RGGB8 format");
            return sensor_msgs::image_encodings::BAYER_RGGB8;
        case PixelType_Gvsp_BayerGB8: 
            RCLCPP_INFO_ONCE(get_logger(), "Camera publishing BAYER_GBRG8 format");
            return sensor_msgs::image_encodings::BAYER_GBRG8;
        case PixelType_Gvsp_BayerBG8: 
            RCLCPP_INFO_ONCE(get_logger(), "Camera publishing BAYER_BGGR8 format");
            return sensor_msgs::image_encodings::BAYER_BGGR8;
        default: 
            RCLCPP_WARN_ONCE(get_logger(), "Unknown pixel format 0x%08X, using default BAYER_RGGB8", format);
            return sensor_msgs::image_encodings::BAYER_RGGB8;
        }
    }

    HikCameraController camera_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_;

    double exposure_;
    double gain_;
    bool trigger_;
    double frame_rate_;
    unsigned int pixel_format_;
    std::string frame_id_;
    std::string serial_number_;
    
    bool settings_applied_ = false;
    rclcpp::Time last_attempt_;
    rclcpp::Time last_fps_time_;
    unsigned int consecutive_failures_ = 0;
    unsigned int frame_count_ = 0;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<HikCameraNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

