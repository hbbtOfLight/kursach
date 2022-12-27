#ifndef NETWORK_H
#define NETWORK_H
#include <opencv2/opencv.hpp>
#include <random>

struct Detection;
class Network
{
public:
    Network(const std::string& class_list_path, const std::string& network_path);
    void GetDetection(cv::Mat& input, float confidence = 0.35, float class_threshold = 0.2);

private:
    std::vector<std::string> class_list_;
    std::vector<cv::Scalar> class_colors_;
    cv::dnn::Net net_;
    void Detect(cv::Mat& image,
                std::vector<Detection>& output, float confidence, float class_threshold);
};

#endif // NETWORK_H
