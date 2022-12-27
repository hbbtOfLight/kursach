#include "network.h"
#include <fstream>


const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float NMS_THRESHOLD = 0.4;
//const float CONFIDENCE_THRESHOLD = 0.35;

std::vector<std::string> LoadClassList(const std::string& path) {
    std::vector<std::string> class_list;
    std::ifstream ifs(path);
    std::string line;
    while (ifs.peek() != std::istream::traits_type::eof()) {
      getline(ifs, line);
      class_list.push_back(line);
    }
    return class_list;
}

std::vector<cv::Scalar> GenerateClassColors(size_t size) {
    static std::mt19937 gen(time(0));
    static std::uniform_int_distribution dis(0, 255);
    std::vector<cv::Scalar> colors(size);
    for (cv::Scalar& c : colors) {
        int red = dis(gen);
        int green = dis(gen);
        int blue = dis(gen);
        c = cv::Scalar(red, green, blue);
    }
    return colors;
}

struct Detection {
  int class_id;
  float confidence;
  cv::Rect box;
};


cv::Mat FormatForYolo(const cv::Mat& source) {
  int col = source.cols;
  int row = source.rows;
  int _max = MAX(col, row);
  cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
  source.copyTo(result(cv::Rect(0, 0, col, row)));
  return result;
}

void Network::Detect(cv::Mat& image,
            std::vector<Detection>& output, float confidence_basic, float class_threshold) {
  cv::Mat blob;

  auto input_image = FormatForYolo(image);

  cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
  std::cerr << blob.size << " blob size\n";
 net_.setInput(blob);
  std::vector<cv::Mat> outputs;
  net_.forward(outputs, net_.getUnconnectedOutLayersNames());

  float x_factor = input_image.cols / INPUT_WIDTH;
  float y_factor = input_image.rows / INPUT_HEIGHT;
  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  const int dimensions = 85;
  int rows = 25200;
  std::cerr << outputs[0].size << "\n" << outputs[1].size << "\n" << outputs[2].size << "\n" << outputs[3].size << "\n";
  float* data = (float*) outputs[3].data;  

  for (int i = 0; i < rows; ++i) {
    float confidence = data[4];
    if (confidence >= confidence_basic) {
      float* classes_scores = data + 5;
      cv::Mat scores(1, class_list_.size(), CV_32FC1, classes_scores);
      cv::Point class_id;
      double max_class_score;
      minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
      std::cerr << "Max class score: " << max_class_score << "\n";
      if (max_class_score > class_threshold) {
          std::cerr << max_class_score << "!\n";

        confidences.push_back(confidence);

        class_ids.push_back(class_id.x);

        float x = data[0];
        float y = data[1];
        float w = data[2];
        float h = data[3];
        int left = int((x - 0.5 * w) * x_factor);
        int top = int((y - 0.5 * h) * y_factor);
        int width = int(w * x_factor);
        int height = int(h * y_factor);
        boxes.push_back(cv::Rect(left, top, width, height));
      }

    }

    data += dimensions;

  }

  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(boxes, confidences, class_threshold, NMS_THRESHOLD, nms_result);
  for (int i = 0; i < nms_result.size(); i++) {
    int idx = nms_result[i];
    Detection result;
    result.class_id = class_ids[idx];
    result.confidence = confidences[idx];
    result.box = boxes[idx];
    output.push_back(result);
  }
}



Network::Network(const std::string& class_list_path, const std::string& network_path) {
    class_list_ = LoadClassList(class_list_path);
    net_ = cv::dnn::readNet(network_path);
    class_colors_ = GenerateClassColors(class_list_.size());
}

void Network::GetDetection(cv::Mat &input, float confidence, float class_threshold)
{
    std::vector<Detection> output;
    Detect(input, output, confidence, class_threshold);
    for (int i = 0; i < output.size(); ++i) {

      auto detection = output[i];
      auto box = detection.box;
      const auto color = class_colors_[detection.class_id];
      cv::rectangle(input, box, color, 3);

      cv::rectangle(input, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
      cv::putText(input,
                  class_list_[detection.class_id].c_str(),
                  cv::Point(box.x, box.y - 5),
                  cv::FONT_HERSHEY_SIMPLEX,
                  0.5,
                  cv::Scalar(0, 0, 0));
    }

}
