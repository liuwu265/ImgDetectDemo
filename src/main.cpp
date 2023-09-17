#include <iostream>
#include <vector>
#include <getopt.h>

#include <opencv2/opencv.hpp>

#include "inference.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    std::string projectBasePath = "/Users/wave/PycharmProjects/ImgDetectDemo"; // Set your ultralytics base path
    std::string modelPath = projectBasePath + "/model/yolov8m.onnx";
    bool runOnGPU = false;

    Inference inf(modelPath, cv::Size(640, 480), "classes.txt", runOnGPU);
    for (int i = 0; i < 5; ++i)
    {
        std::string imgFile = projectBasePath + "/img/" + std::to_string(i) + ".jpg";
        std::cout << "start inferences img: " << imgFile << std::endl;
        cv::Mat frame = cv::imread(imgFile);

        // Inference starts here...
        std::vector<Detection> output = inf.runInference(frame);

        int detections = output.size();
        std::cout << "Number of detections:" << detections << std::endl;

        for (int i = 0; i < detections; ++i)
        {
            Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

            // Detection box
            cv::rectangle(frame, box, color, 2);

            // Detection box text
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            cv::rectangle(frame, textBox, color, cv::FILLED);
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }
        // Inference ends here...

        // This is only for preview purposes
        float scale = 0.8;
        cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
        cv::imshow("Inference", frame);

        std::string saveFile = projectBasePath + "/img/res/out_" + std::to_string(i) + ".jpg";
        cv::imwrite(saveFile, frame);
        cv::waitKey(-1);
    }
}
