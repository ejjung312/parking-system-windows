#include <opencv2/opencv.hpp>
#include <filesystem>

#include "yolo.h"
#include "centroidtracker.h"
#include "common_functions.h"

//#define CUDA_ACC
#define OPENCL_ACC

std::set<int> vehicles = { 2, 3, 5, 7 }; // 차량 클래스 ID 목록

int main() {
    std::string car_model_path = GetResourcePath("yolo11n.onnx");
    std::string license_model_path = GetResourcePath("license_plate_best.onnx");
    std::string input_path = GetResourcePath("car2.mp4");

    //std::string output_path = input_path.substr(0, input_path.find_last_of(".")) + "_out.mp4";

    auto centroidTracker = new CentroidTracker(20);

    cv::VideoCapture cap(input_path);
    assert(cap.isOpened() && "Error: Cannot open video file");

    /*int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);*/

    /*cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
    assert(writer.isOpened() && "Error: Cannot open output video file");*/

    Yolo11 car_model(car_model_path, 0.25f, 0.25f,
        [](int lbl_id, const std::string lbl)
        { return lbl_id >= 0 && lbl_id <= 8; });

    Yolo11 license_model(license_model_path, 0.25f, 0.25f,
        [](int lbl_id, const std::string lbl)
        { return lbl_id >= 0 && lbl_id <= 8; },
        GetResourcePath("license.names"));

    cv::Mat frame;
    while (cap.read(frame))
    {
        std::vector<ObjectBBox> bbox_1 = car_model.detect(frame);

        std::vector<ObjectBBox> boxes;
        for (auto& bbox : bbox_1)
        {
            if (vehicles.find(bbox.class_id) != vehicles.end()) // vehicles.end(): 탐색 실패. 존재하지 않음
            {
                // 마지막 위치에 bbox 추가
                boxes.insert(boxes.end(), bbox);

                bbox.draw(frame, cv::Scalar(0, 255, 0));
            }

            /*std::cout << "Label:" << bbox.label << " Conf: " << bbox.conf;
            std::cout << "(" << bbox.x1 << ", " << bbox.y1 << ") ";
            std::cout << "(" << bbox.x2 << ", " << bbox.y2 << ")" << std::endl;*/
        }

        // track vehicles
        auto objects = centroidTracker->update(boxes);

        if (!objects.empty()) {
            // 차번호판 감지
            std::vector<ObjectBBox> bbox_2 = license_model.detect(frame);

            for (auto& bbox : bbox_2)
            {
                std::cout << "Label:" << bbox.label << " Conf: " << bbox.conf;
                std::cout << "(" << bbox.x1 << ", " << bbox.y1 << ") ";
                std::cout << "(" << bbox.x2 << ", " << bbox.y2 << ")" << std::endl;
                
                bbox.draw(frame, cv::Scalar(255, 255, 0));
            }
        }

        cv::imshow("result", frame);
        //writer.write(frame);

        char key = cv::waitKey(1);
        if (key == 27 || key == 'q')
            break;
    }

    cap.release();
    //writer.release();
    cv::destroyAllWindows();

    //std::cout << "Video saved as: " << output_path << std::endl;

    return 0;
}