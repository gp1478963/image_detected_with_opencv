#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>
#include <string>
#include <format>
#include <math.h>
#include "augment.h"
using namespace std;
using namespace cv;
using namespace ::filesystem;
using namespace augment;

constexpr float IOU_THRESHOLD = 0.7;
constexpr int32_t GRID_CELL = 7;
constexpr float GRID_CELL_SIZE= 1.0 / GRID_CELL;
const vector<string> CLASSIERS = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

void ReadImage(const std::string& imagePath, Mat& imageOriginal) {
    if(!::exists(imagePath)) {
        throw std::runtime_error(std::format("Image path {} not exist.", imagePath));
     }
     imageOriginal = imread(imagePath, IMREAD_COLOR);
}

void LoadModelFromOnnx(const std::string& path, dnn::Net& net) {
    if(!::exists(path)) {
        throw std::runtime_error(std::format("Onnx path {} not exist.", path));
    }
    net = dnn::readNetFromONNX(path);
}

void UnCode(std::vector<float>& boxes, int width, int height) {
//    static_assert(boxes.size() >= 4, "");
    boxes[0] = std::min(boxes[0] - boxes[2]/2., 1.0); // x_c
    boxes[1] = std::min(boxes[1] - boxes[3]/2., 1.0);
    boxes[2] = std::min(boxes[0] + boxes[2], (float)1.0)*width; // x_c
    boxes[3] = std::min(boxes[1] + boxes[3], (float)1.0)*height;
    boxes[0] *= width;
    boxes[1] *= height;
}

template<typename T>
int argMax(const T& array, int begin, int end) {
    int index = 0;
    float maxNum = 0;
    for(int i = begin; i < end; i++) {
        if(maxNum < array[i]) {
            maxNum = array[i];
            index = i;
        }
    }
    return index;
}

void ArgSort(std::vector<float>& array, std::vector<int>& order) {
    for(int i = 0; i < array.size()-1; i++) {
        for(int j = i + 1; j < array.size(); j++) {
            if(array[i] < array[j]) {
                auto tempScore = array[i];
                auto tempIndex = order[i];
                array[i] = array[j];
                order[i] = order[j];
                array[j] = tempScore;
                order[j] = tempIndex;
            }
        }
    }
}

float CalcIOU(const std::vector<float>& box1,const std::vector<float>& box2) {
    float box1Area = (box1[3] - box1[1])*(box1[2] - box1[0]);
    float box2Area = (box2[3] - box2[1])*(box2[2] - box2[0]);
    float leftX = std::max(box1[0], box2[0]);
    float leftY = std::max(box1[1], box2[1]);
    float rightX = std::min(box1[2], box2[2]);
    float rightY = std::min(box1[3], box2[3]);
    float areaUnion = std::max(rightX - leftX, (float )0.0) * std::max(rightY - leftY, (float )0);
    std::cout << areaUnion << "\t" << box1Area << "\t" << box2Area << "\n";
    return areaUnion / (box1Area + box2Area - areaUnion);
}

std::vector<float> CalcIOUs(const std::vector<float>& box1,const std::vector<std::vector<float>>& box2es) {
    std::vector<float> IOUs;
    for(const auto& box2: box2es) {
        IOUs.emplace_back(CalcIOU(box1, box2));
    }
    return IOUs;
}



std::vector<int> NMS(std::vector<std::vector<float>> boxes, float iouThreshold)  {
    std::vector<int> order;
    std::vector<float> scores;
    // 提取scores
    for(int i = 0; i < boxes.size(); i++) {
        scores.push_back(boxes[i][4]);
        order.push_back(i);
    }
    // 根据socres排序
    ArgSort(scores, order);
//    for(const auto& index : order)  {
//        std::cout << index << std::endl;
//    }
    // 根据keep 非极大值抑制

    for(int i = 0; i < order.size() - 1; i++) {
        if(order[i] == -1) continue;
        auto IOUs = CalcIOUs(boxes[order[i]], boxes);

        for(const auto& iou : IOUs)  {
            std::cout << iou << std::endl;
        }

        for(int j = 0; j < order.size(); j++) {
            if(IOUs[order[j]] > iouThreshold && order[j] != -1 && i != j) {
                order[j] = -1;
            }
        }
    }

    std::vector<int> keep;
    for(const auto& index : order) {
        if(index != -1) {
            keep.push_back(index);
        }
    }
    return keep;
}

void ParseCoordarance(Mat& prediction, std::vector<std::vector<float>>& predictBoxes) {
    prediction.convertTo(prediction, CV_32F);
    auto ty = prediction.type() ;
    auto N = prediction.size[0];
    auto width = prediction.size[1];
    auto height = prediction.size[2];
    auto depth = prediction.size[3];
    for (int grid_x = 0; grid_x < width; ++grid_x) {
        for (int grid_y = 0; grid_y < height; ++grid_y) {
            auto grid_pred = prediction.at<Vec<float , 30>>(0,grid_x, grid_y);
            auto index = grid_pred[4] >= grid_pred[9] ? 0: 1;
            if(grid_pred[index*5 + 4] < IOU_THRESHOLD) continue;
            std::vector<float> box; box.resize(6);
            box[0] = grid_pred[index*5 + 0]*GRID_CELL_SIZE + grid_x * GRID_CELL_SIZE; // x_c
            box[1] = grid_pred[index*5 + 1]*GRID_CELL_SIZE + grid_y * GRID_CELL_SIZE;
            box[2] = grid_pred[index*5 + 2] * grid_pred[index*5 + 2];
            box[3] = grid_pred[index*5 + 3] * grid_pred[index*5 + 3];
            box[4] = grid_pred[index*5 + 4];
            box[5] = (float )argMax(grid_pred, 10, 30) - 10;
            std::cout << std::format("score:{0:.4f}\tx<{1}> y<{2}> w<{3}> h<{4}> classier{5}\n", box[4], box[0], box[1], box[2], box[3], box[5]);
            predictBoxes.push_back(box);
        }
    }
}

std::vector<std::vector<float>> DoInference(Mat& image, dnn::Net& inferenceModel){
    auto blob = dnn::blobFromImage(image);
    inferenceModel.setInput(blob);
    inferenceModel.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    inferenceModel.setPreferableTarget(dnn::DNN_TARGET_CPU);
    vector<Mat> prediction;
    inferenceModel.forward(prediction);
    std::cout << std::format("dimension of prediction: width<{}>,height<{}>,depth<{}>\n",
                             prediction[0].size[1],prediction[0].size[2], prediction[0].size[3]);
    std::vector<std::vector<float>> predictBoxes;
    ParseCoordarance(prediction[0], predictBoxes);
    for (auto& box: predictBoxes) {
        UnCode(box, image.cols, image.rows);
        std::cout << std::format("score:{0:.4f}\tx<{1}> y<{2}> w<{3}> h<{4}> \n", box[4], box[0], box[1], box[2], box[3]);
    }
    return predictBoxes;
}

void DrawRectOnImage(const std::vector<float>& boxes, Mat& image) {
    rectangle(image,Point((int)boxes[0], (int)boxes[1]), Point((int)boxes[2], (int)boxes[3]),  (0, 255, 255), 3);
    putText(image, std::format("{}:{}",CLASSIERS[(int)boxes[5]], boxes[4]) ,Point((int)boxes[0], (int)boxes[1]),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255), 1, 8);
}

int main()
{
    cout << "launching..." << endl;
    Mat imageOriginal;
    dnn::Net YOLO_V1_Net;
    try
    {
        ReadImage("../000112.jpg", imageOriginal);
        // 前处理

//        augment::TransFormerDo transFormerDo({
//                                                     make_shared<TransFormerResize>(448, 447),
//                                                     make_shared<TransFormerNormalize>() });

        TransFormerResize(448, 447)(imageOriginal);
        auto image = imageOriginal.clone();
        TransFormerNormalize()(imageOriginal);
//        transFormerDo(imageOriginal);
        std::cout << imageOriginal.at<Vec3f>(0,0) << std::endl;
        LoadModelFromOnnx("../lmodel_cpu.onnx", YOLO_V1_Net);
        auto boxes = DoInference(imageOriginal, YOLO_V1_Net);
        float NMS_IOU_THRESHOLD=0.01;
        auto keep = NMS(boxes, NMS_IOU_THRESHOLD);
        for (const auto& index: keep) {
            DrawRectOnImage(boxes[index], image);
        }
        imshow("s", image);
        waitKey(0);
        imwrite("./112.jpg", image);
    }
    catch (std::exception& error) {
        cout << error.what() << endl;
    }
    return 0;
}