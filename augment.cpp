//
// Created by gp147 on 1/3/2024.
//

#include "augment.h"

#include <utility>

using namespace cv;
using namespace std;

augment::TransFormerResize::TransFormerResize(int32_t width, int32_t height) : _width(width), _height(height) {}

void augment::TransFormerResize::operator()(cv::Mat &image)
{
    resize(image, image, {_width, _height}, 0, 0, INTER_CUBIC);
}

void augment::TransFormerNormalize::operator()(Mat &image)
{
    image.convertTo(image, CV_32F, 1.0/255);
}

augment::TransFormerDo::TransFormerDo(std::initializer_list<std::shared_ptr<TransFormerInterface>> transformerGroup)
        : transformerGroup(std::move(transformerGroup)) {}

void augment::TransFormerDo::operator()(Mat &image) {
    for(auto& translate: transformerGroup) {
        (*translate)(image);
    }
}
