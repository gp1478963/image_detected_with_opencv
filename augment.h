//
// Created by gp147 on 1/3/2024.
//

#ifndef UNTITLED1_AUGMENT_H
#define UNTITLED1_AUGMENT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

namespace augment {
    class TransFormerInterface
    {
    public:
        virtual void operator()(cv::Mat& image) = 0;
    };

    class TransFormerResize: public TransFormerInterface
    {
    public:
        TransFormerResize(int32_t width, int32_t height);
        void operator()(cv::Mat& image) override;
    private:
        int32_t _width;
        int32_t _height;
    };

    struct TransFormerNormalize: public TransFormerInterface
    {
        void operator()(cv::Mat& image) override;
    };

    struct TransFormerDo: public TransFormerInterface
    {
        explicit TransFormerDo(std::initializer_list<std::shared_ptr<TransFormerInterface>> transformerGroup);
        void operator()(cv::Mat& image) override;
        std::initializer_list<std::shared_ptr<TransFormerInterface>> transformerGroup;
    };

}










#endif //UNTITLED1_AUGMENT_H
