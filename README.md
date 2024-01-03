# image_detected_with_opencv
先使用pytorch框架搭建yolo模型进行训练，训练数据集是voc2007，训练完成后保存的模型为torch框架pt文件，再使用框架转换成onnx通用模型文件。这部分工作在自己的另一个python项目中完成。项目地址：https://github.com/gp1478963/yolov1-with-pytorch.git
权重参数在阿里云网盘中： https://www.alipan.com/s/bJYzPmSKvNo 提取码: 73nt
之后推理流程使用opencv中的dnn进行onnx模型加载工作。加载图片，前处理图片，forward，推理过程中使用的cpu会比较慢，batch size设置为1后模型输出维度（1，7，7，30）之后反编码进行NMS画框就可以了。
