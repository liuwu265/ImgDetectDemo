

This example demonstrates how to perform inference using YOLOv8 in C++ with ONNX Runtime and OpenCV's API.

## 安装c++ opencv
opencv安装教程1: [参考教程1](https://docs.opencv.org/4.x/df/d65/tutorial_table_of_content_introduction.html)

opencv安装教程2(从opencv源码安装): [参考教程2](https://thecodinginterface.com/blog/opencv-cpp-vscode/)


## 编译流程
说明：编译前，先查看代码中用到的模型，图片等文件路径是否正确。主要检查的内容
1. main.cpp文件中，项目路径是否正确，项目路径下面的model, img的路径是否有模型及图片文件；
2. CMakeLists.txt中安装的opencv的路径(OpenCV_DIR)是否正确；

```bash
# 进入项目目录
cd xx/ImgDetectDemo

# 开始编译，最终的图片结果保存在img/res目录。
mkdir build
cd build
cmake ../src
make
./Yolov8CPPInference
```
