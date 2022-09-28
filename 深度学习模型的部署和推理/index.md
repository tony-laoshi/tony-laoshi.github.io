# 深度学习模型的部署和推理

<br/>
<br/>

## 简介
介绍深度学习训练好的模型在C++中如何部署以及使用不同的推理引擎进行推理。这里是用Pytorch训练好的模型为例（用什么学习框架学习得到的模型没关系，都有对应的模块导出模型的）。PyTorch 模型经常需要部署在 C++ 程序中，目前我知道的方法有三种（还有其他方式我没有尝试过）：

- LibTorch: PyTorch 官方 C++ 库
- ONNX Runtime
- OpenCV: DNN 模块
<br/>
<br/>

## ONNX Runtime
ONNX Runtime 是一个跨平台推理和训练机器学习加速器。也就是说，训练好的模型导出为 ONNX 格式后，可以用 ONNX Runtime 进行推理。
<br/>
<br/>

### 模型导出
**ONNX**（Open Neural Network Exchange）是一种针对机器学习所设计的开放式的文件格式，用于存储训练好的模型。它使得不同的人工智能框架（如Pytorch、MXNet）可以采用相同格式存储模型数据并交互。 ONNX的规范及代码主要由微软，亚马逊，Facebook和IBM等公司共同开发，以开放源代码的方式托管在Github上。目前官方支持加载ONNX模型并进行推理的深度学习框架有： Caffe2, PyTorch, MXNet，ML.NET，TensorRT 和 Microsoft CNTK，并且 TensorFlow 也非官方的支持ONNX。
使用`torch.onnx.export `可以直接保存模型为 onnx 格式，实例代码如下：
```python
def export(model, inputL=None, inputR=None):
    import onnx
    
    device = torch.device('cpu')
    model = model.module.to(device)
    model.eval()
    
    if inputL is None:
        dummyL = torch.randn((1, 3, 368, 1232)).to(device)
    else:
        dummyL = inputL.to(device)
        
        if inputR is None:
            dummyR = torch.randn((1, 3, 368, 1232)).to(device)
        else:
            dummyR = inputR.to(device)
            
            onnx_path = "anynet2.onnx"
            # onnx_path = "anynet_with_spn.onnx"
            torch.onnx.export(
                model,
                (dummyL, dummyR),
                onnx_path,
                opset_version=11,
                input_names=["imgL", "imgR"],
                output_names=["disp1", "disp2", "disp3",]
                # output_names=["disp1", "disp2", "disp3", "disp4"]
            )
            
            onnx.checker.check_model(onnx.load(onnx_path))
            print("ONNX model exported to: " + onnx_path)

```
<br/>
<br/>

### 推理模型
如果使用 ONNX Runtime 进行推理的话，事先还需要安装 ONNX Runtime（使用Python 的话比较简单，使用C++的话略微复杂）
<br/>
<br/>

#### ONNX Runtime 安装（C++）
安装的方式一般分从源码编译安装（可以参考这个，流程基本一致，[MacOS源码编译onnxruntime](https://zhuanlan.zhihu.com/p/411887386)）和使用预编译好的包 。
[官网](https://onnxruntime.ai/)上的的Linux预编译包下载指引让人摸不着头脑，这里建议直接从[Github Release](https://github.com/microsoft/onnxruntime/releases/)页面下载。解压后你会发现预编译包里除了一些文档之外，只有头文件和二进制的库文件，没有任何包管理相关 (CMake、pkg-config 之类) 的配置文件。虽然[源码 CMakeLists](https://github.com/microsoft/onnxruntime/blob/96bb4b1ce83efd13b7dba54f707b27303354e480/cmake/CMakeLists.txt#L1738-L1748) 中明明有 pkg-config 配置文件 .pc 的生成，但不知为何并没有被打包进预编译包。总之，拿到预编译包你没法直接通过 CMake/pkg-config 引入自己的工程。
所以，在 CMake 项目中无法通过 find_package 找到 ONNX Runtime。可以仿照[这个仓库](https://github.com/leimao/ONNX-Runtime-Inference)，使用 find_path 和 find_library 来查找：
<br/>

```cmake
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h
  HINTS /usr/local/include/onnxruntime/core/session/)
find_path(ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS cuda_provider_factory.h
  HINTS /usr/local/include/onnxruntime/core/providers/cuda/)
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)

add_executable(inference inference.cpp)
target_include_directories(inference PRIVATE
  ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS}
  ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS})
target_link_libraries(inference PRIVATE ${ONNX_RUNTIME_LIB})
```
<br/>

分别指定了包含路径、库路径、链接库。
或者也可以在预编译包的根目录下建立 share/cmake/onnxruntime 文件夹，在里面创建 onnxruntimeConfig.cmake 文件，内容为：

```cmake

#This will define the following variables:
#   onnxruntime_FOUND        -- True if the system has the onnxruntime library
#   onnxruntime_INCLUDE_DIRS -- The include directories for onnxruntime
#   onnxruntime_LIBRARIES    -- Libraries to link against
#   onnxruntime_CXX_FLAGS    -- Additional (required) compiler flags

include(FindPackageHandleStandardArgs)

# Assume we are in <install-prefix>/share/cmake/onnxruntime/onnxruntimeConfig.cmake
get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(onnxruntime_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

set(onnxruntime_INCLUDE_DIRS ${onnxruntime_INSTALL_PREFIX}/include)
set(onnxruntime_LIBRARIES onnxruntime)
set(onnxruntime_CXX_FLAGS "") # no flags needed


find_library(onnxruntime_LIBRARY onnxruntime
    PATHS "${onnxruntime_INSTALL_PREFIX}/lib"
)

add_library(onnxruntime SHARED IMPORTED)
set_property(TARGET onnxruntime PROPERTY IMPORTED_LOCATION "${onnxruntime_LIBRARY}")
set_property(TARGET onnxruntime PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_INCLUDE_DIRS}")
set_property(TARGET onnxruntime PROPERTY INTERFACE_COMPILE_OPTIONS "${onnxruntime_CXX_FLAGS}")

find_package_handle_standard_args(onnxruntime DEFAULT_MSG onnxruntime_LIBRARY onnxruntime_INCLUDE_DIRS)

```
<br/>

然后将预编译包的 include、lib、share 三个文件夹拷到系统路径，或者[注册用户包](https://blog.csdn.net/mightbxg/article/details/114089740?spm=1001.2014.3001.5501)，就能在自己 CMake 项目中使用 find_package 找到 onnxruntime 了：
```cmake
# 增加opencv的依赖
FIND_PACKAGE( OpenCV REQUIRED )
# onnxruntime
find_package( onnxruntime REQUIRED)

include_directories(${onnxruntime_INCLUDE_DIRS})

ADD_EXECUTABLE( main main.cpp )

TARGET_LINK_LIBRARIES( main 
${OpenCV_LIBS} 
${onnxruntime_LIBRARIES}
)
```


#### 使用 ONNX Runtime 推理
这里附上一段在C++中使用 ONNX Runtime 进行推理的代码。其中一定要注意进行推理之前，要进行正确的图像预处理。同时，推理得到的Tensor 之后，如需转换成其他格式，也需要进行合适的后处理。在我这段例子当中，是将模型推理得到的Tensor 转换成视差图也就是cv::Mat 格式
```cpp
#include <algorithm>  
#include <assert.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <png.h>
#include <experimental_onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <stdio.h>
#include <torch/torch.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace std;
using namespace cv;
using namespace torch;



// pretty prints a shape dimension vector
std::string print_shape(const std::vector<int64_t>& v) {
    std::stringstream ss("");
    for (size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

// image pre-process
Mat imagePreprocess(Mat sourceImg, Mat preprocessedImage, std::vector<int64_t> inputDims){
    // step 1: Resize the image.
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage;
    cv::resize(sourceImg, resizedImageBGR,
               cv::Size(inputDims.at(3), inputDims.at(2)),
               cv::InterpolationFlags::INTER_CUBIC);
    
    // step 2: Convert the image to HWC RGB UINT8 format.
    cv::cvtColor(resizedImageBGR, resizedImageRGB,
                 cv::ColorConversionCodes::COLOR_BGR2RGB);
    // step 3: Convert the image to HWC RGB float format by dividing each pixel by 255.
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);
    
    // step 4: Split the RGB channels from the image.   
    cv::Mat channels[3];
    cv::split(resizedImage, channels);
    
    //step 5: Normalize each channel.
    // Normalization per channel
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;
    
    //step 6: Merge the RGB channels back to the image.
    cv::merge(channels, 3, resizedImage);
    
    //step 7: Convert the image to CHW RGB float format.
    //HWC to CHW
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);
    return preprocessedImage;
}



int main(int argc, char** argv) {
    
    // set path
    string modelFile = "/home/standard/dengqi/codes/ONNX_testDemo/anynet2.onnx"; 
    string leftImageFile = "/home/standard/dengqi/codes/ONNX_testDemo/kittiTestImages/imgL/000000.png";
    string rightImageFile = "/home/standard/dengqi/codes/ONNX_testDemo/kittiTestImages/imgR/000000.png";
    
    
    // onnxruntime setup
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "anynet");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Experimental::Session session = Ort::Experimental::Session(env, modelFile, session_options);
    Ort::AllocatorWithDefaultOptions allocator;
    
    
    // print name/shape of inputs
    auto input_names = session.GetInputNames();
    auto input_shapes = session.GetInputShapes();
    auto inputDims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    cout << "Input Node Name/Shape (" << input_names.size() << "):" << endl;
    for (size_t i = 0; i < input_names.size(); i++) {
        cout << "\t" << input_names[i] << " : " << print_shape(input_shapes[i]) << endl;
    }
    
    // print name/shape of outputs
    auto output_names = session.GetOutputNames();
    auto output_shapes = session.GetOutputShapes();
    cout << "Output Node Name/Shape (" << output_names.size() << "):" << endl;
    for (size_t i = 0; i < output_names.size(); i++) {
        cout << "\t" << output_names[i] << " : " << print_shape(output_shapes[i]) << endl;
    }
    
    assert(input_names.size() == 2 && output_names.size() == 3);
    
    
    // ************** Load input images ***************************************************
    Mat sourceImageL = imread(leftImageFile, 1); 
    Mat sourceImageR = imread(rightImageFile, 1);
    cv::imshow("imageL", sourceImageL);
    
    // Pre-processing input images
    Mat imgL, imgR;
    imgL = imagePreprocess(sourceImageL, imgL, inputDims);
    imgR = imagePreprocess(sourceImageR, imgR, inputDims);
    
    
    
    
    //*************** Create Ort tensors for inputs ***************************************
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    
    Ort::Value tensor1 = Ort::Value::CreateTensor<float>(memory_info, imgL.ptr<float>(), imgL.total(), input_shapes[0].data(), input_shapes[0].size());
    assert(tensor1.IsTensor());
    Ort::Value tensor2 = Ort::Value::CreateTensor<float>(memory_info, imgR.ptr<float>(), imgR.total(), input_shapes[1].data(), input_shapes[1].size());   
    assert(tensor2.IsTensor());
    
    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(std::move(tensor1));
    input_tensors.emplace_back(std::move(tensor2));
    
    //*************** Forward ***************************************************************
    cout << "Running model...";
    
    try 
    {
        std::vector<Ort::Value> output_tensors;
        auto start = chrono::system_clock::now();
        output_tensors = session.Run(input_names, input_tensors, output_names);
        auto end   = chrono::system_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        cout <<  " Successfully inferencing, taking  " 
            << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << " s" << endl;
        cout << "done" << endl;
        
        // Convert to cv::Mat and show
        for (int i=0; i<3; i++){
            // Get mutable data
            float* output_tensor_data = output_tensors[i].GetTensorMutableData<float>();
            cv::Mat dispMap(368, 1232, CV_32F, (void*)output_tensor_data); 
            
            //colorise and show
            Mat colorised_dispMap;
            cv::normalize(dispMap, dispMap, 255.0, 0.0, NORM_MINMAX);
            dispMap.convertTo(dispMap, CV_8U);
            cv::applyColorMap(dispMap, colorised_dispMap, cv::COLORMAP_JET);
            string window_name = "disparity_map_" + std::to_string(i);
            cv::imshow(window_name, colorised_dispMap);
        }
        cv::waitKey(0);
        
    } 
    catch (const Ort::Exception& exception) 
    {
        cout << "ERROR running model inference: " << exception.what() << endl;
        exit(-1);
    }
}	

```
<br/>

若感兴趣完整的推理过程，可以直接在我的[仓库](https://github.com/tony-laoshi/AnyNet)，获取测试用的代码（C++, Python都有），以及已经保存好为onnx格式的网络模型。
<br/>
<br/>
<br/>

## OpenCV DNN
参见[官方文档](https://docs.opencv.org/4.5.2/dd/d55/pytorch_cls_c_tutorial_dnn_conversion.html)，OpenCV DNN 模块能够直接加载 onnx 格式的模型，使用这个模块可以直接进行推理。但是缺点是有一些网络中自定义的操作，可能不被opencv dnn模块所支持。实例代码如下：
<br/>

```cpp
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace dnn;

int main()
{
    const bool use_cuda = false;
    const std::string fn_image = "cat.jpg";
    const std::string fn_model = "super_resolution.onnx";

    // load and config model
    Net net = readNetFromONNX(fn_model);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // source image
    auto image = imread(fn_image, cv::IMREAD_GRAYSCALE);
    Mat blob;
    blobFromImage(image, blob, 1.0 / 255.0);

    // inference and output
    net.setInput(blob);
    auto output = net.forward();
    int new_size[] = { output.size[2], output.size[3] };
    output = output.reshape(1, 2, new_size);
    convertScaleAbs(output, output, 255.0);
}

```
<br/>
<br/>


## Libtorch
### 模型导出
如果使用 LibTorch 进行推理的话，那么需要事先将模型导出为 Torch Script (.pt 格式) 文件，参见[官方教程](https://pytorch.org/tutorials/advanced/cpp_export.html)。有两种方法：Tracing 和 Scripting。
Tracing 就是提供一个示例输入，让 PyTorch 跑一遍整个网络，将过程中的全部操作记录下来，从而生成 Torch Script 模型：
<br/>
<br/>

```python
x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
traced_script_model = torch.jit.trace(torch_model, x)
traced_script_model.save("super_resolution.pt")
```
<br/>

Scripting 则是直接分析网络结构转化模型：
```python
traced_script_model = torch.jit.script(torch_model)
traced_script_model.save("super_resolution.pt")

```
<br/>
这两种方法各有优缺点：如果模型正向传播的控制流跟输入相关，显然 Tracing 只能得到一种输入下的控制流，此时应该用 Scripting；而当模型使用了一些 Torch Script 不支持的特性，同时模型源码又无法修改时（如果能访问源码，Scripting 可以通过加入注释的方法忽略它们），Scripting 便无能为力了，此时只能考虑 Tracing。更多关于 Tracing 和 Scripting 的区别可以参考 [Mastering TorchScript](https://paulbridger.com/posts/mastering-torchscript/)。
<br/>

### 推理部分
LibTorch 下载参见 [PyTorch: Get Started](https://pytorch.org/get-started/locally/)。CMake 工程中使用 LibTorch 只需要加入 find_package(Torch REQUIRED)，并将自己的可执行文件/库链接到 ${TORCH_LIBRARIES} 即可。具体进行推理的 C++ 代码如下：
```cpp
#include <opencv2/opencv.hpp>
#include <torch/script.h>

torch::Tensor toTensor(const cv::Mat& image)
{
    // convert 8UC1 image to 4-D float tensor
    CV_Assert(image.type() == CV_8UC1);
    return torch::from_blob(image.data,
                            { 1, 1, image.rows, image.cols }, torch::kByte)
        .toType(torch::kFloat32)
        .mul(1.f / 255.f);
}

cv::Mat toMat(const torch::Tensor& tensor)
{
    // convert tensor to 8UC1 image
    using namespace torch;
    Tensor t = tensor.mul(255.f).clip(0, 255).toType(kU8).to(kCPU).squeeze();
    CV_Assert(t.sizes().size() == 2);
    return cv::Mat(t.size(0), t.size(1), CV_8UC1, t.data_ptr()).clone();
}

int main()
{
    const bool use_cuda = false;
    const std::string fn_image = "cat.jpg";
    const std::string fn_model = "super_resolution.pt";
    
    // load model
    auto module = torch::jit::load(fn_model);
    if (use_cuda)
        module.to(torch::kCUDA);
    
    // load source image
    auto image = imread(fn_image, cv::IMREAD_GRAYSCALE);
    auto input = toTensor(image);
    if (use_cuda)
        input = input.to(torch::kCUDA);
    
    // inference
    auto output = module.forward({ input }).toTensor();
    auto result_torch = toMat(output);
    imwrite("result_torch.png", result_torch);
}

```
<br/>


# 参考链接
[pytorch模型部署](https://www.codeleading.com/article/30425879409/)
[深度学习之从 python 到 C++](https://www.guyuehome.com/35775)



