# Opencv图像数据类型


<a name="XIptC"></a>
# 数据类型的形式
数据类型的形式一般为`CV_<bit_depth>(S|U|F)C<number_of_channels>`，其各自的含义如下：

- `bit_depth`：比特数，代表图像像素的位数，即像素深度，比如8bite、16bites、32bites、64bites
- `S|U|F`: **S**代表`signed int` 有符号整形。**U**代表 `unsigned int `无符号整形。**F**代表 `float` 单精度浮点型
- `C<number_of_channels>`: 代表一张图片的通道数比如:
   - channels = 1：灰度图片。是单通道图像
   - channels = 3：RGB彩色图像。是3通道图像
   - channels = 4：带**alpha**通道的RGB图像，表示透明度。是4通道图像

例如：CV_8U 代表的是像素位数为8，无符号数据，单通道格式

<br />
<br />

<a name="YqmMn"></a>
# 常见Opencv的数据类型
| cv类型 | 枚举数值 | 空间大小 | 范围 | 常规类型 |
| --- | --- | --- | --- | --- |
| CV_8U | 0 | 8bits | 0~255 | unsigned char或uint8_t |
| CV_8S | 1 | 8bits | -128~127 | char或int8_t |
| CV_16U | 2 | 16bits | 0~65535 | ushort, unsigned short int, unsigned short或uint16_t |
| CV_16S | 3 | 16bits | -32768~32767 | short, short int或int16_t |
| CV_32S | 4 | 32bits | -2147483648~2147483647 | int,long或int32_t/int64_t |
| CV_32F | 5 | 32bits | 1.18e-38~3.40e38 | float |
| CV_64F | 6 | 64bits | 2.23e-308~1.79e308 | double |

| **类型** | **C1** | **C2** | **C3** | **C4** |
| --- | --- | --- | --- | --- |
| CV_8U | 0 | 8 | 16 | 24 |
| CV_8S | 1 | 9 | 17 | 25 |
| CV_16U | 2 | 10 | 18 | 26 |
| CV_16S | 3 | 11 | 19 | 27 |
| CV_32S | 4 | 12 | 20 | 28 |
| CV_32F | 5 | 13 | 21 | 29 |
| CV_64F | 6 | 14 | 22 | 30 |

<br />
<br />

<a name="OKJCq"></a>
# 
<a name="pXtHH"></a>
# Mat 的创建

- 常使用的是利用构造函数 `Mat(nrows, ncols, type, [fillValue])`，最后一个参数中括号里代表可选参数，表示填入的初始值，如果不写默认为0。`nrows`、`ncols`一般为int型的整数，但ncols也可以为数组，这样表示建立一个多维Mat,此时传入的nrows表示维度。前两个参数还可以用`Size(ncol,nrow)`代替(注意Size里行列的顺序和函数参数的顺序是反的)，这就是另一种构造函数了。
- 也可以先声明一个Mat，然后利用其成员函数创建Mat。`create()`成员函数格式如下：`Mat(nrows,ncols,type)`，它只能接收3个参数，与构造函数的区别就是无法赋初值。使用示例如下：
```cpp
cv::Mat m1(4, 3, CV_8UC1);

cv::Mat m2(3, 5, CV_8UC1, 200);

cv::Mat m4;
m4.create(2, 3, CV_8UC1);
```
<br />
<br />

<a name="MNvef"></a>
# Mat 常用属性和函数

- `rows`：返回Mat的行数，仅对二维Mat有效，高维返回-1
- `cols`：返回Mat的列数，仅对二维Mat有效，高维返回-1
- `size`：返回Mat的尺寸大小(长x宽x高…)
- `dims`：返回Mat中数据的维度，OpenCV中维度永远大于等于2。如 3 * 4 的矩阵为 2 维， 3 * 4 * 5 的为3维

- `depth()`：用来度量每一个像素中每一个通道的精度，但它本身与图像的通道数无关！depth数值越大，精度越高。在Opencv中，Mat.depth()得到的是一个0~6的数字，分别代表不同的位数，对应关系如下：                enum{CV_8U=0,CV_8S=1,CV_16U=2,CV_16S=3,CV_32S=4,CV_32F=5,CV_64F=6}          
- `channels()`：通道数量，矩阵中表示一个元素所需要的值的个数。Mat矩阵元素拥有的通道数。例如常见的RGB彩色图像，channels==3；而灰度图像只有一个灰度分量信息，channels==1。
- `at()`：返回Mat某行某列的元素(可修改)
- `tr()`：返回指向Mat的行指针
- `clone()`：返回Mat的深拷贝
- `copyTo()`：返回Mat的深拷贝
- `eye()`：生成单位阵
- zeros()：生成元素全为0的矩阵
- `ones()`：生成元素全为1的矩阵
- `type()`：返回Mat的元素类型索引
- `inv()`：Mat求逆
- `mul()`：Mat矩阵乘法
- `data()`：返回指向矩阵数据单元的指针
- `total()`：返回矩阵中的元素总数
- `cv::hconcat()`：水平拼接两个矩阵
- `cv::vconcat()`：数值拼接两个矩阵
<br />
<br />

<a name="vZADM"></a>
# Mat的元素操作

- 获取Mat中的元素可以使用成员函数 `at()`。`.at<>()` 用法是尖括号`<>`传入模板参数，表示数据类型，小括号`()`中再传入行、列的索引，返回元素的引用。也可以只传入行或列，这样就是整行或者整列。`at()`**函数传入的模板参数数据类型必须与**`Mat`**的数据类型严格一致。**
```cpp
cv::Mat m1 = cv::Mat::eye(3, 3, CV_32F);
m1.at<float>(1,1) = 2.0;
```

- 也可以使用`ptr()`成员函数。它返回的是一个指针。
```cpp
Mat M = Mat::eye(10, 10, CV_64F);
    double sum = 0;
    for (int i = 0; i < M.rows; i++) {
        const double *Mi = M.ptr<double>(i);
        for (int j = 0; j < M.cols; j++)
            sum += std::max(Mi[j], 0.);
    }
```
<br />
<br />
<br />
<br />

<a name="WNnXL"></a>
# 
<a name="pXAzI"></a>
# 图像的显示
`imshow`函数在显示图像时，会将各种类型的数据都映射到[0, 255]。<br />比如：如果载入的图像是8U类型，就显示图像本来的样子。如果图像是16U 或32S（32S去掉符号位只有16位），便用像素值除以256。也就是说，值的范围是 [0, 255*256]映射到[0,255]。如果图像是32位或64位浮点型（32For 64F），像素值便要乘以255。也就是说，该值的范围是 [0,1]映射到[0,255]。 <br />如：CV_8U的灰度或BGR图像的颜色分量都在0~255之间。直接imshow可以显示图像。CV_32F或者CV_64F取值范围为0~1.0，imshow的时候会把图像乘以255后再显示。
<br />
<br />
<br />
<br />

<a name="m2iRw"></a>
# 图像的尺寸操作
```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
 
int main()
{
	Mat srcImage = imread("1.jpg");
	Mat temImage, dstImage1, dstImage2;
	temImage = srcImage;
 
	//尺寸调整
    //第一个参数：输入图像
    //第二个参数：输出图像
    //第三个参数输出图像的尺寸，如果是0，则有dsize=Size(round(fx*src.cols),round(fy*src,rows))计算得出
    //第四个参数：水平轴的缩放系数，默认为0
    //第五个参数：y轴撒谎能够的缩放系数，默认为0
    //第六个参数：插值方式，默认为INTER_LINEAR线性插值

	resize(temImage, dstImage1, Size(temImage.cols / 2, temImage.rows / 2), 0, 0, INTER_LINEAR);
	resize(temImage, dstImage2, Size(temImage.cols * 2, temImage.rows * 2), 0, 0, INTER_LINEAR);
 
	imshow("缩小", dstImage1);
	imshow("放大", dstImage2);
	waitKey();
	return 0;

```
<br />
<br />
<br />
<br />


<a name="hkPHf"></a>
# 图像标准化操作(减去均值，除以方差)
```cpp
cv::Mat Normalizer(cv::Mat src, cv::Mat dst){
    
    std::vector<float> mean = {0.485, 0.456, 0.406};
    std::vector<float> std = {0.229, 0.224, 0.225};
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(src, bgrChannels);
    for (auto i = 0; i < bgrChannels.size(); i++)
    {
        bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1.0 / std_value[i], (0.0 - mean_value[i]) / std_value[i]);
    }
    cv::meger(bgrChannels, dst);
}
```
<br />
<br />
<br />
<br />

<a name="uZpbg"></a>
# 数据类型之间的转换
```cpp
void convertTo( OutputArray m, int rtype, double alpha=1, double beta=0 ) const;
```

- 描述：把一个矩阵从一种数据类型转换到另一种数据类型，同时可以带上缩放因子和增量。
- 参数解释
   - `m`  目标矩阵。如果`m`在运算前没有合适的尺寸或类型，将被重新分配。
   - `rtype` 目标矩阵的类型。因为目标矩阵的通道数与源矩阵一样，所以`rtype`也可以看做是目标矩阵的位深度。如果`rtype`为负值，目标矩阵和源矩阵将使用同样的类型。
   - `alpha`  尺度变换因子（可选）。默认值是1。即把原矩阵中的每一个元素都乘以`alpha`。
   - `beta`    附加到尺度变换后的值上的偏移量（可选）。默认值是0。即把原矩阵中的每一个元素都乘以`alpha`，再加上`beta`。

