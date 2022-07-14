# Opencv SGBM立体匹配算法

# 前言
传统的立体匹配算法当中，BM（Block Matching），SGBM（Semi-Global Block matching），是两种常用的算法。并且这两种算法都在opencv中已经有了良好的实现。算法速度上BM > SGBM，匹配精度上：BM < SGBM。


# 代码实例
```cpp
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>

//for stereo matching
#include<opencv2/calib3d.hpp>

//for point cloud visualization
#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv){
    // load RGB images
    Mat imgL = imread("/home/codes/opencvTest/kittiTestImages/imgL/000000_10.png", 1);
    Mat imgR = imread("/home/codes/opencvTest/kittiTestImages/imgR/000000_10.png", 1);
    
    //keep left rgb image
    Mat rgb = Mat(imgL);
    
    // convert rgb images to grayscale
    cvtColor(imgL, imgL, CV_RGB2GRAY);
    cvtColor(imgR, imgR, CV_RGB2GRAY);
    
    int width = imgL.cols;
    int height = imgL.rows;
    
    
    // set parameters
    // 每个参数的意义和调整设计SGBM算法背后的原理，这里不做解释，可自行查阅
    int minDisparity = 0; 
    int numDisparities = 92;  //max disparity - min disparity
    int blockSize = 9; 
    int P1 = 8 * blockSize*blockSize;
    int P2 = 32 *blockSize*blockSize;
    int disp12MaxDiff = 1;
    int preFilterCap = 63;
    int uniquenessRatio = 10;
    int speckleWindowSize = 100;
    int speckleRange = 32;
    int mode = StereoSGBM::MODE_SGBM;
    
    //init stereoSGBM
    cv::Ptr<StereoSGBM> sgbm = StereoSGBM::create(minDisparity, numDisparities, blockSize,
                                                  P1, P2, disp12MaxDiff,
                                                  preFilterCap,  uniquenessRatio,
                                                  speckleWindowSize,  speckleRange, mode);
    
    //computer disparity
    Mat imgDisparity16S = Mat(height, width, CV_16S);
    chrono::steady_clock::time_point startTime = chrono::steady_clock::now();
    sgbm->compute(imgL, imgR, imgDisparity16S);
    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    double computerDuration = chrono::duration_cast<std::chrono::duration<double> >(endTime - startTime).count();
    cout << "Time of computer disparity: " << computerDuration << endl;
    
    
    //转换成32F格式获得真实视差值，这里记得缩放
    Mat imgDisparity32F = Mat(height, width, CV_32F);
    imgDisparity16S.convertTo(imgDisparity32F, CV_32F, 1.0 / 16); 
    
    //如果想看下获得的视差图效果
    /*
    Mat imgDispMap8U = Mat(height, width, CV_8U);
    double max, min;
    Point minLoc, maxLoc;
    minMaxLoc(imgDisparity32F, &min, &max, &maxLoc, &minLoc);
    double alpha = 255.0 / (max - min);
    imgDisparity32F.convertTo(imgDispMap8U, CV_8U, alpha, -alpha * min);
    cv::Mat colorisedDispMap;
    cv::applyColorMap(imgDispMap8U, colorisedDispMap, cv::COLORMAP_JET);
    imshow("color_32F-8U", colorisedDispMap);
    cv::waitKey(0);
    */
}
```
# 
# 视差图可视化
通过SGBM算法获得的视差图效果如下，由于SGBM算法的参数众多，所以调整不同的参数或许有改善的空间，具体操作可自行查阅。<br />![000000.png](/posts/Opencv-SGBM立体匹配算法/000000.png "原双目图像rgb左图")<br />


![colorised_disp.png](/posts/Opencv-SGBM立体匹配算法/colorised_disp.png "通过SGBM算法立体匹配获得的视差图。")
左边有一部分没有视差是正常的，那是右图中看不到的部分，这部分的宽窄与SGBM算法参数设置有关
<br />
<br />


![colorised_inserted_disp.png](/posts/Opencv-SGBM立体匹配算法/colorised_inserted_disp.png "进行空洞值插值处理")
从视差图得到直观效果来说貌似还可以。但是在SLAM中，拥有视差图是为后面映射为点云做准备的，所以这里将2D视差图映射成3D点云看看效果
<br />
<br />
<br />
<br />

<a name="UOoST"></a>
# 点云可视化
![point_cloud.png](/posts/Opencv-SGBM立体匹配算法/point_cloud.png "点云正面")

![point_cloud_rightside.png](/posts/Opencv-SGBM立体匹配算法/point_cloud_rightside.png "点云侧面")<br />正面看着效果勉强还行，但是实际上并不如此。缩放比例，同时旋转下方位，从侧面观察可以看出，其实估计点云模型存在着放射，发散等问题。原因是因为存在较多无效点，需要经过剔除处理。
<br />
<br />
<br />
<br />


# 附带代码
```cpp
float fx = 718.856;
    float fy = 718.856;
    float cx = 607.1928;
    float cy = 185.2157;
    float baseline =  0.3861;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	for (int v = 0; v < imgL.rows; v++)
	{
		for (int u = 0; u < imgL.cols; u++)
		{
			if (imgDisparity32F.at<float>(v, u) <= 10 || imgDisparity32F.at<float>(v, u) >= 96) continue;
			pcl::PointXYZRGB point;
			double x = (u - cx) / fx;
			double y = (v - cy) / fy;
			double depth = fx * baseline / (imgDisparity32F.at<float>(v, u));
			point.x = x * depth;
			point.y = y * depth;
			point.z = depth;
			point.b = rgb.at<cv::Vec3b>(v, u)[0];
			point.g = rgb.at<cv::Vec3b>(v, u)[1];
			point.r = rgb.at<cv::Vec3b>(v, u)[2];
			pointcloud->push_back(point);
		}
	}

    pcl::visualization::CloudViewer viewer("viewer");
    while(1){
        viewer.showCloud(pointcloud);
    }
```


```cpp
void insertDepth32f(cv::Mat& depth)
{
    const int width = depth.cols;
    const int height = depth.rows;
    float* data = (float*)depth.data;
    cv::Mat integralMap = cv::Mat::zeros(height, width, CV_64F);
    cv::Mat ptsMap = cv::Mat::zeros(height, width, CV_32S);
    double* integral = (double*)integralMap.data;
    int* ptsIntegral = (int*)ptsMap.data;
    memset(integral, 0, sizeof(double) * width * height);
    memset(ptsIntegral, 0, sizeof(int) * width * height);
    for (int i = 0; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 0; j < width; ++j)
        {
            int id2 = id1 + j;
            if (data[id2] > 1e-3) {
                integral[id2] = data[id2];
                ptsIntegral[id2] = 1;
            }
        }
    }
    // 积分区间
    for (int i = 0; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 1; j < width; ++j) {
            int id2 = id1 + j;
            integral[id2] += integral[id2 - 1];
            ptsIntegral[id2] += ptsIntegral[id2 - 1];
        }
    }
    for (int i = 1; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 0; j < width; ++j) {
            int id2 = id1 + j;
            integral[id2] += integral[id2 - width];
            ptsIntegral[id2] += ptsIntegral[id2 - width];
        }
    }
    int wnd;
    double dWnd = 2;
    while (dWnd > 1)
    {
        wnd = int(dWnd);
        dWnd /= 2;
        for (int i = 0; i < height; ++i)
        {
            int id1 = i * width;
            for (int j = 0; j < width; ++j)
            {
                int id2 = id1 + j;
                int left = j - wnd - 1;
                int right = j + wnd;
                int top = i - wnd - 1;
                int bot = i + wnd;
                left = max(0, left);
                right = min(right, width - 1);
                top = max(0, top);
                bot = min(bot, height - 1);
                int dx = right - left;
                int dy = (bot - top) * width;
                int idLeftTop = top * width + left;
                int idRightTop = idLeftTop + dx;
                int idLeftBot = idLeftTop + dy;
                int idRightBot = idLeftBot + dx;
                int ptsCnt = ptsIntegral[idRightBot] + ptsIntegral[idLeftTop] - (ptsIntegral[idLeftBot] + ptsIntegral[idRightTop]);
                double sumGray = integral[idRightBot] + integral[idLeftTop] - (integral[idLeftBot] + integral[idRightTop]);
                if (ptsCnt <= 0) {
                    continue;
                }
                data[id2] = float(sumGray / ptsCnt);
            }
        }
        int s = wnd / 2 * 2 + 1;
        if (s > 201) {
            s = 201;
        }
        cv::GaussianBlur(depth, depth, cv::Size(s, s), s, s);
    }
```
<br/>
<br/>

# 参考链接
[OpenCV3.4两种立体匹配算法效果对比 - 一度逍遥 - 博客园](https://www.cnblogs.com/riddick/p/8318997.html)
<br />
[KITTI下使用SGBM立体匹配算法获得深度图_逆水独流的博客-CSDN博客_kitti深度图](https://blog.csdn.net/ns2942826077/article/details/105023570)<br />
[SLAM学习之路---双目相机照片生成点云（附C++代码）_McQueen_LT的博客-CSDN博客_双目摄像头点云](https://blog.csdn.net/McQueen_LT/article/details/118876191?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-3-118876191-null-null.pc_agg_new_rank&utm_term=%E5%8F%8C%E7%9B%AE%E8%8E%B7%E5%8F%96%E7%82%B9%E4%BA%91&spm=1000.2123.3001.4430)
<br />
[双目视觉三维重建点云分层，断层、放射，散射，锥形](https://blog.csdn.net/RanchoLin/article/details/114702996)
