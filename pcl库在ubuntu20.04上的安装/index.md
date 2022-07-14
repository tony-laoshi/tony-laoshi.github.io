# Ubuntu20.04安装PCL库

 
# 前言
pcl库的编译安装真心让人想吐，运气好一次通过，运气不好（各种环境的、各种依赖的问题）两三天就过去了。在这里分享下我安装pcl所遇到的问题。


# 通过sudo apt 安装
通过这种方式安装的最大的好处就是简单且不容易出现编译安装的问题，缺点就是可能会有部分功能无法使用
## 安装依赖
```bash
sudo apt-get update
sudo apt-get install git build-essential linux-libc-dev
sudo apt-get install cmake cmake-gui
sudo apt-get install libusb-1.0-0-dev libusb-dev libudev-dev
sudo apt-get install mpi-default-dev openmpi-bin openmpi-common
sudo apt-get install libflann1.9 libflann-dev
sudo apt-get install libeigen3-dev
sudo apt-get install libboost-all-dev
sudo apt-get install libqhull* libgtest-dev  
sudo apt-get install freeglut3-dev pkg-config  
sudo apt-get install libxmu-dev libxi-dev   
sudo apt-get install mono-complete   
sudo apt-get install libopenni-dev   
sudo apt-get install libopenni2-dev 
sudo apt-get install libvtk7-dev libvtk6-dev
sudo apt-get install qt5-default
```
## 安装PCL
```bash
sudo apt-get install libpcl-dev
```
至此，PCL库的安装和配置就算是完成了，接下来测试一下PCL库是否可以正常运行

## 测试
### 编写源文件
写个测试用的源文件 test.cpp （网上copy的）
```cpp
#include <iostream>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>


int main(int argc, char **argv) {
    std::cout << "Test PCL !!!" << std::endl;
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    uint8_t r(255), g(15), b(15);
    for (float z(-1.0); z <= 1.0; z += 0.05)
    {
        for (float angle(0.0); angle <= 360.0; angle += 5.0)
        {
            pcl::PointXYZRGB point;
            point.x = 0.5 * cosf (pcl::deg2rad(angle));
            point.y = sinf (pcl::deg2rad(angle));
            point.z = z;
            uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
                            static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
            point.rgb = *reinterpret_cast<float*>(&rgb);
            point_cloud_ptr->points.push_back (point);
        }
        if (z < 0.0)
        {
            r -= 12;
            g += 12;
        }
        else
        {
            g -= 12;
            b += 12;
        }
    }
    point_cloud_ptr->width = (int) point_cloud_ptr->points.size ();
    point_cloud_ptr->height = 1;
    
    pcl::visualization::CloudViewer viewer ("test");
    viewer.showCloud(point_cloud_ptr);
    while (!viewer.wasStopped()){ };
    return 0;
}

```

### 编写CMakeLists.txt
在目录下创建TestPCL文件夹，用于存储测试项目的文件，将test.cpp和CMakeLists.txt存储至TestPCL文件夹，创建TestPCL/bulid文件夹以储存中间文件。
```cmake
cmake_minimum_required(VERSION 2.6)
project(TEST)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(TEST test.cpp)

target_link_libraries (TEST ${PCL_LIBRARIES})

install(TARGETS TEST RUNTIME DESTINATION bin)
```
### 编译安装运行
```bash
cd build
cmake ..
make
./TEST
```
### 正常运行结果
![result](/posts/Ubuntu20.04安装PCL库/test_result.png)


# 通过源码安装
## 安装PCL依赖
```bash
sudo apt-get update
sudo apt-get install git build-essential linux-libc-dev
sudo apt-get install cmake cmake-gui
sudo apt-get install libusb-1.0-0-dev libusb-dev libudev-dev
sudo apt-get install mpi-default-dev openmpi-bin openmpi-common
sudo apt-get install libflann1.9 libflann-dev # 有说ubuntu16对应1.8，ubuntu18对应1.9，我直接用了1.9
sudo apt-get install libeigen3-dev
sudo apt-get install libboost-all-dev
sudo apt-get install libqhull* libgtest-dev
sudo apt-get install freeglut3-dev pkg-config
sudo apt-get install libxmu-dev libxi-dev
sudo apt-get install mono-complete
sudo apt-get install libopenni-dev
sudo apt-get install libopenni2-dev
```
## 安装VTK
### 安装vtk依赖
```bash
#首先安装VTK的依赖：X11，OpenGL；cmake和cmake-gui在安装pcl依赖的时候安装过了的话可以跳过
sudo apt-get install libx11-dev libxext-dev libxtst-dev libxrender-dev libxmu-dev libxmuu-dev

#OpenGL
sudo apt-get install build-essential libgl1-mesa-dev libglu1-mesa-dev

#cmake && cmake-gui
sudo apt-get install cmake cmake-gui
```

### 下载vtk

**官网下载链接：**[Download | VTK](https://vtk.org/download/)
这里要注意以下VTK 和 PCL 版本之间的兼容性。我不确切肯定有版本兼容的问题，只是在我尝试了不同VTK版本后，发现VTK版本和PCL的版本还是有一定要求的。最好安装两者的版本如下（自己尝试过）:

| ubuntu版本呢 | pcl 版本 | vtk版本 |
| --- | --- | --- |
| 18.04 / 20.04 | 1.9.1 | 8.2.0 |
| 18.04 / 20.04 | 1.8.1 | 7.1.1 |
| 16.04 | 1.7.2 | 5.10.1 /6.2.0 |

[
](https://github.com/PointCloudLibrary/pcl/releases)

### 编译安装
下载完成之后解压到准备好的安装目录，再通过终端打开 cmake GUI 模式
```bash
cmake-gui
```
在cmake-gui中：

1. 配置` where is the source code` 为VTK-8.2.0所在目录。（然后在VTK-8.2.0所在目录下新建一个build文件夹）
1. 配置 `where to build the binaries `为VTK-8.2.0下的build文件夹
1. 点击`Configure`，（用“Unix Makefiles”就可以）。配置完成后，显示“Configuring done”
1. 勾选`VTK-Group-Qt`，`VTK_QT_VERSION`选为5，再点击`Configure`，配置完成后，显示“Configuring done”
1. 点击`Generate`，显示“Generating done”，在build文件夹下生成工程文件
1. 推出cmake-gui

在终端里切换到VTK-8.2.0安装目录下的build文件夹
```bash
make -j4    #性能好内存大的电脑就用 -j8 吧
sudo make install
```


## 安装PCL
**官网下载连接：**[Releases · PointCloudLibrary/pcl](https://github.com/PointCloudLibrary/pcl/releases)
到PCL的github主页下载需要的版本(这里我下载的1.9.1)，放到准备好的安装目录下。打开终端，进到pcl的安装目录下：
```bash
mkdir build    
cd build    
# 设置CMAKE_INSTALL_PREFIX是为了把pcl安装到指定文件夹内，所以这个路径根据自己的情况设置
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/pcl-1.9.1 -DCMAKE_TYPE=None .. 
make -j4   #一样的，根据实际情况调整
sudo make install
```

## 测试
同上




## 遇到的问题
### 问题1
在make pcl的时候，报错：`“/usr/bin/ld: cannot find -lvtkIOMPIImage /usr/bin/ld: cannot find -lvtkIOMPIParallel /usr/bin/ld: cannot find -lvtkFiltersParallelDIY2”`
#### 解决方案
重新编译安装 vtk。在cmake-gui模式下，完成第3步的`Configure`，然后勾选`Advanced`，在`search`中把`/usr/bin/ld`找不到的`vtkIOMPIImage`，`vtkIOMPIParallel`，`vtkFiltersParallelDIY2`都选上。包括如果出现了类似的问题也可以县看看这里面是否可以选上的。

再点击`Configure`，显示“Configuring done”，点击`Generate`，显示“Generating done”。
打开终端，完成 `make`  和  `sudo make install`


### 问题2
cmake pcl 的时候，提示`Checking for module ‘metslib’ – No package ‘metslib’ found`

#### 解决方案
安装metslib（我用的是0.5.3版本）
下载链接：[https://www.coin-or.org/download/source/metslib/metslib-0.5.3.tgz](https://www.coin-or.org/download/source/metslib/metslib-0.5.3.tgz)
解压后，在metslib-0.53打开命令终端并执行
```bash
sudo sh ./configure
sudo make
sudo make install
```
顺利完成安装metslib
[
](https://blog.csdn.net/yunluoxiaobin/article/details/103078386)

### 问题3
`No rule to make target in /usr/lib/x86_64-linux-gnu/libpcl_surface. so **后面的内容不记得了**`

#### 解决方案
```bash
lcoate libpcl_surface. so
```
`/usr/lib/x86_64-linux-gnu/libpcl_surface.so.1.10`   结果没有上面说的libpcl_surface. so，这是因为我之前通过sudo apt install libpcl-dev 安装过pcl，所以动态库还保留在那里。 简而言之就是，要在/usr/lib/x86_64-linux-gnu/  找libpcl_surface. so这个动态库。  所以先找到 libpcl_surface. so在哪里
```bash
whereis libpcl_surface. so
```
`libpcl_surface:  /usr/lib/libpcl_surface.so` 发现在`/usr/lib/`这个路径。所以只需要把这个路径下的动态库建立一个软链接到`/usr/lib/x86_64-linux-gnu/`就行。大概了，还有很多动态库都是这种情况，所以可以使用候补符号一次性解决。
```bash
sudo ln -s /usr/lib/libpcl_*.so /usr/lib/x86_64-linux-gnu
```


# 参考链接
[二、PLC安装踩坑总结（Ubuntu 16.4+PCL1.8.1+VTK7.1+Qt5.9.9)_way7486chundan的博客-CSDN博客](https://blog.csdn.net/way7486chundan/article/details/110296785?utm_term=%E6%80%8E%E4%B9%88%E5%8D%B8%E8%BD%BDVTK&utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~sobaiduweb~default-2-110296785-null-null&spm=3001.4430)

