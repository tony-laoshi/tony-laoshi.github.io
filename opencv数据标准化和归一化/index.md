# Opencv数据标准化和归一化



# 介绍
归一化和标准化都是对数据做变换，将原始的一组数据转换到某个范围或者说到某种状态。在数据处理的时候我们常会碰到如下三种处理：

- 归一化（Normalization）：将一组数据变化到某个固定区间内，比较常见的如[0, 1]。广义来说可以是各种区间，比如在对图像进行处理是会经常映射到[0, 255]，也有映射到[-1, 1]的情况
- 标准化（Standardization）：将一组数据变换为均值为0，标准差为1的分布（该分布并非一定符合正态分布）
- 中心化：中心化也叫零均值处理，就是用一组原始数据减去这些数据的均值。比如在ICP算法中会先对数据进行中心化

# 联系和区别
本质上，归一化和标准化都是对数据做线性变换。
但是也存在一些区别。比如第一，归一化会严格的限定变换后数据的区间。而标准化没有严格的区间限定，只是其数据的均值为0，标准差为1。第二，归一化对数据的缩放比例仅仅和极值有关，比如100个数，你除去极大值和极小值其他数据都更换掉，缩放比例![](https://cdn.nlark.com/yuque/__latex/8502840d3c0ce63573d0058539d2ef7b.svg#card=math&code=%5Calpha%20%3D%20X_%7Bmax%7D-X_%7Bmin%7D&id=fzGvv)是不变的；但是标准化中，如果除去极大值和极小值其他数据都更换掉，那么均值和标准差大概率会改变，这时候，缩放比例自然也改变了。


# 标准化和归一化的多种形式
广义的说，标准化和归一化同为对数据的线性变化，所以我们没必要规定死，归一化就是必须到[ 0 , 1]之间，我到[ 0 , 1 ]之间之后再乘一个255也没有问题对吧。常见的有以下几种：

- 归一化的最通用模式Normalization，也称线性归一化：

![](https://cdn.nlark.com/yuque/__latex/90005fcf88fb84f86bc440af3cf78126.svg#card=math&code=X_%7Bn%20e%20w%7D%3D%5Cfrac%7BX_%7Bi%7D-X_%7B%5Cmin%20%7D%7D%7BX_%7B%5Cmax%20%7D-X_%7B%5Cmin%20%7D%7D%2C%20%5Ctext%20%7B%20%E8%8C%83%E5%9B%B4%20%7D%5B0%2C1%5D&id=yYVNy)
[
](https://blog.csdn.net/weixin_36604953/article/details/102652160)

- 均值归一化 （Mean normalization）：

![](https://cdn.nlark.com/yuque/__latex/bdb6bb5dc00be3423dc275d5325c3ca4.svg#card=math&code=X_%7Bn%20e%20w%7D%3D%5Cfrac%7BX_%7Bi%7D-%5Coperatorname%7Bmean%7D%28X%29%7D%7BX_%7B%5Ctext%20%7Bmax%20%7D%7D-X_%7B%5Cmin%20%7D%7D%20%5Ctext%20%7B%2C%20%E8%8C%83%E5%9B%B4%20%7D%5B-1%2C1%5D&id=DP3Mt)

- 标准化(Standardization)，也叫标准差标准化：

![](https://cdn.nlark.com/yuque/__latex/8ebea37c777a4dc7f0ebd687f971f223.svg#card=math&code=X_%7Bn%20e%20w%7D%3D%5Cfrac%7BX_%7Bi%7D-%5Cmu%7D%7B%5Csigma%7D%20%5Ctext%20%7B%2C%20%E8%8C%83%E5%9B%B4%E5%AE%9E%E6%95%B0%E9%9B%86%20%7D&id=AADQo)


# 标准化、归一化的原因、用途
为何统计模型、机器学习和深度学习任务中经常涉及到数据(特征)的标准化和归一化呢，一半原因以下几点，当然可能还有一些其他的作用，大家见解不同，我说的这些是通常情况下的原因和用途。

1. 统计建模中，如回归模型，自变量X 的量纲不一致导致了回归系数无法直接解读或者错误解读；需要将X都处理到统一量纲下，这样才可比；
1. 机器学习任务和统计学任务中有很多地方要用到“距离”的计算，比如PCA，比如KNN，比如kmeans等等，假使算欧式距离，不同维度量纲不同可能会导致距离的计算依赖于量纲较大的那些特征而得到不合理的结果；
1. 参数估计时使用梯度下降，在使用梯度下降的方法求解最优化问题时， 归一化/标准化后可以加快梯度下降的求解速度，即提升模型的收敛速度。


# 什么时候Standardization，什么时候Normalization

- 如果对输出结果范围有要求，用归一化
- 如果数据较为稳定，不存在极端的最大最小值，用归一化
- 如果数据存在异常值和较多噪音，用标准化，可以间接通过中心化避免异常值和极端值的影响


# cv::normalize() 函数介绍
```cpp
void normalize(InputArray src, OutputArray dst, double alpha=1, double beta=0, int norm_type=NORM_L2, int dtype=-1, InputArray mask=noArray());
```

参数

- `src`  - 输入数组
- `dst` - 输出数组，支持原地运算
- `alpha` -  range normalization模式的最小值
- `beta `- range normalization模式的最大值，不用于norm normalization(范数归一化)模式。
- `norm_type` - 归一化的类型，可以有以下的取值
   - `NORM_MINMAX`:数组的数值被平移或缩放到一个指定的范围，线性归一化，一般较常用。比如归一化到[min, max]范围内，则计算公式如下：
      - ![](https://cdn.nlark.com/yuque/__latex/fa52501609bf524a2981bd907ddbaa2f.svg#card=math&code=d%20s%20t%28i%2C%20j%29%3D%5Cfrac%7B%5B%5Coperatorname%7Bsrc%7D%28i%2C%20j%29-%5Cmin%20%28%5Coperatorname%7Bsrc%7D%28x%2C%20y%29%29%5D%20%2A%28%5Cmax%20-%5Cmin%20%29%7D%7B%5Cmax%20%28%5Coperatorname%7Bsrc%7D%28x%2C%20y%29%29-%5Cmin%20%28%5Coperatorname%7Bsrc%7D%28x%2C%20y%29%29%7D%2B%5Cmin&id=OATmo)
   -  `NORM_INF`: 归一化数组的无穷范数(绝对值的最大值)。每个值除以最大值来进行无穷范数归一化。同上最终归一化的值为单位向量的每个值乘以参数要归一化的范数值alpha
   - ` NORM_L1` :  归一化数组的L1-范数(绝对值的和)。数组元素绝对值求和，然后算出每一个元素比上总和的比值，加起来总为1。这里要归一化的范数值为1.0，所求出的比值即为最后归一化后的值，若归一化范数值alpha为2.0，则每个比值分别乘以2.0即得到最后归一化后的结果为0.2, 0.8, 1.0，以此类推
   - ` NORM_L2`: 归一化数组的L2-范数(各元素的平方和然后求平方根 ，欧氏距离)。即将该向量归一化为单位向量，每个元素值除以该向量的模长。同上最终归一化的值为单位向量的每个值乘以参数要归一化的范数值alpha
- dtype - 为负值时, 输出数据类型和输入数据类型一致，否则和src通道一致，depth =CV_MAT_DEPTH(dtype)
- mask -  掩码。选择感兴趣区域，选定后只能对该区域进行操作。


# 参考链接




