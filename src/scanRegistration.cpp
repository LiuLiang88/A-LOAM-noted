// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#include <cmath>
#include <vector>
#include <string>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <nav_msgs/Odometry.h>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::atan2;
using std::cos;
using std::sin;

const double scanPeriod = 0.1;   //10HZ,100毫秒

const int systemDelay = 0; 
int systemInitCount = 0;//用来记录一开始不做处理数据的帧数
bool systemInited = false;//系统是否初始化(false说明需要初始化,也就是第一帧数据来的时候要初始化)  初始化就是前几帧数据不做处理
int N_SCANS = 0;
float cloudCurvature[400000];
int cloudSortInd[400000];
int cloudNeighborPicked[400000];
int cloudLabel[400000];

bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubRemovePoints;
std::vector<ros::Publisher> pubEachScan;

bool PUB_EACH_LINE = false;

double MINIMUM_RANGE = 0.1; //点云中最小距离阈值

template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres)
{
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;
    // 把点云距离小于给定阈值的去除掉
    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j); //重设容器大小
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}
// 订阅lidar消息
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    // 如果系统没有初始化的话，就等几帧
    if (!systemInited)
    { 
        systemInitCount++;
        // 也就是只有第一帧数据不做处理
        if (systemInitCount >= systemDelay)
        {
            systemInited = true;
        }
        else
            return;
    }

    TicToc t_whole;//记录现在的时间
    TicToc t_prepare;
    std::vector<int> scanStartInd(N_SCANS, 0);//创建一个容器 初始化为64个0
    std::vector<int> scanEndInd(N_SCANS, 0);

    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;//创建一个点云对象
    // 把点云从ros格式转到pcl的格式 将结果保存在laserCloudIn
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn); //这个函数需要包含头文件pcl_conversions
    std::vector<int> indices;
    // 去除掉点云中的nan点
    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    // 去除距离小于阈值的点
    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);

    // 计算起始点和结束点的角度，由于激光雷达是顺时针旋转，这里取反就相当于转成了逆时针
    int cloudSize = laserCloudIn.points.size();
    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    // atan2范围是[-Pi,PI]，这里加上2PI是为了保证起始到结束相差2PI符合实际(也就是起始和终止点之间转了360度)
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                          laserCloudIn.points[cloudSize - 1].x) +
                   2 * M_PI;

    // 总有一些例外，比如这里大于3PI，和小于PI，就需要做一些调整到合理范围
    if (endOri - startOri > 3 * M_PI)
    {
        endOri -= 2 * M_PI;
    }
    else if (endOri - startOri < M_PI)
    {
        endOri += 2 * M_PI;
    }
    //printf("end Ori %f\n", endOri);

    bool halfPassed = false;
    int count = cloudSize;
    // PointType是啥意思,就是带强度的点云数据PointXYZI 
    PointType point;
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
    // 遍历每一个点 将每个点分好是那根线上的然后对应位置装到laserCloudScans中
    // 也就是第0号线上的点,都放到laserCloudScans容器的第一个位置上,
    // 第1号线上的点,都放到laserCloudScans容器的第二个位置上,
    for (int i = 0; i < cloudSize; i++)
    {
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;
        // 计算他的俯仰角  (弧度转角度)
        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        int scanID = 0;
        // 计算是第几根scan   16线雷达就是上下扫描15度,也就是正负15度,每跟线相隔2度
        if (N_SCANS == 16)
        {
            // 加0.5是为了四舍五入,如果求出12.9则int会取12,但是明显是第13跟线更合适
            scanID = int((angle + 15) / 2 + 0.5);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 32)
        {
            scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 64)
        {   
            if (angle >= -8.83)
                scanID = int((2 - angle) * 3.0 + 0.5);
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // use [0 50]  > 50 remove outlies 
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else
        {
            printf("wrong scan number\n");
            ROS_BREAK();
        }
        //printf("angle %f scanID %d \n", angle, scanID);
        // 计算水平角 是为了确定这一根线在整个圆周的那个地方(雷达是360度旋转的,所以需要判断这根线在哪)
        float ori = -atan2(point.y, point.x);
        if (!halfPassed)   //判断当前点过没过一半
        { 
            // 确保-PI / 2 < ori - startOri < 3 / 2 * PI
            if (ori < startOri - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }
            // 如果超过180度，就说明过了一半了
            if (ori - startOri > M_PI)
            {
                halfPassed = true;
            }
        }
        else
        {
            // 确保-PI * 3 / 2 < ori - endOri < PI / 2
            ori += 2 * M_PI;    // 先补偿2PI
            if (ori < endOri - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }
        // 角度的计算是为了计算相对的起始时刻的时间
        float relTime = (ori - startOri) / (endOri - startOri);
        // 整数部分是scan的索引，小数部分是相对起始时刻的时间
        point.intensity = scanID + scanPeriod * relTime;
        // 根据scan的idx送入各自数组
        laserCloudScans[scanID].push_back(point); 
    }
    // cloudSize是有效的点云的数目
    cloudSize = count;
    printf("points size %d \n", cloudSize);

    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());//定义一个指针
    // 将所有线数上的点全部集合到一个点云里面去，但是使用两个数组标记起始和结果，这里分别+5和-6是为了计算曲率方便
    for (int i = 0; i < N_SCANS; i++)
    { 
        scanStartInd[i] = laserCloud->size() + 5;
        *laserCloud += laserCloudScans[i];
        scanEndInd[i] = laserCloud->size() - 6;
    }

    printf("prepare time %f \n", t_prepare.toc());
    // 开始计算曲率 计算一帧点云中除了最左边和最右边五个点之外的全部点的曲率 就是用当前点左边5个点和右边五个点之和减去10倍的当前点
    for (int i = 5; i < cloudSize - 5; i++)
    { 
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
        // 存储曲率，索引
        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;  //存的是每个点的曲率至
        cloudSortInd[i] = i;  //当前点在点云中的id（第几个）,后边还会根据曲率大小排序,所以顺序肯定会乱,这是为了记录原先的id
        //下边两个都是标志位 这有什么用呢？
        // 就是选了一个点作为角点，就把他置为1，为了防止点的集中化，就把其附近的五个点都设置成1,这样提取的时候就不会选取到附近点了
        // 提取时只会选标志位为0的点
        // cloudNeighborPicked这个是用来表示点云中第i个点有没有取过（取过就是有没有选他做角点或面点，1代表取了，0代表没取）
        cloudNeighborPicked[i] = 0;  
        // cloudLabel这个是用来表示点云中的第i个点的状态，2代表曲率最大的角点，1代表一般大角点 -1代表面点 0代表一般面点
        cloudLabel[i] = 0;
    }


    TicToc t_pts;

    pcl::PointCloud<PointType> cornerPointsSharp;
    pcl::PointCloud<PointType> cornerPointsLessSharp;
    pcl::PointCloud<PointType> surfPointsFlat;
    pcl::PointCloud<PointType> surfPointsLessFlat;

    float t_q_sort = 0;
    // 遍历每个scan
    for (int i = 0; i < N_SCANS; i++)
    {
        // 没有有效的点了，就continue
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        // 用来存储除了角点之外的面点
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
        // 将每个scan等分成6等分
        for (int j = 0; j < 6; j++)
        {
            // 每个等分的起始和结束点
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

            TicToc t_tmp;  //一个计时的东西
            // 对点云按照曲率进行排序，小的在前，大的在后 
            // 这个接口返回的就是结束地址的下一个位置
            // cloudSortInd是数组，数组加整数表示数组中整数位置的元素地址，也就是从cloudSortInd的sp位置索引到ep位置，然后返回ep+1的地址
            std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);
            t_q_sort += t_tmp.toc();//返回当前时间

            int largestPickedNum = 0;
            // 挑选曲率比较大的部分
            for (int k = ep; k >= sp; k--)
            {
                // 排序后顺序就乱了，这个时候索引的作用就体现出来了
                // 我要挑选的点在原来点云中的id
                int ind = cloudSortInd[k]; 

                // 看看这个点是否是有效点，同时曲率是否大于阈值
                // 0.1就是阈值，大于0.1的就是角点，小于0.1的是面点
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] > 0.1)
                {

                    largestPickedNum++;
                    // 每段选2个曲率大的点
                    if (largestPickedNum <= 2)
                    {                        
                        // label为2是曲率大的标记
                        cloudLabel[ind] = 2;
                        // cornerPointsSharp存放大曲率的点
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    // 以及20个曲率稍微大一些的点
                    else if (largestPickedNum <= 20)
                    {                        
                        // label置1表示曲率稍微大
                        cloudLabel[ind] = 1; 
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    // 超过20个就算了
                    else
                    {
                        break;
                    }
                    // 这个点被选中后 pick标志位置1
                    cloudNeighborPicked[ind] = 1; 
                    // 为了保证特征点不过度集中，将选中的点周围5个点都置1,避免后续会选到
                    for (int l = 1; l <= 5; l++)
                    {
                        // 查看相邻点距离是否差异过大，如果差异过大说明点云在此不连续，是特征边缘，就会是新的特征，因此就不置位了
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    // 下面同理
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }
            // 下面开始挑选面点
            int smallestPickedNum = 0;
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k];
                // 确保这个点没有被pick且曲率小于阈值
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] < 0.1)
                {
                    // -1认为是平坦的点
                    cloudLabel[ind] = -1; 
                    surfPointsFlat.push_back(laserCloud->points[ind]);

                    smallestPickedNum++;
                    // 这里不区分平坦和比较平坦，因为剩下的点label默认是0,就是比较平坦
                    if (smallestPickedNum >= 4)
                    { 
                        break;
                    }
                    // 下面同理
                    cloudNeighborPicked[ind] = 1;
                    for (int l = 1; l <= 5; l++)
                    { 
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            for (int k = sp; k <= ep; k++)
            {
                // 这里可以看到，剩下来的点都是一般平坦，这个也符合实际
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
        }

        pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
        pcl::VoxelGrid<PointType> downSizeFilter;
        // 一般平坦的点比较多，所以这里做一个体素滤波
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilter.filter(surfPointsLessFlatScanDS);

        //surfPointsLessFlat里边装的是经体素滤波后的点云 
        surfPointsLessFlat += surfPointsLessFlatScanDS;
    }
    printf("sort q time %f \n", t_q_sort);
    printf("seperate points time %f \n", t_pts.toc());

    // 分别将当前点云、四种特征的点云发布出去
    sensor_msgs::PointCloud2 laserCloudOutMsg;
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    laserCloudOutMsg.header.frame_id = "/camera_init";
    pubLaserCloud.publish(laserCloudOutMsg);

    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);//发布角点，每一帧雷达数据分为6份，每份提取二个角点

    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);//发布曲率较大的角点 提取20个角点

    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat2.header.frame_id = "/camera_init";
    pubSurfPointsFlat.publish(surfPointsFlat2);//发布面点，每一帧雷达数据分为6份，每份提取四个面点

    sensor_msgs::PointCloud2 surfPointsLessFlat2;
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlat2.header.frame_id = "/camera_init";
    pubSurfPointsLessFlat.publish(surfPointsLessFlat2);//发布一般面点

    // pub each scam
    // 可以按照每个scan发出去，不过这里是false 也就是这一段不用看
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i< N_SCANS; i++)
        {
            sensor_msgs::PointCloud2 scanMsg;
            pcl::toROSMsg(laserCloudScans[i], scanMsg);
            scanMsg.header.stamp = laserCloudMsg->header.stamp;
            scanMsg.header.frame_id = "/camera_init";
            pubEachScan[i].publish(scanMsg);
        }
    }

    printf("scan registration time %f ms *************\n", t_whole.toc());
    // 因为雷达的频率是10hz，也就是100ms执行一次回调函数，如果执行一次超过100ms，则会丢帧
    if(t_whole.toc() > 100)
        ROS_WARN("scan registration process over 100ms");
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "scanRegistration");
    ros::NodeHandle nh;
    // 从参数服务器(配置文件中)获取多少线的激光雷达
    nh.param<int>("scan_line", N_SCANS, 16); //从launch文件中读取scan_line这个参数设置,如果没有读到就设置为16
    // 最小有效距离
    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);

    printf("scan line number %d \n", N_SCANS);
    // 只有线束是16 32 64的才可以继续
    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }

    // 接受话题名为"/velodyne_points"的消息,一旦接受到就执行laserCloudHandler回调函数
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);

    // 定义一些发布者
    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);

    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);

    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);

    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);

    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);

    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);

    if(PUB_EACH_LINE)
    {
        for(int i = 0; i < N_SCANS; i++)
        {
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            pubEachScan.push_back(tmp);
        }
    }
    ros::spin();

    return 0;
}
