#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <iomanip>
#include <random>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "Particlefilter.h"
#include <vector>
#include <rplidar.h>

using namespace rp::standalone::rplidar;
using namespace cv;
using namespace std;
char a[6];
double b,c;

int omp[1005][1005];
int x[205];
int y[205];
int temp[205];
int head[205];


int gethead(int x) 
{
	if(head[x]==x)
		return x;
	return head[x]=gethead(head[x]);
}

int tc;
int main( int argc, char** argv )
{
	time_t start_time,current_time;
	// port
	const char* port = "com3"; // "\\\\.\\com99";
	int baudrate = 115200;
	int lock = 0;

	// driver
	RPlidarDriver* driver = RPlidarDriver::CreateDriver(DRIVER_TYPE_SERIALPORT);
	if (!driver) {
		cout << "Failed to create driver" << endl;
		return 0;
	}

	// connect
	if (IS_FAIL(driver->connect(port, baudrate))) {
		cout << "Failed to connect" << endl;
		RPlidarDriver::DisposeDriver(driver);
		return 0;
	}

	// device info
	rplidar_response_device_info_t device_info;
	if (IS_FAIL(driver->getDeviceInfo(device_info)))
	{
		cout << "Failed to get device info" << endl;
		RPlidarDriver::DisposeDriver(driver);
		return 0;
	}

	// check health
	rplidar_response_device_health_t health_info;
	u_result result = driver->getHealth(health_info);
	if (IS_FAIL(result)) {
		cout << "Failed to get health info" << endl;
		RPlidarDriver::DisposeDriver(driver);
		return 0;
	}
	cout << "RPLidar health status : " << health_info.status << endl;
	if (health_info.status == RPLIDAR_STATUS_ERROR) {
		cout << "Rplidar internal error detected" << endl;
		RPlidarDriver::DisposeDriver(driver);
		return 0;
	}
	
	// start
	driver->startMotor();
	driver->startScan(false, true);
	
	
	double sigma_pos [4] = {100, 100, 180, 2};
	double sigma_on [4] = {1, 1, 3,3};
	default_random_engine gen;
	normal_distribution<double> N_x_init(0, sigma_pos[0]);
	normal_distribution<double> N_y_init(0, sigma_pos[1]);
	normal_distribution<double> N_theta_init(0, sigma_pos[2]);
	normal_distribution<double> N_v_init(0, sigma_pos[3]);
	double n_x, n_y, n_theta, n_v;
	ParticleFilter pf;
	
	Mat image(800, 800, CV_8UC3, Scalar(0, 0, 0));
	Mat image2(800, 800, CV_8UC3, Scalar(0, 0, 0));
	Mat image3(800, 800, CV_8UC3, Scalar(0, 0, 0));
	Scalar black( 255, 255, 255 );
	Scalar red( 0, 255, 0 );
	namedWindow( "Display window", WINDOW_AUTOSIZE );
	namedWindow( "Display window2", WINDOW_AUTOSIZE );
	namedWindow( "Display window3", WINDOW_AUTOSIZE );
	int allall=0;
	time(&start_time);
	while(true) {
		rplidar_response_measurement_node_hq_t nodes[8192];
		size_t count = _countof(nodes);
		allall+=count;
		if (IS_OK(driver->grabScanDataHq(nodes, count))) {
			driver->ascendScanData(nodes, count);
			for (int i=0; i<(int)count; i++) {
				double b = nodes[i].angle_z_q14 * 90.f / (1 << 14);
				double c = nodes[i].dist_mm_q2 / (1 << 2);
				//printf("%lf %lf %u\n",b,c,nodes[i].flag);
				double ypos = c * sin((M_PI * b)/180.0)/20 +400;
				double xpos = c * cos((M_PI * b)/180.0)/20 +400;
				for(int i=-3;i<=3;i++)
					for(int j=-3;j<=3;j++)
						if((int)xpos+i>=0&&(int)ypos+j>=0&&omp[(int)xpos+i][(int)ypos+j]<255)
							omp[(int)xpos+i][(int)ypos+j]+=1;
			}
		}
		time(&current_time);
		if (current_time-start_time>=10) {
			break;
		}
	}
	
	printf("%d\n",allall);

	int countt = 0;
	int pc=0;
	int thres = 2;
	
	
	//particle filter start
	n_x = N_x_init(gen);
	n_y = N_y_init(gen);
	n_theta = N_theta_init(gen);
	n_v = N_v_init(gen);
	pf.init(n_x, n_y, n_theta, n_v, sigma_pos);
	
	
	
	while(true) {
		rplidar_response_measurement_node_hq_t nodes[8192];
		size_t count = _countof(nodes);
		if (IS_OK(driver->grabScanDataHq(nodes, count))) {
			driver->ascendScanData(nodes, count);
			for (int i=0; i<(int)count; i++) {
				int a = nodes[i].flag;
				double b = nodes[i].angle_z_q14 * 90.f / (1 << 14);
				double c = nodes[i].dist_mm_q2 / (1 << 2);
				if(a==1) {
					countt++;
				}
				if(countt==thres) 
				{
					vector<MapPoint> mp;
					for(int i=1;i<=pc;i++)
					{
						circle(image,Point(x[i],y[i]),2,Scalar(0,0,255),-1);
						circle(image2,Point(x[i],y[i]),2,Scalar(0,0,255),-1);
						circle(image3,Point(x[i],y[i]),2,Scalar(255,255,255),-1);
						MapPoint mm {
						  x[i],
						  y[i],
						};
						mp.push_back(mm);
					}
						
						
					//particle
					pf.prediction(1,sigma_on,lock);
					pf.updateWeights(mp,lock);
					pf.resample(n_x, n_y, n_theta, n_v,sigma_pos);
					pf.updateWeights(mp,lock);
					for(int i=0;i<200;i++) 
						head[i]=i;
					
					for(int i=0;i<200;i++) 
					{
						for(int i2=0;i2<200;i2++)
						{
							if((pf.particles[i].x-pf.particles[i2].x)*(pf.particles[i].x-pf.particles[i2].x) + (pf.particles[i].y-pf.particles[i2].y)*(pf.particles[i].y-pf.particles[i2].y) <=400)
								if(gethead(i)!=gethead(i2)&&pf.particles[i].weight>30&&pf.particles[i2].weight>30)
									head[gethead(i)]=gethead(i2);
						}					
						//circle(image,Point(pf.particles[i].x,pf.particles[i].y),20,Scalar(255,0,0),-1);
						ellipse( image2,
						   Point( pf.particles[i].x,pf.particles[i].y ),
						   Size( 13,6 ),
						   pf.particles[i].theta,
						   181,
						   360,
						   Scalar( 0, 255, 255 ),
						   3,
						8 );
						
						
							/*ellipse( image,
							   Point( pf.particles[i].x,pf.particles[i].y ),
							   Size( 13,6 ),
							   pf.particles[i].theta,
							   181,
							   360,
							   Scalar( 255, 0, 0 ),
							   3,
							8 );*/
						/*else
							ellipse( image,
							   Point( pf.particles[i].x,pf.particles[i].y ),
							   Size( 13,6 ),
							   pf.particles[i].theta,
							   181,
							   360,
							   Scalar( 255, 255, 0 ),
							   3,
							8 );*/
						
					}
					/*for(i=201;i<400;i++)
						ellipse( image2,
						   Point( pf.particles[i].x,pf.particles[i].y ),
						   Size( 13,6 ),
						   pf.particles[i].theta,
						   181,
						   360,
						   Scalar( 255, 255, 0 ),
						   3,
						8 );
						*/
					for(int i=0;i<200;i++) {
						int count=0,maxer=0;
						int index;
						double cw=0;
						double xx=0;
						double yy=0;
						double tt=0;
						for(int i2=0;i2<200;i2++) {
							if(gethead(i2)==i)
							{
								count++;
								xx+=pf.particles[i2].x*pf.particles[i2].weight;
								yy+=pf.particles[i2].y*pf.particles[i2].weight;
								tt+=pf.particles[i2].theta*pf.particles[i2].weight;
								cw+=pf.particles[i2].weight;
							}
						}
						if(count>30) {
							ellipse( image,
							   Point( xx/cw,yy/cw ),
							   Size( 13,6 ),
							   tt/cw,
							   181,
							   360,
							   Scalar( 0, 255, 0 ),
							   3,
							8 );
							if(lock>0) {
								for(int i2=0;i2<200;i2++) {
									if(gethead(i2)==i)
										pf.particles[i2].theta=tt/cw;
								}
							}
						}
					}	
					imshow( "Display window", image );  
					imshow( "Display window2", image2 ); 
					imshow( "Display window3", image3 ); 
					/*while(1)
					{
						if(waitKey(33)==0)
							break;
					}*/
					waitKey(1);			
					countt=0;
					pc=0;
					image = Mat(800, 800, CV_8UC3, Scalar(0, 0, 0));
					image2 = Mat(800, 800, CV_8UC3, Scalar(0, 0, 0));
					image3 = Mat(800, 800, CV_8UC3, Scalar(0, 0, 0));
				}
				else
				{
					double ypos = c * sin((M_PI * b)/180.0)/20;
					double xpos = c * cos((M_PI * b)/180.0)/20;
					circle(image3,Point(int(xpos)+400,int(ypos)+400),2,Scalar(255,255,255),-1);
					if(omp[int(xpos)+400][int(ypos)+400]<50) {
						pc++;
						x[pc] = int(xpos)+400;
						y[pc] = int(ypos)+400;
					}
					else {
						circle(image,Point(int(xpos)+400,int(ypos)+400),2,Scalar(255,255,255),-1);
						circle(image2,Point(int(xpos)+400,int(ypos)+400),2,Scalar(255,255,255),-1);
					}
				}
			}
		}
	}
    //namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    //imshow( "Display window", image );                   // Show our image inside it.

    //waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}