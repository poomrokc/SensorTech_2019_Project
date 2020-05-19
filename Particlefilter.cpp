#define _USE_MATH_DEFINES
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>
#include "Particlefilter.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;

const int NUMBER_OF_PARTICLES = 400;
const double INITIAL_WEIGHT = 1.0;
const double INITIAL_VELOCITY = 5;

/***************************************************************
 * Set the number of particles. Initialize all particles to first position
 * (based on estimates of x, y, theta and their uncertainties from GPS) and all weights to 1.
 * random gaussian noise is added to each particle
 ***************************************************************/
void ParticleFilter::init(double x, double y, double theta, double v, double std[]) {

  this->num_particles = NUMBER_OF_PARTICLES;
  random_device random_device;
  mt19937 gen(random_device());
  normal_distribution<> particle_x(x, std[0]);
  normal_distribution<> particle_y(y, std[1]);
  normal_distribution<> particle_theta(theta, std[2]);
  normal_distribution<> particle_v(v, std[3]);

  for (int i = 0; i < NUMBER_OF_PARTICLES; i++) {

    Particle p = {
      i,
      particle_x(gen)+std[0]+300,
      particle_y(gen)+std[1]+500,
      particle_theta(gen)+std[2],
	  INITIAL_VELOCITY,
      INITIAL_WEIGHT
    };

    this->weights.push_back(INITIAL_WEIGHT);
    this->particles.push_back(p);
  }

  this->is_initialized = true;
}


/***************************************************************
 *  Add measurements to each particle and add random Gaussian noise.
 ***************************************************************/
void ParticleFilter::prediction(double delta_t, double std[], int lock) {
  
  random_device random_device;
  mt19937 gen(random_device());
  normal_distribution<> noise_x(0.0, std[0]);
  normal_distribution<> noise_y(0.0, std[1]);
  normal_distribution<> noise_theta(0.0, std[2]);
  normal_distribution<> noise_v(0.0, std[3]);
    
  for (int i = 0;  i < NUMBER_OF_PARTICLES; i++) {

    double d = particles[i].v * delta_t;
	if(lock==0)
		this->particles[i].theta += noise_theta(gen);
    double theta = M_PI * this->particles[i].theta/180.0;
    
    this->particles[i].x += d * sin(theta);
    this->particles[i].y -= d * cos(theta);
	//this->particles[i].v += noise_v(gen);
  }
}


/***************************************************************
 *  Update the weights of each particle using a mult-variate Gaussian distribution.
 *  NOTE: The observations are given in the VEHICLE'S coordinate system. Particles are located
 *        according to the MAP'S coordinate system. So transformation is done.
 * For each particle:
 *   1. transform observations from vehicle to map coordinates assuming it's the particle observing
 *   2. find landmarks within the particle's range
 *   3. find which landmark is likely being observed based on nearest neighbor method
 *   4. determine the weights based difference particle's observation and actual observation
 ***************************************************************/
void ParticleFilter::updateWeights(std::vector<MapPoint> mappoint, int& lock) {

  std::vector<double> bp,bw;

  Mat image(800, 800, CV_8U);
  Mat mask(800, 800, CV_8U);
  for(int i=0;i<mappoint.size();i++) 
	  image.at<unsigned char>(mappoint[i].x,mappoint[i].y) = 1;

  for (int  i = 0; i < NUMBER_OF_PARTICLES; i++) {

	bp.push_back(this->particles[i].weight);
	bw.push_back(this->weights[i]);
    double px = this->particles[i].x;
    double py = this->particles[i].y;
    double ptheta = this->particles[i].theta;
	double pv = this->particles[i].v;

    double w = 1,gg=0,endw=0;
	int count100=0;

	circle( mask, Point( px,py ), 20, 2, -1);
	ellipse( mask,
	   Point( px,py ),
	   Size( 13,6 ),
	   ptheta,
	   181,
	   360,
	   1,
	   3,
	8 );
	
	for(int m=0;m<mappoint.size();m++)
	{
		if((mappoint[m].x-px)*(mappoint[m].x-px)+(mappoint[m].y-py)*(mappoint[m].y-py)<=2500) {
			gg+=sqrt((mappoint[m].x-px)*(mappoint[m].x-px)+(mappoint[m].y-py)*(mappoint[m].y-py));
			count100++;
		}
	}
	
	if(count100>10) {
		gg=50-gg/count100;
		w+=gg*gg*count100/2;
	}
	
	for(int k=px-20;k<px+20;k++)
		for(int kk=py-20;kk<py+20;kk++)
			if(k>=0&&kk>=0&&k<800&&kk<800) {
				if(mask.at<unsigned char>(k,kk) == 1 && image.at<unsigned char>(k,kk)==1) {
					endw+=1;
					mask.at<unsigned char>(k,kk) = 0;
				}
				if(mask.at<unsigned char>(k,kk) == 2 && image.at<unsigned char>(k,kk)==1) {
					endw-=0.5;
					mask.at<unsigned char>(k,kk) = 0;
				}
			}
	if(endw>0&&w>2)
		w+=endw*500;
	//printf("%lf\n",w);
    this->particles[i].weight = w;
    this->weights[i] = w;
  }
  if((*std::max_element((this->weights).begin(),(this->weights).end())) < 5.0 && lock<40)
  {
	  lock++;
	  for (int  i = 0; i < NUMBER_OF_PARTICLES; i++) {
		this->particles[i].weight = bp[i];
		this->weights[i] = bw[i];
	  }
  }
  else
	  lock=0;
}


/**************************************************************
 * Resample particles with replacement with probability proportional to their weight.
 ***************************************************************/
void ParticleFilter::resample(double x, double y, double theta, double v, double std[]){

  vector<Particle> resampled_particles;
  random_device random_device;
  mt19937 gen(random_device());
  discrete_distribution<int> index(this->weights.begin(), this->weights.end());

  for (int c = 0; c < NUMBER_OF_PARTICLES-200; c++) {

    int i = index(gen);

    Particle p {
      i,
      this->particles[i].x,
      this->particles[i].y,
      this->particles[i].theta,
	  this->particles[i].v,
      this->particles[i].weight
    };

    resampled_particles.push_back(p);
	this->weights[c] = this->particles[i].weight;
  }

  
  normal_distribution<> particle_x(x, std[0]);
  normal_distribution<> particle_y(y, std[1]);
  normal_distribution<> particle_theta(theta, std[2]);
  normal_distribution<> particle_v(v, std[3]);

  for (int i = 0; i < 200; i++) {

    Particle p = {
      i,
      particle_x(gen)+std[0]+300,
      particle_y(gen)+std[1]+500,
      particle_theta(gen)+std[2],
	  INITIAL_VELOCITY,
      INITIAL_WEIGHT
    };
	
    resampled_particles.push_back(p);
	this->weights[200+i] = INITIAL_WEIGHT;
  }

  this->particles = resampled_particles;
}