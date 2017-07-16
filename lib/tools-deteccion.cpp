//#include <sys/types.h>
//#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <stddef.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>
#include <stdarg.h>
#include <string.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include <iostream>
#include <cassert>


#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "helpers.hpp"
#include "tools-deteccion.hpp"

using namespace std;
using namespace cv;


void getROIDetection( cv::Mat *frame, std::vector<cv::Rect> det_rois ){
   cv::Rect roi_c;
   cv::Rect roi_b;
   
   size_t ic = 0;
   int ac = 0;

   size_t ib = 0;
   int ab = 0;

   cv::Mat crop;

   char hora[20];


   for( ic = 0; ic < det_rois.size(); ic++ ){
      roi_c.x = det_rois[ic].x;
      roi_c.y = det_rois[ic].y;
      roi_c.width = (det_rois[ic].width);
      roi_c.height = (det_rois[ic].height);

      ac = roi_c.width * roi_c.height;

      /*roi_b.x = det_rois[ib].x;
      roi_b.y = det_rois[ib].y;
      roi_b.width = (det_rois[ib].width);
      roi_b.height = (det_rois[ib].height);

      ab = roi_b.width * roi_b.height;

      if (ac > ab){
         ib = ic;
         roi_b.x = det_rois[ib].x;
         roi_b.y = det_rois[ib].y;
         roi_b.width = (det_rois[ib].width);
         roi_b.height = (det_rois[ib].height);
      }*/
      hora2( 0, hora );

      crop = (*frame)( roi_c );
      cv::imwrite( std::string("/home/diego/Imágenes/") + hora + std::string(".jpg"), crop );

      cv::Point pt1(det_rois[ic].x, det_rois[ic].y);
      cv::Point pt2((det_rois[ic].x + det_rois[ic].height), (det_rois[ic].y + det_rois[ic].width));
      cv::rectangle( *frame, pt1, pt2, cv::Scalar(0, 0, 250), 2, 8, 0 );      
   }
   
}

void Cholesky( const cv::Mat &A, cv::Mat &S ){
    CV_Assert(A.type() == CV_32F);

    int dim = A.rows;
    S.create(dim, dim, CV_32F);

    int i, j, k;

    for( i = 0; i < dim; i++ )
    {
        for( j = 0; j < i; j++ )
            S.at<float>(i,j) = 0.f;

        float sum = 0.f;
        for( k = 0; k < i; k++ )
        {
            float val = S.at<float>(k,i);
            sum += val*val;
        }

        S.at<float>(i,i) = std::sqrt(std::max(A.at<float>(i,i) - sum, 0.f));
        float ival = 1.f/S.at<float>(i, i);

        for( j = i + 1; j < dim; j++ )
        {
            sum = 0;
            for( k = 0; k < i; k++ )
                sum += S.at<float>(k, i) * S.at<float>(k, j);

            S.at<float>(i, j) = (A.at<float>(i, j) - sum)*ival;
        }
    }
}

void eval_kernel( const Mat &m1, const Mat &m2, Mat &K, double lambda, int kernel_type ){

   CV_Assert( m1.type() == CV_32F );
   CV_Assert( m2.type() == CV_32F );

   int m1_hight = m1.rows;
   int m1_width = m1.cols;
   int m2_hight = m2.rows;
   int m2_width = m2.cols;

   //double lambda = 0.55;

   Mat m1_2;
   Mat m2_2;
   K.create( m1_hight, m2_hight, CV_32F);
   multiply(m1,m1,m1_2,1);
   multiply(m2,m2,m2_2,1);

   reduce(m1_2,m1_2,1,CV_REDUCE_SUM);
   reduce(m2_2,m2_2,1,CV_REDUCE_SUM);

   for( int i=0; i < m1_hight; i++ ){
      for( int j=0; j < m2_hight; j++ ){
        K.at<float>(i,j) = -lambda*(m1_2.at<float>(i,0) + m2_2.at<float>(j,0) - 2*m1.row(i).dot(m2.row(j))) + 1e-6;
      }
   }

   exp(K,K);
}

void eval_efficient_kernel_descriptor( const Mat &set_img_features, 
                                       const Mat &Z,
                                       const Mat &G, 
                                       const int dim_patch, 
                                       const double lambda,
                                       const int kernel_type,
                                       Mat &kernel_descriptor ){

   Mat sum_descriptors = Mat::zeros( Z.rows, 1, CV_32F );
   Mat Kz;
  
   for( int k=0; k < 9; k++ ){
      eval_kernel( Z, set_img_features.row(k), Kz, lambda, kernel_type );
      add( sum_descriptors, Kz, sum_descriptors );
   }
   
   kernel_descriptor = 0.111111111111*G*sum_descriptors; //(1/9 = 0.11111)
   
}
