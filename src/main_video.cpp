#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <iostream>
#include <cassert>
#include <math.h>

/*OPENCV LIB*/
#include <opencv/cv.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "tools-deteccion.hpp"
#include "helpers.hpp"
#include "stringtools.hpp"

using namespace cv;
using namespace std;

int main( int argc, const char * argv[] ){
   
   char path_src[200];
   char total_patch_src[200];

   /*Variables HOG*/
   int size_patch = 16;
   int size_block = 8;
   int size_step = 8;
   int bins = 9;
   int dim_patch = 2*(size_patch/size_block)*bins;

   /*Variables K-Means*/
   int clusterCount = 335;
   cv::Mat labels;
   int attempts = 5;
   cv::Mat centers;

   /*Variables modelo*/
   int resize_img = 32;
   int size_set_feature_base;
   cv::Mat Z;
   cv::Mat K;
   cv::Mat iK;
   cv::Mat G;
  
 
   if ( !argv[1] ){
      printf("Faltan parametros. <path_images>\n");
      exit(-1);
   }
 
   strcpy( path_src, argv[1] );


   /*Extract HOGs descriptors*/

   size_set_feature_base = getSizePaths(path_src);

   cv::Mat Ps( 9*size_set_feature_base, dim_patch, CV_32F );
   cv::vector<float> descriptorsValues;
   cv::vector<cv::Point> locations;
   cv::Mat im;

   char **paths_eye_detection = (char**) malloc( size_set_feature_base*sizeof(char*) );

   getPathImages( path_src, paths_eye_detection, 100 );

   int j = 0;
   for( int i=0; *(paths_eye_detection+i); i++ ){
      strcpy( total_patch_src, path_src );
      strcat( total_patch_src, "/" );
      strcat( total_patch_src, *(paths_eye_detection+i) );
      im = cv::imread( total_patch_src, CV_LOAD_IMAGE_GRAYSCALE );
      //imshow("Detection",im);
      resize( im, im, cv::Size(resize_img,resize_img) );
      
      cv::HOGDescriptor d( cv::Size(im.rows,im.cols), cv::Size(size_patch,size_patch), cv::Size(size_block,size_block), cv::Size(size_step,size_step), bins );
      d.compute( im, descriptorsValues, cv::Size(0,0), cv::Size(0,0), locations ); 

      char* buf1;
      char* buf2;
      
      for( int k = 0; k < 9; k++ ){
         buf1 = (char*) Ps.row(j).data;
         buf2 = (char*) &descriptorsValues[0] + k*dim_patch;
         memcpy( buf1, buf2, sizeof(cv::vector<float>)*dim_patch );
         j++;
      }

      im.release();
      if(!*(paths_eye_detection+i)) free(*(paths_eye_detection+i));
   }

   if(!paths_eye_detection) free(paths_eye_detection);

   /*for( int ii=0; ii < Ps.rows; ii++ ){
      for( int jj=0; jj < Ps.cols; jj++ ){
         printf("%f ", Ps.at<float>(ii,jj) );
      }
      printf("\n");
   }*/

   /*Find set base Z with K-means*/
   printf("Find features bases ...\n");
   kmeans( Ps, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers );
   printf("Features bases found ...\n");
   /*Centers*/
   /*printf("CENTROIDES\n");
   for( int ii=0; ii < centers.rows; ii++ ){
      for( int jj=0; jj < centers.cols; jj++ ){
         printf("%f ", centers.at<float>(ii,jj) );
      }
      printf("\n");
   }*/

   /*Construct Efficient Kernel Descriptor*/
   Z = centers.clone( );

   eval_kernel( Z, Z, K, 1 );
   invert(K,iK,DECOMP_CHOLESKY);

   Cholesky( iK, G );

   /*printf("\nG=\n");
   for( int i=0; i < G.rows; i++ ){
      for( int j=0; j < G.cols; j++ ){
        printf("%f ", G.at<float>(i,j) );
      }
      printf("\n");
   }*/


   /*Train SVM*/   

   /*Variables SVM Train*/
   
   int size_set_train_t = 0;
   int size_set_train_f = 0;
   char patch_src_t[100];
   char patch_src_f[100];
   char total_patch_src_t[100];
   char total_patch_src_f[100];

   strcpy( patch_src_t, argv[2] );
   strcpy( patch_src_f, argv[3] );

   size_set_train_t = getSizePaths( patch_src_t );
   size_set_train_f = getSizePaths( patch_src_f );
   char** patch_eye_detection_t = (char**) malloc( size_set_train_t*sizeof(char*) );
   char** patch_eye_detection_f = (char**) malloc( size_set_train_f*sizeof(char*) );
   getPathImages( patch_src_t, patch_eye_detection_t, 100 );
   getPathImages( patch_src_f, patch_eye_detection_f, 100 );
   
   cv::Mat train_data( size_set_train_t+size_set_train_f, Z.rows , CV_32F, cv::Scalar(0,0) );
   cv::Mat labels_data( size_set_train_t+size_set_train_f, 1, CV_32F, cv::Scalar(0,0) );
   cv::Mat kernel_descriptor;
   cv::Mat kernel_descriptor_t;

   cv::SVM svm;
   CvSVMParams params;
   params.kernel_type = CvSVM::LINEAR; //CvSVM::RBF, CvSVM::LINEAR ...
   params.degree = 0; // for poly
   //params.gamma = 0.05; // for poly/rbf/sigmoid
   params.gamma = 0.005; // for poly/rbf/sigmoid
   params.coef0 = 0; // for poly/sigmoid

   params.C = 7; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
   params.nu = 0.0; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
   params.p = 0.0; // for CV_SVM_EPS_SVR
   params.svm_type = CvSVM::C_SVC;
   //params.kernel_type = CvSVM::RBF;
   params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER, 10000, 1e-6 );
   


   j = 0;
   cv::Mat img;
   cv::Mat set_img_features(9,dim_patch,CV_32F);
   for( int i=0; *(patch_eye_detection_t+i); i++ ){
      strcpy( total_patch_src_t, patch_src_t );
      strcat( total_patch_src_t, "/" );
      strcat( total_patch_src_t, *(patch_eye_detection_t+i) );
      img = cv::imread( total_patch_src_t, CV_LOAD_IMAGE_GRAYSCALE );
      //imshow("Detection",im);
      resize( img, img, cv::Size(resize_img,resize_img) );

      cv::HOGDescriptor d( cv::Size(img.rows,img.cols), cv::Size(size_patch,size_patch), cv::Size(size_block,size_block), cv::Size(size_step,size_step), bins );
      d.compute( img, descriptorsValues, cv::Size(0,0), cv::Size(0,0), locations );

      char *buff1;
      char *buff2;
 
      for( int k=0; k < 9; k++ ){
         buff1 = (char*) set_img_features.row(k).data;
         buff2 = (char*) &descriptorsValues[0] + k*dim_patch;
         memcpy( buff1, buff2, sizeof(float)*dim_patch );
      }
      
      eval_efficient_kernel_descriptor( set_img_features, Z, G, dim_patch, kernel_descriptor );
      transpose( kernel_descriptor, kernel_descriptor_t );
      kernel_descriptor_t.copyTo( train_data.row(j) );
      labels_data.at<float>(j,1) = 1;
      j++;
 
      if(!*(patch_eye_detection_t+i)) free(*(patch_eye_detection_t+i));
      img.release();
   }

   if(!patch_eye_detection_t) free(patch_eye_detection_t);

   for( int i=0; *(patch_eye_detection_f+i); i++ ){
      strcpy( total_patch_src_f, patch_src_f );
      strcat( total_patch_src_f, "/" );
      strcat( total_patch_src_f, *(patch_eye_detection_f+i) );
      img = cv::imread( total_patch_src_f, CV_LOAD_IMAGE_GRAYSCALE );
      //imshow("Detection",im);
      resize( img, img, cv::Size(resize_img,resize_img) );

      cv::HOGDescriptor d( cv::Size(img.rows,img.cols), cv::Size(size_patch,size_patch), cv::Size(size_block,size_block), cv::Size(size_step,size_step), bins );
      d.compute( img, descriptorsValues, cv::Size(0,0), cv::Size(0,0), locations );

      char *buff1;
      char *buff2;

      for( int k=0; k < 9; k++ ){
         buff1 = (char*) set_img_features.row(k).data;
         buff2 = (char*) &descriptorsValues[0] + k*dim_patch;
         memcpy( buff1, buff2, sizeof(float)*dim_patch );
      }
      
      eval_efficient_kernel_descriptor( set_img_features, Z, G, dim_patch, kernel_descriptor );
      transpose( kernel_descriptor, kernel_descriptor_t );
      kernel_descriptor_t.copyTo( train_data.row(j) );
      labels_data.at<float>(j,1) = 0;
      j++;

      if(!*(patch_eye_detection_f+i)) free(*(patch_eye_detection_f+i));
      img.release();
   }

   if(!patch_eye_detection_f) free(patch_eye_detection_f);
   
   printf("Training SVM ...\n");
   svm.train( train_data, labels_data, cv::Mat(), cv::Mat(), params );
   svm.save("modelo_svm.xml");
   printf("Finish training ...\n");

   VideoCapture cap(CV_CAP_ANY);
   //cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
   cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
   //cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
   cap.set(CV_CAP_PROP_FRAME_HEIGHT, 360);
   if (!cap.isOpened())
       return -1;

   Mat frame;
   Mat frame_gray;
   Mat crop;
   Mat res;
   Mat gray;
   
   const std::string xml = "haarcascade_eye.xml";
   cv::CascadeClassifier eyeCascade;
   if (!eyeCascade.load(xml)) {
      std::cout << "Erro carregando faceCascade. Funcao \"detect_eye\"." << std::endl;
      return -1;
   }

   
   namedWindow("video capture", CV_WINDOW_AUTOSIZE);
   while (true)
   {
      cap >> frame;
      if (!frame.data)
         continue;

      cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
      equalizeHist(frame_gray, frame_gray);

      std::vector<cv::Rect> eyes;

      //En el paper utilizan el parametro de scala 1.05 aqui es de 1.1
      //faceCascade.detectMultiScale(frame_gray, faces, 1.01, 2, 0 | CASCADE_SCALE_IMAGE, cv::Size(30,30));
      eyeCascade.detectMultiScale(frame_gray, eyes, 1.5, 3, 0 | CASCADE_SCALE_IMAGE, cv::Size(3,3) );

      // Set Region of Interest
      cv::Rect roi_b;
      cv::Rect roi_c;

      size_t ic = 0; // ic is index of current element
      int ac = 0; // ac is area of current element

      size_t ib = 0; // ib is index of biggest element
      int ab = 0; // ab is area of biggest element

      for (ic = 0; ic < eyes.size(); ic++) // Iterate through all current elements (detected faces)
      {
         roi_c.x = eyes[ic].x;
         roi_c.y = eyes[ic].y;
         roi_c.width = (eyes[ic].width);
         roi_c.height = (eyes[ic].height);

         ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)

         roi_b.x = eyes[ib].x;
         roi_b.y = eyes[ib].y;
         roi_b.width = (eyes[ib].width);
                    roi_b.height = (eyes[ib].height);

         ab = roi_b.width * roi_b.height; // Get the area of biggest element, at beginning it is same as "current" element

         if (ac > ab){
            ib = ic;
            roi_b.x = eyes[ib].x;
            roi_b.y = eyes[ib].y;
            roi_b.width = (eyes[ib].width);
            roi_b.height = (eyes[ib].height);
         }

         crop = frame(roi_b);

         resize( crop, crop, cv::Size(resize_img,resize_img) );
         cv::HOGDescriptor d( cv::Size(crop.rows,crop.cols), cv::Size(size_patch,size_patch), cv::Size(size_block,size_block), cv::Size(size_step,size_step), bins );
         d.compute( crop, descriptorsValues, cv::Size(0,0), cv::Size(0,0), locations );

         char *buff1;
         char *buff2;

         for( int k=0; k < 9; k++ ){
            buff1 = (char*) set_img_features.row(k).data;
            buff2 = (char*) &descriptorsValues[0] + k*dim_patch;
            memcpy( buff1, buff2, sizeof(float)*dim_patch );
         }

         eval_efficient_kernel_descriptor( set_img_features, Z, G, dim_patch, kernel_descriptor );
         transpose( kernel_descriptor, kernel_descriptor_t );

         float result = svm.predict( kernel_descriptor_t );
         //resize(crop, res, cv::Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
         //cvtColor(crop, gray, cv::COLOR_BGR2GRAY); // Convert cropped image to Grayscale
         if( result == 1 ){
            cv::Point pt1(eyes[ic].x, eyes[ic].y); // Display detected faces on main window - live stream from camera
            cv::Point pt2((eyes[ic].x + eyes[ic].height), (eyes[ic].y + eyes[ic].width));
            rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
         }

      }

      imshow("video capture", frame);
      if (waitKey(20) >= 0)
          break;
   }
   
   return 0;
}
