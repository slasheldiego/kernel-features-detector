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
   int size_patch = 22; //16
   int size_block = 11;  //8
   int size_step = 11;   //8
   int bins = 9;
   int dim_patch = 2*(size_patch/size_block)*bins;
   int count_feature = 0;

   /*Variables K-Means*/
   int clusterCount = atoi( argv[5] );
   cv::Mat labels;
   int attempts = 5;
   cv::Mat centers;

   /*Variables modelo*/
   double lambda = atof( argv[6] );
   int resize_img = 44; //32
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
   
   if( size_patch <= resize_img ){
      count_feature++;
      for ( int i = size_patch + size_step; i <= resize_img; i = i + size_step ){
          count_feature++;
      }
      count_feature = pow( count_feature, 2 );
   }else{
      printf("El valor de size_path es mayor que el de la image ...\n");
      exit(-1);
   }
   printf("Cantidad de caracteristicas: %d\n",count_feature);

   /*Extract HOGs descriptors*/

   size_set_feature_base = getSizePaths(path_src);
   cv::Mat Ps( count_feature*size_set_feature_base, dim_patch, CV_32F );
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
      printf("====>%s\n",*(paths_eye_detection+i));
      //imshow("Detection",im);
      equalizeHist( im, im );
      resize( im, im, cv::Size(resize_img,resize_img) );
      
      cv::HOGDescriptor d( cv::Size(im.rows,im.cols), cv::Size(size_patch,size_patch), cv::Size(size_block,size_block), cv::Size(size_step,size_step), bins );
      d.compute( im, descriptorsValues, cv::Size(0,0), cv::Size(0,0), locations ); 

      char* buf1;
      char* buf2;
      

      for( int k = 0; k < count_feature; k++ ){
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

   eval_kernel( Z, Z, K, lambda, 1 );
   invert(K,iK,DECOMP_CHOLESKY);

   Cholesky( iK, G );

   /*printf("\nG=\n");
   for( int i=0; i < G.rows; i++ ){
      for( int j=0; j < G.cols; j++ ){
        printf("%f ", G.at<float>(i,j) );
      }
      printf("\n");
   }*/

   printf("Starting Train SVM...\n");
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
   

   double media = 0;
   int cont = 0;
   j = 0;
   cv::Mat img;
   cv::Mat set_img_features(count_feature,dim_patch,CV_32F);
   for( int i=0; *(patch_eye_detection_t+i); i++ ){
      strcpy( total_patch_src_t, patch_src_t );
      strcat( total_patch_src_t, "/" );
      strcat( total_patch_src_t, *(patch_eye_detection_t+i) );
      //printf("=====>%s\n", *(patch_eye_detection_t+i));
      img = cv::imread( total_patch_src_t, CV_LOAD_IMAGE_GRAYSCALE );
      media = media + img.rows;
      cont++;
      //printf("=====>%d\n", img.rows );
      //imshow("Detection",im);
      equalizeHist( img, img );
      resize( img, img, cv::Size(resize_img,resize_img) );

      cv::HOGDescriptor d( cv::Size(img.rows,img.cols), cv::Size(size_patch,size_patch), cv::Size(size_block,size_block), cv::Size(size_step,size_step), bins );
      d.compute( img, descriptorsValues, cv::Size(0,0), cv::Size(0,0), locations );

      char *buff1;
      char *buff2;
 
      for( int k=0; k < count_feature; k++ ){
         buff1 = (char*) set_img_features.row(k).data;
         buff2 = (char*) &descriptorsValues[0] + k*dim_patch;
         memcpy( buff1, buff2, sizeof(float)*dim_patch );
      }
      
      eval_efficient_kernel_descriptor( set_img_features, Z, G, dim_patch, lambda, 1, kernel_descriptor );
      transpose( kernel_descriptor, kernel_descriptor_t );
      kernel_descriptor_t.copyTo( train_data.row(j) );
      labels_data.at<float>(j,1) = 1;
      j++;
 
      if(!*(patch_eye_detection_t+i)) free(*(patch_eye_detection_t+i));
      img.release();
   }
   printf("MEDIA: %f\n",media/cont);

   if(!patch_eye_detection_t) free(patch_eye_detection_t);

   for( int i=0; *(patch_eye_detection_f+i); i++ ){
      strcpy( total_patch_src_f, patch_src_f );
      strcat( total_patch_src_f, "/" );
      strcat( total_patch_src_f, *(patch_eye_detection_f+i) );
      img = cv::imread( total_patch_src_f, CV_LOAD_IMAGE_GRAYSCALE );
      //imshow("Detection",im);
      equalizeHist( img, img );
      resize( img, img, cv::Size(resize_img,resize_img) );

      cv::HOGDescriptor d( cv::Size(img.rows,img.cols), cv::Size(size_patch,size_patch), cv::Size(size_block,size_block), cv::Size(size_step,size_step), bins );
      d.compute( img, descriptorsValues, cv::Size(0,0), cv::Size(0,0), locations );

      char *buff1;
      char *buff2;

      for( int k=0; k < count_feature; k++ ){
         buff1 = (char*) set_img_features.row(k).data;
         buff2 = (char*) &descriptorsValues[0] + k*dim_patch;
         memcpy( buff1, buff2, sizeof(float)*dim_patch );
      }
      
      eval_efficient_kernel_descriptor( set_img_features, Z, G, dim_patch, lambda, 1, kernel_descriptor );
      transpose( kernel_descriptor, kernel_descriptor_t );
      kernel_descriptor_t.copyTo( train_data.row(j) );
      labels_data.at<float>(j,1) = 0;
      j++;

      if(!*(patch_eye_detection_f+i)) free(*(patch_eye_detection_f+i));
      img.release();
   }

   if(!patch_eye_detection_f) free(patch_eye_detection_f);
   
   printf("Making Model SVM ...\n");
   svm.train( train_data, labels_data, cv::Mat(), cv::Mat(), params );
   svm.save("modelo_svm.xml");
   printf("Finish training ...\n");


   int size_set_train_test = 0;
   char patch_src_test[100];
   char total_patch_src_test[100];

   strcpy( patch_src_test, argv[4] );

   size_set_train_test = getSizePaths( patch_src_test );
   char** patch_eye_detection_test = (char**) malloc( size_set_train_test*sizeof(char*) );
   getPathImages( patch_src_test, patch_eye_detection_test, 100 );

   int fp = 0;
   int fn = 0;
   int tn = 0;
   int tp = 0;
   double precision = 0;
   double recall = 0;
   double accuaracy = 0;
   double miss_rate = 0;


   for( int i=0; *(patch_eye_detection_test+i); i++ ){
      char** tokens;
      char clase;
      char *buf1;
      char *buf2;
      char patch_eye_detection_str[100];
      strcpy( patch_eye_detection_str, *(patch_eye_detection_test+i) );

      printf("=====>%s\n",*(patch_eye_detection_test+i));
      tokens = split_me( *(patch_eye_detection_test+i), '_' );
      if( tokens ){
         free(*tokens);
         free(*(tokens + 1));
         //free(*(tokens + 2));
         buf1 = (char*) *(tokens+2);
         buf2 = &clase;
         memcpy( buf2, buf1, 1 );
         free(*(tokens + 2));
         free(tokens);
      }
      
      strcpy( total_patch_src_test, patch_src_test );
      strcat( total_patch_src_test, "/" );
      strcat( total_patch_src_test, patch_eye_detection_str );
      img = cv::imread( total_patch_src_test, CV_LOAD_IMAGE_GRAYSCALE );
      //imshow("Detection",im);
      equalizeHist( img, img );
      resize( img, img, cv::Size(resize_img,resize_img) );
      
      cv::HOGDescriptor d( cv::Size(img.rows,img.cols), cv::Size(size_patch,size_patch), cv::Size(size_block,size_block), cv::Size(size_step,size_step), bins );
      d.compute( img, descriptorsValues, cv::Size(0,0), cv::Size(0,0), locations );

      char *buff1;
      char *buff2;

      for( int k=0; k < count_feature; k++ ){
         buff1 = (char*) set_img_features.row(k).data;
         buff2 = (char*) &descriptorsValues[0] + k*dim_patch;
         memcpy( buff1, buff2, sizeof(float)*dim_patch );
      }

      eval_efficient_kernel_descriptor( set_img_features, Z, G, dim_patch, lambda, 1, kernel_descriptor );
      transpose( kernel_descriptor, kernel_descriptor_t );

      float result = svm.predict( kernel_descriptor_t );

      printf("Clase:%c\n",clase);
      if( clase == 'f' && result == 0.0f ){
         tn++;
      }else if( clase == 't' && result == 1.0f ){
         tp++;
      }else if( clase == 'f' && result == 1.0f ){
         fp++;
      }else if( clase == 't' && result == 0.0f ){
         fn++;
      }
      printf("Resultado: %s: %f \n", patch_eye_detection_str, result);
      if(!*(patch_eye_detection_test+i)) free(*(patch_eye_detection_test+i));
      img.release();
     
   }

   printf("\nIndicadores\n\n");

   precision = (double) tp/(tp+fp);
   recall = (double) tp/(tp+fn);
   accuaracy = (double) (tp + tn)/( tp + tn + fp + fn );
   miss_rate = (double) fn/(fn+tp);

   printf( "Precision: %f \n", precision );
   printf( "Recall: %f \n", recall );
   printf( "Accuary: %f \n", accuaracy );
   printf( "Miss rate: %f \n", miss_rate );
   printf( "%d;%d;%d;%d\n",tp,tn,fp,fn);
   if(!patch_eye_detection_test) free(patch_eye_detection_test);

   
   return 0;
}
