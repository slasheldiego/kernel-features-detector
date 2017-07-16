//
//  main.cpp
//  ReproductorVideo
//
//  Created by Diego Benavides on 6/04/15.
//  Copyright (c) 2015 Diego Benavides. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "tools-deteccion.hpp"

int g_slider_position = 0;
CvCapture* g_capture = NULL;

void onTrackbarSlide(int pos)
{
    cvSetCaptureProperty(
        g_capture,
        CV_CAP_PROP_POS_FRAMES,
        pos
        );
}

using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
    

    cvNamedWindow( "Reproductor" , CV_WINDOW_AUTOSIZE);
   
    if( argv[1] == NULL ){
	printf( "Debe colocar la ruta del video como primer parametro ...\n" );
	exit(-1);
    }

 
    cv::CascadeClassifier eyeCascade;

    if (!eyeCascade.load( argv[2] )) {
       std::cout << "Erro carregando faceCascade. Funcao \"detect_eye\"." << std::endl;
       return -1;
    }

    g_capture = cvCreateFileCapture( argv[1] );
    
    int frames = (int) cvGetCaptureProperty(
                            g_capture,
                            CV_CAP_PROP_FRAME_COUNT
                                          );
    
    if( frames!= 0 ){
        cvCreateTrackbar(
                "Position",
                "Reproductor",
                &g_slider_position,
                frames,
                onTrackbarSlide
                        );
    }

    
    IplImage* frame;
    Mat frame_m_gray;

    while( 1 ){
	printf("===>\n");
        frame = cvQueryFrame( g_capture );
        Mat frame_m( frame );
        cvtColor( frame_m, frame_m_gray, COLOR_BGR2GRAY );
	equalizeHist( frame_m_gray, frame_m_gray );
  
        //resize( frame_m, frame_m, cv::Size(480,240), 0, 0, INTER_LINEAR );
        //resize( frame_m_gray, frame_m_gray, cv::Size(480,240), 0, 0, INTER_LINEAR );

        std::vector<cv::Rect> eyes;

        //eyeCascade.detectMultiScale(frame_m_gray, eyes, 1.09, 3, 0 | CASCADE_SCALE_IMAGE, cv::Size(30,30), Size(60,60));
        eyeCascade.detectMultiScale(frame_m_gray, eyes, 1.9, 1, 0 | CASCADE_SCALE_IMAGE, cv::Size(10,10), Size(60,60));

        getROIDetection( &frame_m, eyes );

        if ( !frame ) break;
        //cvShowImage( "Reproductor" , frame );
	imshow( "Reproductor", frame_m );
        char c = cvWaitKey( 10 );
        if ( c == 'q' ) break;
    }
    
    cvReleaseCapture( &g_capture );
    cvDestroyWindow( "Reproductor" );
   
    
    return 0;
}
