
using namespace cv;

void getROIDetection( cv::Mat *image, std::vector<cv::Rect> det_rois );

void Cholesky( const cv::Mat &A, cv::Mat &S );

void eval_kernel( const Mat &m1, const Mat &m2, Mat &K, double lambda, int kernel_type );

void eval_efficient_kernel_descriptor( const Mat &set_img_features,
                                       const Mat &Z,
                                       const Mat &G,
                                       const int dim_patch,
                                       const double lambda,
                                       const int kernel_type,
                                       Mat &kernel_descriptor );
