#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>

using namespace cv;
using namespace cv::cuda;

int main(int argc, char** argv)
{
    // Load the image
    Mat src = imread("input.jpg", IMREAD_GRAYSCALE);

    // Convert the image to a GpuMat
    GpuMat d_src(src);

    // Create a GpuMat to store the output
    GpuMat d_dst;

    // Detect corners using the Harris corner detection algorithm
    Ptr<cuda::CornersDetector> detector = cuda::createHarrisDetector();
    detector->detect(d_src, d_dst);

    // Copy the output back to the host
    Mat dst(d_dst);

    // Draw circles around the corners
    for (int i = 0; i < dst.rows; ++i)
    {
        for (int j = 0; j < dst.cols; ++j)
        {
            if (dst.at<float>(i, j) > 0)
            {
                circle(src, Point(j, i), 5, Scalar(0, 0, 255), 2);
            }
        }
    }

    // Display the output
    imshow("Output", src);
    waitKey();

    return 0;
}
