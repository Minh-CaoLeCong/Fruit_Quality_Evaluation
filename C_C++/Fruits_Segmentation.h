#ifndef FRUITS_SEGMENTATION_H
#define FRUITS_SEGMENTATION_H

#include "OpenCV.h"
#include "Sorting_Algorithms.h"

using namespace cv;

extern int M;
extern int N;

extern char InputFolder_Path[100];
extern char OutputFolder_Path[100];

extern string OutputFolder_GrayImage_Path;
extern string OutputFolder_BlurImage_Path;
extern string OutputFolder_InrangeImage_Path;
extern string OutputFolder_CannyImage_Path;
extern string OutputFolder_DilateImage_Path;
extern string OutputFolder_FloodFillImage_Path;
extern string OutputFolder_RemoveNoiseImage_Path;
extern string OutputFolder_ContourImage_Path;

extern Mat Original_Image;
extern Mat GrayScale_Image;
extern Mat Blur_Image;
extern Mat GaussianFilter_Image;
extern Mat inRange_Image;
extern Mat Threshold_Image;
extern Mat Canny_Image;
extern Mat Dilate_Image;
extern Mat FloodFill_Image;
extern Mat RemoveNoise_Image;
extern Mat DrawFruitContour_Image;

// OPENCV-CUDA
extern cv::cuda::GpuMat GPU_Original_Image;
extern cv::cuda::GpuMat GPU_GrayScale_Image;
extern cv::cuda::GpuMat GPU_GaussianFilter_Image;
extern cv::cuda::GpuMat GPU_Threshold_Image;
extern cv::cuda::GpuMat GPU_Canny_Image;
extern cv::cuda::GpuMat GPU_Dilate_Image;
extern cv::cuda::GpuMat GPU_FloodFill_Image;
extern cv::cuda::GpuMat GPU_RemoveNoise_Image;
extern cv::cuda::GpuMat GPU_DrawFruitContour_Image;

extern vector<string> vFileNames;
extern int FileNames_index;
//extern vector<string> vPathFileNames;
extern int Num_FileNames;
extern int Num_PathFileNames;

extern int check_mkdir;

extern vector<vector<Point>> contours;
extern vector<Vec4i> hierarchy;

extern int area[MaxIntensity];
extern int max_area;
extern int rmax_area;

extern int num_contours;

extern int largest_contour_value;
extern int largest_contour_index;

extern void GetInput(void);
extern vector<string> GetFileNamesInDirectory(string directory);
extern void Remove_Noise(string InputImagePath);
extern void Fruits_Segmentation(void);


#endif // FRUITS_SEGMENTATION_H