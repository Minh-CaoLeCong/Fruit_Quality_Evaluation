#ifndef OPEN_CV_H
#define OPEN_CV_H

#define MaxIntensity 256

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

// CUDA
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>

extern bool Cuda_Checked;

#endif //OPEN_CV_H