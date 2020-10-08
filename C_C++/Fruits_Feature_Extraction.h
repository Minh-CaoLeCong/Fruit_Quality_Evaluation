#ifndef FRUITS_FEATURE_EXTRACTION_H
#define FRUITS_FEATURE_EXTRACTION_H

#include "OpenCV.h"
#include "Sorting_Algorithms.h"
#include "Fruits_Segmentation.h"

using namespace cv;

extern Mat Dimensions_Image;
extern Mat Area_Image;
extern Mat Isolate_Image;

extern string OutputFolder_AreaImage_Path;
extern string OutputFolder_DimensionImage_Path;
extern string OutputFolder_IsolateImage_Path;
extern string Output_FeatureExtraction_TextFileName_Path;

extern FILE *fp;

extern int fruit_height;
extern int fruit_width;

extern double fruit_contourArea;
extern double fruit_contourPerimeter;

extern void Fruits_Feature_Extraction(void);
extern void Import(void);
extern void Fruit_Dimensions(void);
extern void Fruit_Perimeter_Area_Contour(void);
extern void Fruit_Area_Image(void);
extern void Export_Ini(void);
extern void Export(void);


#endif // FRUITS_FEATURE_EXTRACTION_H