#include "Fruits_Feature_Extraction.h"

using namespace cv;

Mat Dimensions_Image;
Mat Area_Image;
Mat Isolate_Image;

string OutputFolder_AreaImage_Path;
string OutputFolder_DimensionImage_Path;
string OutputFolder_IsolateImage_Path;
string Output_FeatureExtraction_TextFileName_Path;

FILE *fp;

int fruit_height = 0;
int fruit_width = 0;

double fruit_contourArea = 0.0;
double fruit_contourPerimeter = 0.0;

void Fruits_Feature_Extraction(void)
{
	printf("[INFOR]: FRUITS FEATURE EXTRACTION C/C++\n");

	Import();
	Export_Ini();

	cout << "----------START-PROCESSING----------" << endl;

	time_t totalTime_Start;
	time_t totalTime_End;

	time(&totalTime_Start);
	for (FileNames_index = 1; FileNames_index <= Num_FileNames; FileNames_index++)
	{
		printf("[INFOR]:\t%s\t%d/%d\n", vFileNames[FileNames_index].c_str(), FileNames_index, Num_FileNames);
		Remove_Noise(InputFolder_Path + vFileNames[FileNames_index]);
		Fruit_Dimensions();
		Fruit_Perimeter_Area_Contour();
		Fruit_Area_Image();
		Export();
		//break;
	}
	time(&totalTime_End);

	double totalTime_Taken = double(totalTime_End - totalTime_Start);
	
	printf("[INFOR]: Total time execution: %f\n", totalTime_Taken);
	cout << "----------------DONE----------------" << endl;

	return;
}

void Import(void)
{
	// enter input directory
	printf("[ENTER]: Input directory (../dataset/mango/): ");
	scanf("%s", InputFolder_Path);
	printf("[INFOR]: Input directory: %s\n", InputFolder_Path);
	if (InputFolder_Path[strlen(InputFolder_Path) - 1] != '*')
	{
		strcat(InputFolder_Path, "*"); // append '*'
	}

	// get all file names of input directory
	vFileNames = GetFileNamesInDirectory(InputFolder_Path);
	Num_FileNames = int(vFileNames.size()) - 1;		// casting to int to remove warning:
													// warning C4267: '=': conversion from 'size_t' to 'int', possible loss of data.

	InputFolder_Path[strlen(InputFolder_Path) - 1] = '\0'; // remove last char (*)

	// enter output directory
	printf("[ENTER]: Output directory (../image_result/): ");
	scanf("%s", OutputFolder_Path);
	printf("[INFOR]: Output directory: %s\n", OutputFolder_Path);

	OutputFolder_GrayImage_Path = OutputFolder_Path;
	OutputFolder_GrayImage_Path = OutputFolder_GrayImage_Path.append("gray/");			// append 'gray'
	// create directory
	check_mkdir = _mkdir(OutputFolder_GrayImage_Path.c_str());
	if (!check_mkdir)
		printf("[INFOR]: The creation of GRAY image directory was: SUCCESSFUL.\n");
	else
		printf("[INFOR]: The creation of GRAY image directory was: UNSUCCESSFUL.\n");

	OutputFolder_BlurImage_Path = OutputFolder_Path;
	OutputFolder_BlurImage_Path = OutputFolder_BlurImage_Path.append("blur/");			// append 'blur'
	// create directory
	check_mkdir = _mkdir(OutputFolder_BlurImage_Path.c_str());
	if (!check_mkdir)
		printf("[INFOR]: The creation of BLUR image directory was: SUCCESSFUL.\n");
	else
		printf("[INFOR]: The creation of BLUR image directory was: UNSUCCESSFUL.\n");

	OutputFolder_InrangeImage_Path = OutputFolder_Path;
	OutputFolder_InrangeImage_Path = OutputFolder_InrangeImage_Path.append("inrange/");	// append 'inrange'
	// create directory
	check_mkdir = _mkdir(OutputFolder_InrangeImage_Path.c_str());
	if (!check_mkdir)
		printf("[INFOR]: The creation of INRANGE image directory was: SUCCESSFUL.\n");
	else
		printf("[INFOR]: The creation of INRANGE image directory was: UNSUCCESSFUL.\n");

	OutputFolder_CannyImage_Path = OutputFolder_Path;
	OutputFolder_CannyImage_Path = OutputFolder_CannyImage_Path.append("canny/");		// append 'canny'
	// create directory
	check_mkdir = _mkdir(OutputFolder_CannyImage_Path.c_str());
	if (!check_mkdir)
		printf("[INFOR]: The creation of CANNY image directory was: SUCCESSFUL.\n");
	else
		printf("[INFOR]: The creation of CANNY image directory was: UNSUCCESSFUL.\n");

	OutputFolder_DilateImage_Path = OutputFolder_Path;
	OutputFolder_DilateImage_Path = OutputFolder_DilateImage_Path.append("dilate/");	// append 'dilate'
	// create directory
	check_mkdir = _mkdir(OutputFolder_DilateImage_Path.c_str());
	if (!check_mkdir)
		printf("[INFOR]: The creation of DILATE image directory was: SUCCESSFUL.\n");
	else
		printf("[INFOR]: The creation of DILATE image directory was: UNSUCCESSFUL.\n");

	OutputFolder_FloodFillImage_Path = OutputFolder_Path;
	OutputFolder_FloodFillImage_Path = OutputFolder_FloodFillImage_Path.append("floodfill/"); // append 'floodfill'
	// create directory
	check_mkdir = _mkdir(OutputFolder_FloodFillImage_Path.c_str());
	if (!check_mkdir)
		printf("[INFOR]: The creation of FLOODFILL image directory was: SUCCESSFUL.\n");
	else
		printf("[INFOR]: The creation of FLOODFILL image directory was: UNSUCCESSFUL.\n");

	OutputFolder_RemoveNoiseImage_Path = OutputFolder_Path;
	OutputFolder_RemoveNoiseImage_Path = OutputFolder_RemoveNoiseImage_Path.append("remove_noise/"); // append 'remove_noise'
	// create directory
	check_mkdir = _mkdir(OutputFolder_RemoveNoiseImage_Path.c_str());
	if (!check_mkdir)
		printf("[INFOR]: The creation of REMOVE_NOISE image directory was: SUCCESSFUL.\n");
	else
		printf("[INFOR]: The creation of REMOVE_NOISE image directory was: UNSUCCESSFUL.\n");

	OutputFolder_ContourImage_Path = OutputFolder_Path;
	OutputFolder_ContourImage_Path = OutputFolder_ContourImage_Path.append("contour/");	// append 'contour'
	// create directory
	check_mkdir = _mkdir(OutputFolder_ContourImage_Path.c_str());
	if (!check_mkdir)
		printf("[INFOR]: The creation of CONTOUR image directory was: SUCCESSFUL.\n");
	else
		printf("[INFOR]: The creation of CONTOUR image directory was: UNSUCCESSFUL.\n");

	OutputFolder_DimensionImage_Path = OutputFolder_Path;
	OutputFolder_DimensionImage_Path = OutputFolder_DimensionImage_Path.append("dimension/");	// append 'dimension'
	// create directory
	check_mkdir = _mkdir(OutputFolder_DimensionImage_Path.c_str());
	if (!check_mkdir)
		printf("[INFOR]: The creation of DIMENSION image directory was: SUCCESSFUL.\n");
	else
		printf("[INFOR]: The creation of DIMENSION image directory was: UNSUCCESSFUL.\n");

	OutputFolder_AreaImage_Path = OutputFolder_Path;
	OutputFolder_AreaImage_Path = OutputFolder_AreaImage_Path.append("area/");	// append 'area'
	// create directory
	check_mkdir = _mkdir(OutputFolder_AreaImage_Path.c_str());
	if (!check_mkdir)
		printf("[INFOR]: The creation of AREA image directory was: SUCCESSFUL.\n");
	else
		printf("[INFOR]: The creation of AREA image directory was: UNSUCCESSFUL.\n");

	OutputFolder_IsolateImage_Path = OutputFolder_Path;
	OutputFolder_IsolateImage_Path = OutputFolder_IsolateImage_Path.append("isolate/");	// append 'isolate'
	// create directory
	check_mkdir = _mkdir(OutputFolder_IsolateImage_Path.c_str());
	if (!check_mkdir)
		printf("[INFOR]: The creation of ISOLATE image directory was: SUCCESSFUL.\n");
	else
		printf("[INFOR]: The creation of ISOLATE image directory was: UNSUCCESSFUL.\n");

	return;
}


void Fruit_Dimensions(void)
{
	// Measure fruit dimensions drawing rectangle around contour
	Dimensions_Image = DrawFruitContour_Image;
	vector<vector<Point>> contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	approxPolyDP(Mat(contours[largest_contour_index]), contours_poly[largest_contour_index], 3, true);
	boundRect[largest_contour_index] = boundingRect(Mat(contours_poly[largest_contour_index]));
	fruit_width = boundRect[largest_contour_index].width;
	fruit_height = boundRect[largest_contour_index].height;
	rectangle(Dimensions_Image, boundRect[largest_contour_index].tl(), boundRect[largest_contour_index].br(), Scalar(0, 0, 255), 1, 8, 0);
	/*namedWindow("Dimensions_Image", CV_WINDOW_AUTOSIZE);
	imshow("Dimensions_Image", Dimensions_Image);*/
	imwrite(OutputFolder_DimensionImage_Path + vFileNames[FileNames_index], Dimensions_Image);
	
	return;
}

void Fruit_Perimeter_Area_Contour(void)
{
	fruit_contourPerimeter = arcLength(contours[largest_contour_index], true);
	fruit_contourArea = contourArea(contours[largest_contour_index]);
	//cout << fruit_contourPerimeter << endl;
	//cout << fruit_contourArea << endl;
	return;
}

void Fruit_Area_Image(void)
{
	Isolate_Image = Mat::zeros(Original_Image.size(), CV_8UC3);
	Area_Image = Mat::zeros(Original_Image.size(), CV_8UC1);

	for (int x = 0; x < M; x++)
		for (int y = 0; y < N; y++)
		{
			if (int(pointPolygonTest(contours[largest_contour_index], Point2f(y, x), false)) > 0)
			{
				Area_Image.at<uchar>(x, y) = MaxIntensity - 1;
				Isolate_Image.at<Vec3b>(x, y) = Original_Image.at<Vec3b>(x, y);
			}
		}
	/*namedWindow("Area_Image", CV_WINDOW_AUTOSIZE);
	imshow("Area_Image", Area_Image);
	namedWindow("Isolate_Image", CV_WINDOW_AUTOSIZE);
	imshow("Isolate_Image", Isolate_Image);*/
	imwrite(OutputFolder_AreaImage_Path + vFileNames[FileNames_index], Area_Image);
	imwrite(OutputFolder_IsolateImage_Path + vFileNames[FileNames_index], Isolate_Image);

	return;
}

void Export_Ini(void)
{
	Output_FeatureExtraction_TextFileName_Path = OutputFolder_Path;
	Output_FeatureExtraction_TextFileName_Path = Output_FeatureExtraction_TextFileName_Path.append("fruit_feature_extraction.txt");

	fp = fopen(Output_FeatureExtraction_TextFileName_Path.c_str(), "w");

	return;
}

void Export(void)
{
	fprintf(fp, "%s\t%lf\t%lf\t%d\t%d\n", vFileNames[FileNames_index], fruit_contourArea, fruit_contourPerimeter, fruit_width, fruit_height);
	return;
}