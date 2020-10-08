#include "Fruits_Segmentation.h"

using namespace cv;

int M;
int N;

char InputFolder_Path[100];
char OutputFolder_Path[100];

string OutputFolder_GrayImage_Path;
string OutputFolder_BlurImage_Path;
string OutputFolder_InrangeImage_Path;
string OutputFolder_CannyImage_Path;
string OutputFolder_DilateImage_Path;
string OutputFolder_FloodFillImage_Path;
string OutputFolder_RemoveNoiseImage_Path;
string OutputFolder_ContourImage_Path;

vector<string> vFileNames;
int FileNames_index;
//vector<string> vPathFileNames;

int Num_FileNames;
int Num_PathFileNames;

int check_mkdir;

Mat Original_Image;
Mat GrayScale_Image;
Mat Blur_Image;
Mat inRange_Image;
Mat Canny_Image;
Mat Dilate_Image;
Mat FloodFill_Image;
Mat RemoveNoise_Image;
Mat DrawFruitContour_Image;

int area[MaxIntensity];
int max_area;
int rmax_area;

// initialize before using 'findContours' function
vector<vector<Point>> contours;
vector<Vec4i> hierarchy;
int num_contours;

int largest_contour_value;
int largest_contour_index;

void Fruits_Segmentation(void)
{
	printf("[INFOR]: FRUITS SEGMENTATION C/C++\n");

	GetInput();

	cout << "----------START-PROCESSING----------" << endl;

	time_t totalTime_Start;
	time_t totalTime_End;

	time(&totalTime_Start);
	for (FileNames_index = 1; FileNames_index <= Num_FileNames; FileNames_index++)
	{
		printf("[INFOR]:\t%s\t%d/%d\n", vFileNames[FileNames_index].c_str(), FileNames_index, Num_FileNames);
		Remove_Noise(InputFolder_Path + vFileNames[FileNames_index]);
	}
	time(&totalTime_End);

	double totalTime_Taken = double(totalTime_End - totalTime_Start);

	printf("[INFOR]: Total time execution: %f\n", totalTime_Taken);
	cout << "----------------DONE----------------" << endl;

	return;
}

void GetInput(void)
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
	//printf("Num_FileNames: %d\n", Num_FileNames);
	//for (int i = 1; i < Num_FileNames + 1; i++) 
	//{
	//	cout << vFileNames[i] << endl;
	//	//printf("%s\n", vFileNames[i]);
	//}

	InputFolder_Path[strlen(InputFolder_Path) - 1] = '\0'; // remove last char (*)

	//// combine input folder path with file name
	//for (int i = 0; i < Num_FileNames; i++)
	//{
	//	vPathFileNames.push_back(InputFolder_Path + vFileNames[i + 1]);
	//}
	//Num_PathFileNames = int(vPathFileNames.size());	// casting to int to remove warning:
	//												// warning C4267: '=': conversion from 'size_t' to 'int', possible loss of data.
	////printf("Num_PathFileNames: %d\n", Num_PathFileNames);
	////for (int i = 0; i < Num_PathFileNames; i++)
	////{
	////	cout << vPathFileNames[i] << endl;
	////	//printf("%s\n", vPathFileNames[i]);
	////}

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

	/*strcat(OutputFolder_BlurImage_Path, OutputFolder_Path);
	strcat(OutputFolder_BlurImage_Path, "blur/");						// append 'blur'

	strcat(OutputFolder_InrangeImage_Path, OutputFolder_Path);
	strcat(OutputFolder_InrangeImage_Path, "inrange/");					// append 'inrange'

	strcat(OutputFolder_CannyImage_Path, OutputFolder_Path);
	strcat(OutputFolder_CannyImage_Path, "canny/");						// append 'canny'

	strcat(OutputFolder_DilateImage_Path, OutputFolder_Path);
	strcat(OutputFolder_DilateImage_Path, "dilate/");					// append 'dilate'

	strcat(OutputFolder_FloodFillImage_Path, OutputFolder_Path);
	strcat(OutputFolder_FloodFillImage_Path, "floodfill/");				// append 'floodfill'

	strcat(OutputFolder_RemoveNoiseImage_Path, OutputFolder_Path);
	strcat(OutputFolder_RemoveNoiseImage_Path, "remove_noise/");		// append 'remove_noise'

	strcat(OutputFolder_ContourImage_Path, OutputFolder_Path);
	strcat(OutputFolder_ContourImage_Path, "contour/");					// append 'contour'
	cout << OutputFolder_ContourImage_Path << endl;
	printf("%s\n", OutputFolder_ContourImage_Path);*/
	
	return;
}

vector<string> GetFileNamesInDirectory(string directory) 
{
	vector<string> files;
	WIN32_FIND_DATA fileData;
	HANDLE hFind;
	if (!((hFind = FindFirstFile(directory.c_str(), &fileData)) == INVALID_HANDLE_VALUE)) 
	{				
		while (FindNextFile(hFind, &fileData)) 
		{		
			files.push_back(fileData.cFileName);
		}
	}
	FindClose(hFind);
	return files;
}

void Remove_Noise(string InputImagePath)
{
	// load image
	Original_Image = imread(InputImagePath);
	/*namedWindow("Original_Image", CV_WINDOW_AUTOSIZE);
	imshow("Original_Image", Original_Image);*/

	// convert to grayscale
	cvtColor(Original_Image, GrayScale_Image, COLOR_RGB2GRAY);
	/*namedWindow("GrayScale_Image", CV_WINDOW_AUTOSIZE);
	imshow("GrayScale_Image", GrayScale_Image);*/
	imwrite(OutputFolder_GrayImage_Path + vFileNames[FileNames_index], GrayScale_Image);

	M = GrayScale_Image.size().height;	// height of image
	N = GrayScale_Image.size().width;	// width of image

	// filter image and reduce noise of background
	blur(GrayScale_Image, Blur_Image, Size(5, 5));
	/*namedWindow("Blur_Image", CV_WINDOW_AUTOSIZE);
	imshow("Blur_Image", Blur_Image);*/
	imwrite(OutputFolder_BlurImage_Path + vFileNames[FileNames_index], Blur_Image);

	// inRange
	inRange(Blur_Image, 30, 250, inRange_Image);
	/*namedWindow("inRange_Image", CV_WINDOW_AUTOSIZE);
	imshow("inRange_Image", inRange_Image);*/
	imwrite(OutputFolder_InrangeImage_Path + vFileNames[FileNames_index], inRange_Image);

	// using canny algorithm to find out the edge
	Canny(inRange_Image, Canny_Image, 0, 3 * 0);
	/*namedWindow("Canny_Image", CV_WINDOW_AUTOSIZE);
	imshow("Canny_Image", Canny_Image);*/
	imwrite(OutputFolder_CannyImage_Path + vFileNames[FileNames_index], Canny_Image);

	// using dilate operation to try to find connected edge components
	Mat w = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(Canny_Image, Dilate_Image, w);
	//namedWindow("Dilate_Image", CV_WINDOW_AUTOSIZE);
	//imshow("Dilate_Image", Dilate_Image);
	imwrite(OutputFolder_DilateImage_Path + vFileNames[FileNames_index], Dilate_Image);

	int gray = 10;
	int x, y;
	int r;
	FloodFill_Image = Dilate_Image.clone();
	// throughout each pixel value of the image
	for (x = 0; x < M; x++)
		for (y = 0; y < N; y++)
		{
			// take the current pixel value
			r = FloodFill_Image.at<uchar>(x, y);
			// check white pixel value
			if (r == MaxIntensity - 1)
			{
				// using 'floodFill' to mark or isolate portions of image
				floodFill(FloodFill_Image, Point(y, x), CV_RGB(gray, gray, gray));
				gray++;
			}
		}
	/*namedWindow("FloodFill_Image", CV_WINDOW_AUTOSIZE);
	imshow("FloodFill_Image", FloodFill_Image);*/
	imwrite(OutputFolder_FloodFillImage_Path + vFileNames[FileNames_index], FloodFill_Image);

	// determine area (the number of pixels) of each portions of image
	for (r = 0; r < MaxIntensity; r++)
		area[r] = 0;
	for (x = 0; x < M; x++)
		for (y = 0; y < N; y++)
		{
			r = FloodFill_Image.at<uchar>(x, y);
			if (r > 0) // the pixel value of each portions of image is now equal or above 'GRAY_FLOODFILL'
				area[r]++;
		}
	max_area, rmax_area = maxElementIndex(area, MaxIntensity);
	// remove noise
	RemoveNoise_Image = Mat(FloodFill_Image.size(), CV_8UC1);
	for (x = 0; x < M; x++)
		for (y = 0; y < N; y++)
		{
			r = FloodFill_Image.at<uchar>(x, y);
			if (r == rmax_area)
				RemoveNoise_Image.at<uchar>(x, y) = MaxIntensity - 1;
			else
				RemoveNoise_Image.at<uchar>(x, y) = 0;
		}
	/*namedWindow("RemoveNoise_Image", CV_WINDOW_AUTOSIZE);
	imshow("RemoveNoise_Image", RemoveNoise_Image);*/
	imwrite(OutputFolder_RemoveNoiseImage_Path + vFileNames[FileNames_index], RemoveNoise_Image);

	// create a black background image to draw contours on it
	DrawFruitContour_Image = Mat::zeros(RemoveNoise_Image.size(), CV_8UC3);
	// find contours
	findContours(RemoveNoise_Image, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	// take the number of contours
	num_contours = (int)(contours.size());	// casting to int to remove warning:
											// warning C4267: '=': conversion from 'size_t' to 'int', possible loss of data.
	// find the largest contour
	largest_contour_value = int(contours[0].size());	// casting to int to remove warning:
														// warning C4267: '=': conversion from 'size_t' to 'int', possible loss of data.
	largest_contour_index = 0;
	for (int i = 1; i < num_contours; i++)
	{
		if (contours[i].size() > largest_contour_value)
		{
			largest_contour_value = int(contours[i].size());	// casting to int to remove warning:
																// warning C4267: '=': conversion from 'size_t' to 'int', possible loss of data.

			largest_contour_index = i;
		}
	}
	// draw the largest contour
	drawContours(DrawFruitContour_Image, contours, largest_contour_index, Scalar(0, 255, 0), 1, 8, hierarchy, 0, Point());
	/*namedWindow("DrawFruitContour_Image", CV_WINDOW_AUTOSIZE);
	imshow("DrawFruitContour_Image", DrawFruitContour_Image);*/
	imwrite(OutputFolder_ContourImage_Path + vFileNames[FileNames_index], DrawFruitContour_Image);

	return;
}