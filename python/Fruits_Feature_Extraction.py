# export results to excel file
from openpyxl import load_workbook
import xlsxwriter
# using OpenCV library to processing image
import cv2
import numpy as np
# estimate time processing
from timeit import default_timer as timer
import time
# path
import os

print("[INFOR]: FRUITS FEATURE EXTRACTION.")

InputFileName_Idx = 0

def multiThread(fileNames_list):
    """
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(fruits_feature_extraction, fileNames_list)

def fruits_feature_extraction(fileName):
    global InputFileName_Idx
    global InputFolder_Path_str
    global OutputFolder_GrayImage_Path_str
    global OutputFolder_BlurImage_Path_str
    global OutputFolder_InrangeImage_Path_str
    global OutputFolder_CannyImage_Path_str
    global OutputFolder_DilateImage_Path_str
    global OutputFolder_FloodFillImage_Path_str
    global OutputFolder_RemoveNoiseImage_Path_str
    global OutputFolder_ContourImage_Path_str
    global OutputFolder_AreaImage_Path_str
    global OutputFolder_IsolateImage_Path_str
    global OutputFolder_DimensionImage_Path_str

    # the timing of processing
    Start_Time = timer()
    # load image
    Original_Image = cv2.imread(InputFolder_Path_str + fileName)

    if ImageProcessing_Method_str == "hsv":
        # convert BGR to HSV
        HSV_Image = cv2.cvtColor(Original_Image, cv2.COLOR_BGR2HSV)
        if SaveImageProcessingResult_Check_bool:
            # save image
            cv2.imwrite(OutputFolder_HsvImage_Path_str + fileName, HSV_Image)
        # convert to grayscale
        GrayScale_Image = cv2.cvtColor(HSV_Image, cv2.COLOR_BGR2GRAY)
        if SaveImageProcessingResult_Check_bool:
            # save image
            cv2.imwrite(OutputFolder_GrayImage_Path_str + fileName, GrayScale_Image)
    else:
        # convert to grayscale
        GrayScale_Image = cv2.cvtColor(Original_Image, cv2.COLOR_BGR2GRAY)
        if SaveImageProcessingResult_Check_bool:
            # save image
            cv2.imwrite(OutputFolder_GrayImage_Path_str + fileName, GrayScale_Image)

    M = GrayScale_Image.shape[0] # Height of image
    N = GrayScale_Image.shape[1] # Width of image

    # smooth image and reduce noise of background
    Blur_Image = cv2.blur(GrayScale_Image, Blur_KernelSize_tuple)
    if SaveImageProcessingResult_Check_bool:
        # save image
        cv2.imwrite(OutputFolder_BlurImage_Path_str + fileName, Blur_Image)

    if ImageProcessing_Method_str == "inrange":
        # inRange
        inRange_Image = cv2.inRange(Blur_Image, inRange1_int, inRange2_int)
        if SaveImageProcessingResult_Check_bool:
            # save image
            cv2.imwrite(OutputFolder_InrangeImage_Path_str + fileName, inRange_Image)

        # using canny algorithm to find out the edge
        Canny_Image = cv2.Canny(inRange_Image, ThresholdCanny1_int, ThresholdCanny2_int)
        if SaveImageProcessingResult_Check_bool:
            # save image
            cv2.imwrite(OutputFolder_CannyImage_Path_str + fileName, Canny_Image)
    else:
        # using canny algorithm to find out the edge
        Canny_Image = cv2.Canny(Blur_Image, ThresholdCanny1_int, ThresholdCanny2_int)
        if SaveImageProcessingResult_Check_bool:
            # save image
            cv2.imwrite(OutputFolder_CannyImage_Path_str + fileName, Canny_Image)

    # using dilate operation to try to find connected edge components
    Dilate_Image = cv2.dilate(Canny_Image, Dilate_KernelSize_np, iterations=1)
    if SaveImageProcessingResult_Check_bool:
        # save image
        cv2.imwrite(OutputFolder_DilateImage_Path_str + fileName, Dilate_Image)

    # using FloodFill to remove noise
    FloodFill_Image = Dilate_Image
    Gray_FloodFill = Gray_FloodFill_Ini_int
    Mask_FloodFill = np.zeros((M+2, N+2), np.uint8)
    count = 0
    for x in range(0, M):
        for y in range(0, N):
            r = Dilate_Image[x, y]
            if (r == L - 1):
                count += 1
                cv2.floodFill(FloodFill_Image, Mask_FloodFill, (y, x), Gray_FloodFill)
                Gray_FloodFill += Gray_FloodFill_Step_int
    if SaveImageProcessingResult_Check_bool:
        # save image
        cv2.imwrite(OutputFolder_FloodFillImage_Path_str + fileName, FloodFill_Image)
    FloodFill_AreaComponents_list = [0] * L
    for x in range(0, M):
        for y in range(0, N):
            r = FloodFill_Image[x, y]
            if (r > 0):
                FloodFill_AreaComponents_list[r] += 1
    FloodFill_AreaComponents_Enumerate = list(enumerate(FloodFill_AreaComponents_list))
    # sorting 'FloodFill_AreaComponents_Enumerate' from max to min
    FloodFill_AreaComponents_MaxMin = sorted(FloodFill_AreaComponents_Enumerate, key=lambda x:x [1], reverse = True)
    rmax = FloodFill_AreaComponents_MaxMin[FloodFill_AreaComponents_MaxMin_Idx0_int][FloodFill_AreaComponents_MaxMin_Idx1_int]
    RemoveNoise_Image = np.zeros((GrayScale_Image.shape[0], GrayScale_Image.shape[1], 1), np.uint8)
    for x in range(0, M):
        for y in range(0, N):
            r = FloodFill_Image[x, y]
            if (r == rmax):
                RemoveNoise_Image[x, y] = L - 1
            else:
                RemoveNoise_Image[x, y] = 0
    if SaveImageProcessingResult_Check_bool:
        # save image
        cv2.imwrite(OutputFolder_RemoveNoiseImage_Path_str + fileName, RemoveNoise_Image)

    # find contours
    """
    Because canny edge detector separate different segments,
    so, using 'findContours' to assemble those edge pixels.
    NOTE: The function 'findContours' computes contours from 
            binary images, so it take images created by 'Canny'
    """
    _, Contours, Hierarchy = cv2.findContours(RemoveNoise_Image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # find fruit contour
    Number_of_Contours = len(Contours)
    # using bubble sort to sorting contours from max to min
    Contours_Sorting_MaxMin = Contours.copy()
    for i in range(Number_of_Contours):
        for j in range(0, Number_of_Contours - i - 1):
            if (len(Contours_Sorting_MaxMin[j]) < len(Contours_Sorting_MaxMin[j + 1])):
                Contours_Sorting_MaxMin[j], Contours_Sorting_MaxMin[j + 1] = Contours_Sorting_MaxMin[j + 1], Contours_Sorting_MaxMin[j]
    # draw fruit contour '0'
    DrawFruitContour_Image = np.zeros((GrayScale_Image.shape[0], GrayScale_Image.shape[1], 1), np.uint8)
    cv2.drawContours(DrawFruitContour_Image, Contours_Sorting_MaxMin, Fruit_Contours_SortingMaxMin_Idx_int, (255, 0, 0), 1)
    if SaveImageProcessingResult_Check_bool:
        # save image
        cv2.imwrite(OutputFolder_ContourImage_Path_str + fileName, DrawFruitContour_Image)

    # estimate fruit area
    Fruit_Area = 0
    Fruit_Perimeter = 0
    Fruit_Area_Contour = 0
    Fruit_Perimeter = cv2.arcLength(Contours_Sorting_MaxMin[Fruit_Contours_SortingMaxMin_Idx_int], True)
    Fruit_Area_Contour = cv2.contourArea(Contours_Sorting_MaxMin[Fruit_Contours_SortingMaxMin_Idx_int])
    Area_Image = np.zeros((GrayScale_Image.shape[0], GrayScale_Image.shape[1], 1), np.uint8)
    Isolate_Image = np.zeros((Original_Image.shape[0], Original_Image.shape[1], 3), np.uint8)
    # throughout the pixel image to check it belongs inside contour or not
    for x in range (0, M):
        for y in range (0, N):
            # using pointPolygonTest to testing if a point is inside contour.
            # the return values are simply +1, -1, or 0 
            # depending on whether the point is inside, out-side, 
            # or on an edge.
            if (cv2.pointPolygonTest(Contours_Sorting_MaxMin[Fruit_Contours_SortingMaxMin_Idx_int], (y, x), False) > 0):
                Fruit_Area += 1
                Area_Image[x, y] = L - 1
                Isolate_Image[x, y] = Original_Image[x, y]
    if SaveImageProcessingResult_Check_bool:
        # save image
        cv2.imwrite(OutputFolder_AreaImage_Path_str + fileName, Area_Image)
        cv2.imwrite(OutputFolder_IsolateImage_Path_str + fileName, Isolate_Image)

    # Measure fruit dimensions
    # drawing rectangle around contour
    Dimensions_Image = DrawFruitContour_Image
    (X_Fruit_Contour, Y_Fruit_Contour, W_Fruit_Contour, H_Fruit_Contour) = cv2.boundingRect(Contours_Sorting_MaxMin[Fruit_Contours_SortingMaxMin_Idx_int])
    cv2.rectangle(Dimensions_Image, (X_Fruit_Contour, Y_Fruit_Contour), (X_Fruit_Contour + W_Fruit_Contour, Y_Fruit_Contour + H_Fruit_Contour), (255, 0, 0), 1)
    if SaveImageProcessingResult_Check_bool:
        # save image
        cv2.imwrite(OutputFolder_DimensionImage_Path_str + fileName, Dimensions_Image)

    Stop_Time = timer() - Start_Time

    # write measurements to worksheet
    Measurements_WorkSheet.write(InputFileName_Idx + 1, 0, fileName)
    Measurements_WorkSheet.write(InputFileName_Idx + 1, 1, Stop_Time)
    Measurements_WorkSheet.write(InputFileName_Idx + 1, 2, Fruit_Area)
    Measurements_WorkSheet.write(InputFileName_Idx + 1, 3, Fruit_Area_Contour)
    Measurements_WorkSheet.write(InputFileName_Idx + 1, 4, Fruit_Perimeter)
    Measurements_WorkSheet.write(InputFileName_Idx + 1, 5, W_Fruit_Contour)
    Measurements_WorkSheet.write(InputFileName_Idx + 1, 6, H_Fruit_Contour)
    Measurements_WorkSheet.write(InputFileName_Idx + 1, 7, str(FloodFill_AreaComponents_MaxMin))
    Measurements_WorkSheet.write(InputFileName_Idx + 1, 8, str(len(Contours_Sorting_MaxMin[0]))) 

    # cv2.waitKey(1)
    

InputFolder_Path_str = input("[ENTER]: Input directory (../dataset/mango/): ") # ../dataset/mango/
print("[INFOR]: Input directory: {}".format(InputFolder_Path_str))
OutputFolder_Path_str = input("[ENTER]: Output directory (../image_result/): ") # ../image_result/
print("[INFOR]: Output directory: {}".format(OutputFolder_Path_str))

FileName_Excel_Result_str = input("[ENTER]: Excel result file name (result): ") # result
print("[INFOR]: Excel result file name: {}".format(FileName_Excel_Result_str))

# check whether using the multi threading or not
MultiThreading_Check_str = input("[ENTER]: Multi threading? y/n: ")
if MultiThreading_Check_str == "y" or MultiThreading_Check_str == "Y" or\
    MultiThreading_Check_str == "yes" or MultiThreading_Check_str == "Yes":
    MultiThreading_Check_bool = True
    print("[INFOR]: Multi threading: {}".format(MultiThreading_Check_bool))
elif MultiThreading_Check_str == "n" or MultiThreading_Check_str == "N" or\
    MultiThreading_Check_str == "no" or MultiThreading_Check_str == "No":
    MultiThreading_Check_bool = False
    print("[INFOR]: Multi threading: {}".format(MultiThreading_Check_bool))
else:
    MultiThreading_Check_bool = False
    print("[INFOR]: Multi threading: (Default) - {}".format(MultiThreading_Check_bool))

# choose image processing method
ImageProcessing_Method_str = input('''[INFOR]: Choosing image processing method:
[INFOR]:    Convert to HSV channel: 1
[INFOR]:    InRange: 2
[INFOR]:    Threshold: 3
[INFOR]:    Default (NOT convert to HSV channel, inrange and threshold): 0
[ENTER]: (0, 1, 2, or 3): ''')
if ImageProcessing_Method_str == "1":
    ImageProcessing_Method_str = "hsv"
    print("[INFOR]: Image processing method: {}".format(ImageProcessing_Method_str))
elif ImageProcessing_Method_str == "2":
    ImageProcessing_Method_str = "inrange"
    print("[INFOR]: Image processing method: {}".format(ImageProcessing_Method_str))
elif ImageProcessing_Method_str == "3":
    ImageProcessing_Method_str = "threshold"
    print("[INFOR]: Image processing method: {}".format(ImageProcessing_Method_str))
elif ImageProcessing_Method_str == "0":
    ImageProcessing_Method_str = "default"
    print("[INFOR]: Image processing method: {}".format(ImageProcessing_Method_str))
else:
    ImageProcessing_Method_str = "default"
    print("[INFOR]: Image processing method: {}".format(ImageProcessing_Method_str))

# check whether save image processing result or not
SaveImageProcessingResult_Check_str = input("[ENTER] Save image processing result? y/n: ")
if SaveImageProcessingResult_Check_str == "y" or SaveImageProcessingResult_Check_str == "Y" or\
    SaveImageProcessingResult_Check_str == "yes" or SaveImageProcessingResult_Check_str == "Yes":
    SaveImageProcessingResult_Check_bool = True
    print("[INFOR]: Save image processing result: {}".format(SaveImageProcessingResult_Check_bool))
elif SaveImageProcessingResult_Check_str == "n" or SaveImageProcessingResult_Check_str == "N" or\
    SaveImageProcessingResult_Check_str == "no" or SaveImageProcessingResult_Check_str == "No":
    SaveImageProcessingResult_Check_bool = False
    print("[INFOR]: Save image processing result: {}".format(SaveImageProcessingResult_Check_bool))
else:
    SaveImageProcessingResult_Check_bool = False
    print("[INFOR]: Save image processing result: (Default) - {}".format(SaveImageProcessingResult_Check_bool))

# the number of possible intensity levels in the image 
# (256 for an 8-bit image).
L = 256

# initialise the kernel size of 'blur' function.
"""
it means each pixel in the output is the simple mean
of all of the pixels in a window (the kernel),
around the corresponding pixel in the input.
"""
Blur_KernelSize_tuple = (5, 5)
print("[INFOR]: Blur kernel size: " + str(Blur_KernelSize_tuple))

# inRange
if ImageProcessing_Method_str == "inrange":
    inRange1_int = 30
    inRange2_int = 250
    print("[INFOR]: inRange1: " + str(inRange1_int))
    print("[INFOR]: inRange2: " + str(inRange2_int))

# initialise two thresholds, a lower (ThresholdCanny1_int) 
# and an upper (ThresholdCanny2_int) of canny edge detector.
"""
If a pixel has a gradient larger than the upper threshold,
then it is accepted as an edge pixel; if a pixel is below
the lower threshold, it is rejected. 
If the pixel’s gradient is between the thresholds, 
then it will be accepted only if it is connected to 
a pixel that is above the high threshold.
NOTE: Canny recommended a ratio of high:low threshold
        between 2:1 and 3:1
"""
ThresholdCanny1_int = 0
ThresholdCanny2_int = 3 * ThresholdCanny1_int
print("[INFOR]: Threshold Canny 1: " + str(ThresholdCanny1_int))
print("[INFOR]: Threshold Canny 2: " + str(ThresholdCanny2_int))

# initialise the kernel of dilation function
"""
In dilation function, any given pixel is replaced with
the local maximum of all of the pixel values covered by
the kernel.
"""
Dilate_KernelSize_np = np.ones((3, 3), np.uint8)
print("[INFOR]: Dilate Kernel Size: ", Dilate_KernelSize_np)

# flood fill
Gray_FloodFill_Ini_int = 10
print("[INFOR]: Gray FloodFill: " + str(Gray_FloodFill_Ini_int))
Gray_FloodFill_Step_int = 1
print("[INFOR]: Gray FloodFill Step: " + str(Gray_FloodFill_Step_int))
FloodFill_AreaComponents_MaxMin_Idx0_int = 0
print("[INFOR]: FloodFill Area Components MaxMin Idx 0: " + str(FloodFill_AreaComponents_MaxMin_Idx0_int))
FloodFill_AreaComponents_MaxMin_Idx1_int = 0
print("[INFOR]: FloodFill Area Components MaxMin Idx 1: " + str(FloodFill_AreaComponents_MaxMin_Idx1_int))

# contour
Fruit_Contours_SortingMaxMin_Idx_int = 0
print("Fruit Contours Sorting MaxMin Idx: " + str(Fruit_Contours_SortingMaxMin_Idx_int))

InputFileNames_list = os.listdir(InputFolder_Path_str)
# print(InputFileNames_list)
# print(len(InputFileNames_list))
# print(type(InputFileNames_list))

if SaveImageProcessingResult_Check_bool == True:
    OutputFolder_GrayImage_Path_str = OutputFolder_Path_str + "gray/"
    if not os.path.exists(OutputFolder_GrayImage_Path_str):
        os.makedirs(OutputFolder_GrayImage_Path_str)
    OutputFolder_BlurImage_Path_str = OutputFolder_Path_str + "blur/"
    if not os.path.exists(OutputFolder_BlurImage_Path_str):
        os.makedirs(OutputFolder_BlurImage_Path_str)
    if ImageProcessing_Method_str == "inrange":
        OutputFolder_InrangeImage_Path_str = OutputFolder_Path_str + "inrange/"
        if not os.path.exists(OutputFolder_InrangeImage_Path_str):
            os.makedirs(OutputFolder_InrangeImage_Path_str)
    elif ImageProcessing_Method_str == "hsv":
        OutputFolder_HsvImage_Path_str = OutputFolder_Path_str + "hsv/"
        if not os.path.exists(OutputFolder_HsvImage_Path_str):
            os.makedirs(OutputFolder_HsvImage_Path_str)
    OutputFolder_CannyImage_Path_str = OutputFolder_Path_str + "canny/"
    if not os.path.exists(OutputFolder_CannyImage_Path_str):
        os.makedirs(OutputFolder_CannyImage_Path_str)
    OutputFolder_DilateImage_Path_str = OutputFolder_Path_str + "dilate/"
    if not os.path.exists(OutputFolder_DilateImage_Path_str):
        os.makedirs(OutputFolder_DilateImage_Path_str)
    OutputFolder_FloodFillImage_Path_str = OutputFolder_Path_str + "floodfill/"
    if not os.path.exists(OutputFolder_FloodFillImage_Path_str):
        os.makedirs(OutputFolder_FloodFillImage_Path_str)
    OutputFolder_RemoveNoiseImage_Path_str = OutputFolder_Path_str + "remove_noise/"
    if not os.path.exists(OutputFolder_RemoveNoiseImage_Path_str):
        os.makedirs(OutputFolder_RemoveNoiseImage_Path_str)
    OutputFolder_ContourImage_Path_str = OutputFolder_Path_str + "contour/"
    if not os.path.exists(OutputFolder_ContourImage_Path_str):
        os.makedirs(OutputFolder_ContourImage_Path_str)
    OutputFolder_AreaImage_Path_str = OutputFolder_Path_str + "area/"
    if not os.path.exists(OutputFolder_AreaImage_Path_str):
        os.makedirs(OutputFolder_AreaImage_Path_str)
    OutputFolder_DimensionImage_Path_str = OutputFolder_Path_str + "dimension/"
    if not os.path.exists(OutputFolder_DimensionImage_Path_str):
        os.makedirs(OutputFolder_DimensionImage_Path_str)
    OutputFolder_IsolateImage_Path_str = OutputFolder_Path_str + "isolate/"
    if not os.path.exists(OutputFolder_IsolateImage_Path_str):
        os.makedirs(OutputFolder_IsolateImage_Path_str)
    
# create workbook
WorkBook = xlsxwriter.Workbook(OutputFolder_Path_str + FileName_Excel_Result_str + ".xlsx")

# create worksheet for writing parameters
Parameters_WorkSheet = WorkBook.add_worksheet("Parameters")

# write parameters to worksheet
Parameters_WorkSheet.write('A1', "Blur Kernel Size: ")
if ImageProcessing_Method_str == "inrange":
    Parameters_WorkSheet.write('A2', "inRange1: ")
    Parameters_WorkSheet.write('A3', "inRange2: ")
Parameters_WorkSheet.write('A4', "Threshold Canny 1: ") 
Parameters_WorkSheet.write('A5', "Threshold Canny 2: ") 
Parameters_WorkSheet.write('A6', "Dilate Kernel Size: ") 
Parameters_WorkSheet.write('A7', "Gray FloodFill: ") 
Parameters_WorkSheet.write('A8', "Gray FloodFill Step: ") 
Parameters_WorkSheet.write('A9', "FloodFill Area Components MaxMin Idx 0: ") 
Parameters_WorkSheet.write('A10', "FloodFill Area Components MaxMin Idx 1: ") 
Parameters_WorkSheet.write('A11', "Fruit Contours Sorting MaxMin Idx: ") 
Parameters_Excel_File_list = []
Parameters_Excel_File_list.append(str(Blur_KernelSize_tuple))
if ImageProcessing_Method_str == "inrange":
    Parameters_Excel_File_list.append(str(inRange1_int))
    Parameters_Excel_File_list.append(str(inRange2_int))
else:
    Parameters_Excel_File_list.append("None")
    Parameters_Excel_File_list.append("None")
Parameters_Excel_File_list.append(str(ThresholdCanny1_int))
Parameters_Excel_File_list.append(str(ThresholdCanny2_int))
Parameters_Excel_File_list.append(str(Dilate_KernelSize_np))
Parameters_Excel_File_list.append(str(Gray_FloodFill_Ini_int))
Parameters_Excel_File_list.append(str(Gray_FloodFill_Step_int))
Parameters_Excel_File_list.append(str(FloodFill_AreaComponents_MaxMin_Idx0_int))
Parameters_Excel_File_list.append(str(FloodFill_AreaComponents_MaxMin_Idx1_int))
Parameters_Excel_File_list.append(str(Fruit_Contours_SortingMaxMin_Idx_int))
Parameters_Row_int = 0
Parameters_Column_int = 1
for item in Parameters_Excel_File_list:
    # write operation perform 
    Parameters_WorkSheet.write(Parameters_Row_int, Parameters_Column_int, item) 
    # incrementing the value of row by one with each iteratons. 
    Parameters_Row_int += 1

# create worksheet for writing measurements
Measurements_WorkSheet = WorkBook.add_worksheet("Measurements")
Measurements_WorkSheet.write('A1', "File_name")
Measurements_WorkSheet.write('B1', "Time")
Measurements_WorkSheet.write('C1', "Area")
Measurements_WorkSheet.write('D1', "Contour_Area")
Measurements_WorkSheet.write('E1', "Perimeter")
Measurements_WorkSheet.write('F1', "Width")
Measurements_WorkSheet.write('G1', "Height")
Measurements_WorkSheet.write('H1', "FloodFill_Area_Components_MaxMin")
Measurements_WorkSheet.write('I1', "Contours_Sorting_MaxMin")

print("------------START-PROCESSING------------")

if MultiThreading_Check_bool == True:
    StartTimeTotal = time.time()
    multiThread(InputFileNames_list)
    print("[INFOR]: Total time processing: {}".format(time.time() - StartTimeTotal))
    print("---------------DONE---------------")
else:
    StartTimeTotal = time.time()
    for InputFileName_Idx, InputFileName in enumerate(InputFileNames_list):
        print('File name:\t{}\t{}/{}'.format(InputFileName, InputFileName_Idx + 1, len(InputFileNames_list)))
        fruits_feature_extraction(InputFileName)

    print("[INFOR]: Total time processing: {}".format(time.time() - StartTimeTotal))
    print("---------------DONE---------------")



# close excel file
WorkBook.close()


# img = cv2.imread("../dataset/mango/Mango_01_A.JPG")
# cv2.imshow("img", img)
# test = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
# M = img.shape[0] # Height of image
# N = img.shape[1] # Width of image
# # for x in range (0, M):
# #         for y in range (0, N):
# #             test[x, y] = img[x, y]
# cv2.imshow("test", test)
# cv2.waitKey(0)