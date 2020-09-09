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
import concurrent.futures

def multiThread(fileNames):
    """
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(fruit_feature_extraction, fileNames)

def fruit_feature_extraction(fileName):
    # load image
    # print('File Name:\t{}\t{}/{}'.format(fileName, File_Name_Idx + 1, len(fileName)))
    # the timing of processing
    Start_Time = timer()
    Original_Image = cv2.imread(Input_Folder_Path + fileName)

    # convert to grayscale
    GrayScale_Image = cv2.cvtColor(Original_Image, cv2.COLOR_BGR2GRAY)
    # save image
    cv2.imwrite(Output_Folder_Gray_Path + fileName, GrayScale_Image)

    M = GrayScale_Image.shape[0] #Height of image
    N = GrayScale_Image.shape[1] #Width of image

    # smooth image and reduce noise of background
    Blur_Image = cv2.blur(GrayScale_Image, Blur_KernelSize)
    # save image
    cv2.imwrite(Output_Folder_Blur_Path + fileName, Blur_Image)

    # inRange
    inRange_Image = cv2.inRange(Blur_Image, inRange1, inRange2)
    # save image
    cv2.imwrite(Output_Folder_inRange_Path + fileName, inRange_Image)

    # using canny algorithm to find out the edge
    Canny_Image = cv2.Canny(inRange_Image, ThresholdCanny1, ThresholdCanny2)
    # save image
    cv2.imwrite(Output_Folder_Canny_Path + fileName, Canny_Image)

    # using dilate operation to try to find connected edge components
    Dilate_Image = cv2.dilate(Canny_Image, Dilate_KernelSize, iterations=1)
    # save image
    cv2.imwrite(Output_Folder_Dilate_Path + fileName, Dilate_Image)

    # using FloodFill to remove noise
    FloodFill_Image = Dilate_Image
    Gray_FloodFill = Gray_FloodFill_Ini
    Mask_FloodFill = np.zeros((M+2, N+2), np.uint8)
    count = 0
    for x in range(0, M):
        for y in range(0, N):
            r = Dilate_Image[x, y]
            if (r == L - 1):
                count += 1
                cv2.floodFill(FloodFill_Image, Mask_FloodFill, (y, x), Gray_FloodFill)
                Gray_FloodFill += Gray_FloodFill_Increase_Step
    # save image
    cv2.imwrite(Output_Folder_FloodFill_Path + fileName, FloodFill_Image)
    FloodFill_Area_Components = [0] * L
    for x in range(0, M):
        for y in range(0, N):
            r = FloodFill_Image[x, y]
            if (r > 0):
                FloodFill_Area_Components[r] += 1
    FloodFill_Area_Components_Enumerate = list(enumerate(FloodFill_Area_Components))
    # sorting 'FloodFill_Area_Components_Enumerate' from max to min
    FloodFill_Area_Components_MaxMin = sorted(FloodFill_Area_Components_Enumerate, key=lambda x:x [1], reverse = True)
    rmax = FloodFill_Area_Components_MaxMin[FloodFill_Area_Components_MaxMin_Idx0][FloodFill_Area_Components_MaxMin_Idx1]
    RemoveNoise_Image = np.zeros((GrayScale_Image.shape[0], GrayScale_Image.shape[1], 1), np.uint8)
    for x in range(0, M):
        for y in range(0, N):
            r = FloodFill_Image[x, y]
            if (r == rmax):
                RemoveNoise_Image[x, y] = L - 1
            else:
                RemoveNoise_Image[x, y] = 0
    # save image
    cv2.imwrite(Output_Folder_RemoveNoise_Path + fileName, RemoveNoise_Image)

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
    cv2.drawContours(DrawFruitContour_Image, Contours_Sorting_MaxMin, Fruit_Contours_Sorting_MaxMin_Idx, (255, 0, 0), 1)
    # save image
    cv2.imwrite(Output_Folder_Contour_Path + fileName, DrawFruitContour_Image)

    # estimate fruit area
    Fruit_Area = 0
    Fruit_Perimeter = 0
    Fruit_Area_Contour = 0
    Fruit_Perimeter = cv2.arcLength(Contours_Sorting_MaxMin[Fruit_Contours_Sorting_MaxMin_Idx], True)
    Fruit_Area_Contour = cv2.contourArea(Contours_Sorting_MaxMin[Fruit_Contours_Sorting_MaxMin_Idx])
    Area_Image = np.zeros((GrayScale_Image.shape[0], GrayScale_Image.shape[1], 1), np.uint8)
    Isolate_Image = np.zeros((GrayScale_Image.shape[0], GrayScale_Image.shape[1], 1), np.uint8)
    # throughout the pixel image to check it belongs inside contour or not
    for x in range (0, M):
        for y in range (0, N):
            # using pointPolygonTest to testing if a point is inside contour.
            # the return values are simply +1, -1, or 0 
            # depending on whether the point is inside, out-side, 
            # or on an edge.
            if (cv2.pointPolygonTest(Contours_Sorting_MaxMin[Fruit_Contours_Sorting_MaxMin_Idx], (y, x), False) > 0):
                Fruit_Area += 1
                Area_Image[x, y] = L - 1
                Isolate_Image[x, y] = GrayScale_Image[x, y]
    # save image
    cv2.imwrite(Output_Folder_Area_Path + fileName, Area_Image)
    cv2.imwrite(Output_Folder_Isolate_Path + fileName, Isolate_Image)

    # Measure fruit dimensions
    # drawing rectangle around contour
    Dimensions_Image = DrawFruitContour_Image
    (X_Fruit_Contour, Y_Fruit_Contour, W_Fruit_Contour, H_Fruit_Contour) = cv2.boundingRect(Contours_Sorting_MaxMin[Fruit_Contours_Sorting_MaxMin_Idx])
    cv2.rectangle(Dimensions_Image, (X_Fruit_Contour, Y_Fruit_Contour), (X_Fruit_Contour + W_Fruit_Contour, Y_Fruit_Contour + H_Fruit_Contour), (255, 0, 0), 1)
    # save image
    cv2.imwrite(Output_Folder_Dimension_Path + fileName, Dimensions_Image)

    Stop_Time = timer() - Start_Time

    # write measurements to worksheet
    global File_Name_Idx
    Measurements_WorkSheet.write(File_Name_Idx + 1, 0, fileName)
    Measurements_WorkSheet.write(File_Name_Idx + 1, 1, Stop_Time)
    Measurements_WorkSheet.write(File_Name_Idx + 1, 2, Fruit_Area)
    Measurements_WorkSheet.write(File_Name_Idx + 1, 3, Fruit_Area_Contour)
    Measurements_WorkSheet.write(File_Name_Idx + 1, 4, Fruit_Perimeter)
    Measurements_WorkSheet.write(File_Name_Idx + 1, 5, W_Fruit_Contour)
    Measurements_WorkSheet.write(File_Name_Idx + 1, 6, H_Fruit_Contour)
    Measurements_WorkSheet.write(File_Name_Idx + 1, 7, str(FloodFill_Area_Components_MaxMin))
    Measurements_WorkSheet.write(File_Name_Idx + 1, 8, str(len(Contours_Sorting_MaxMin[0]))) 

    File_Name_Idx += 1

    # cv2.waitKey(1)
    

# the number of possible intensity levels in the image 
# (256 for an 8-bit image).
L = 256
# initialise the kernel size of 'blur' function.
"""
it means each pixel in the output is the simple mean
of all of the pixels in a window (the kernel),
around the corresponding pixel in the input.
"""
Blur_KernelSize = (5, 5)
print("Blur kernel size: " + str(Blur_KernelSize))

# inRange
inRange1 = 30
inRange2 = 250
print("inRange1: " + str(inRange1))
print("inRange2: " + str(inRange2))

# initialise two thresholds, a lower (ThresholdCanny1) 
# and an upper (ThresholdCanny2) of canny edge detector.
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
ThresholdCanny1 = 0
ThresholdCanny2 = 3 * ThresholdCanny1
print("Threshold Canny 1: " + str(ThresholdCanny1))
print("Threshold Canny 2: " + str(ThresholdCanny2))
# initialise the kernel of dilation function
"""
In dilation function, any given pixel is replaced with
the local maximum of all of the pixel values covered by
the kernel.
"""
Dilate_KernelSize = np.ones((3, 3), np.uint8)
print("Dilate Kernel Size: " + str(Dilate_KernelSize))
# flood fill
Gray_FloodFill_Ini = 10
print("Gray FloodFill: " + str(Gray_FloodFill_Ini))
Gray_FloodFill_Increase_Step = 1
print("Gray FloodFill Increase Num: " + str(Gray_FloodFill_Increase_Step))
FloodFill_Area_Components_MaxMin_Idx0 = 0
print("FloodFill Area Components MaxMin Idx 0: " + str(FloodFill_Area_Components_MaxMin_Idx0))
FloodFill_Area_Components_MaxMin_Idx1 = 0
print("FloodFill Area Components MaxMin Idx 1: " + str(FloodFill_Area_Components_MaxMin_Idx1))
# contour
Fruit_Contours_Sorting_MaxMin_Idx = 0
print("Fruit Contours Sorting MaxMin Idx: " + str(Fruit_Contours_Sorting_MaxMin_Idx))
# cv2.waitKey(100)

# Input and output image folder path
Input_Folder_Path = "../dataset/Mango/"
Files_Name_List = []
File_Name_Idx = 1
# the list of image file extensions
Image_Extention = ['.JPG', '.jpg', '.JPEG', '.jpeg', '.png', '.PNG']
# r = root, d = directories, f = files
for r, d, f in os.walk(Input_Folder_Path):
    for file_name in f:
        if any(Each_Image_Extension in file_name for Each_Image_Extension in Image_Extention):
            Files_Name_List.append(file_name)

Output_Folder_Gray_Path = "../image_result/inRange/Mango1/gray/"
Output_Folder_Blur_Path = "../image_result/inRange/Mango1/blur/"
Output_Folder_inRange_Path = "../image_result/inRange/Mango1/inrange/"
Output_Folder_Canny_Path = "../image_result/inRange/Mango1/canny/"
Output_Folder_Dilate_Path = "../image_result/inRange/Mango1/dilate/"
Output_Folder_FloodFill_Path = "../image_result/inRange/Mango1/floodfill/"
Output_Folder_RemoveNoise_Path = "../image_result/inRange/Mango1/remove_noise/"
Output_Folder_Contour_Path = "../image_result/inRange/Mango1/contour/"
Output_Folder_Area_Path = "../image_result/inRange/Mango1/area/"
Output_Folder_Isolate_Path = "../image_result/inRange/Mango1/isolate/"
Output_Folder_Dimension_Path = "../image_result/inRange/Mango1/dimension/"

# Excel file path
Output_Folder_Excel_File = "../image_result/inRange/Mango1/"

# create workbook
WorkBook = xlsxwriter.Workbook(Output_Folder_Excel_File + "Mango1_inRange_14_250.xlsx")

# create worksheet for writing parameters
Parameters_WorkSheet = WorkBook.add_worksheet("Parameters")
# write parameters to worksheet
Parameters_WorkSheet.write('A1', "Blur Kernel Size: ")
Parameters_WorkSheet.write('A2', "inRange1: ") 
Parameters_WorkSheet.write('A3', "inRange2: ") 
Parameters_WorkSheet.write('A4', "Threshold Canny 1: ") 
Parameters_WorkSheet.write('A5', "Threshold Canny 2: ") 
Parameters_WorkSheet.write('A6', "Dilate Kernel Size: ") 
Parameters_WorkSheet.write('A7', "Gray FloodFill: ") 
Parameters_WorkSheet.write('A8', "Gray FloodFill Increase Num: ") 
Parameters_WorkSheet.write('A9', "FloodFill Area Components MaxMin Idx 0: ") 
Parameters_WorkSheet.write('A10', "FloodFill Area Components MaxMin Idx 1: ") 
Parameters_WorkSheet.write('A11', "Fruit Contours Sorting MaxMin Idx: ") 
Parameters_Excel_File_List = []
Parameters_Excel_File_List.append(str(Blur_KernelSize))
Parameters_Excel_File_List.append(str(inRange1))
Parameters_Excel_File_List.append(str(inRange2))
Parameters_Excel_File_List.append(str(ThresholdCanny1))
Parameters_Excel_File_List.append(str(ThresholdCanny2))
Parameters_Excel_File_List.append(str(Dilate_KernelSize))
Parameters_Excel_File_List.append(str(Gray_FloodFill_Ini))
Parameters_Excel_File_List.append(str(Gray_FloodFill_Increase_Step))
Parameters_Excel_File_List.append(str(FloodFill_Area_Components_MaxMin_Idx0))
Parameters_Excel_File_List.append(str(FloodFill_Area_Components_MaxMin_Idx1))
Parameters_Excel_File_List.append(str(Fruit_Contours_Sorting_MaxMin_Idx))
# print(Parameters_Excel_File_List)
Parameters_Row = 0
Parameters_Column = 1
for item in Parameters_Excel_File_List:
    # write operation perform 
    Parameters_WorkSheet.write(Parameters_Row, Parameters_Column, item) 
    # incrementing the value of row by one with each iteratons. 
    Parameters_Row += 1 

# create worksheet for writing measurements
Measurements_WorkSheet = WorkBook.add_worksheet("Measurements")
Measurements_WorkSheet.write('B1', "Time")
Measurements_WorkSheet.write('C1', "Area")
Measurements_WorkSheet.write('D1', "Contour_Area")
Measurements_WorkSheet.write('E1', "Perimeter")
Measurements_WorkSheet.write('F1', "Width")
Measurements_WorkSheet.write('G1', "Height")
Measurements_WorkSheet.write('H1', "FloodFill_Area_Components_MaxMin")
Measurements_WorkSheet.write('I1', "Contours_Sorting_MaxMin")


startTimeSum = time.time()
multiThread(Files_Name_List)
print("[INFOR]: Sum time multi threading: {}".format(time.time() - startTimeSum))
print("----------------------------------DONE---------------------------------")


# startTimeSum = time.time()
# for filename in Files_Name_List:
#     print('File Name:\t{}\t{}/{}'.format(filename, File_Name_Idx + 1, len(Files_Name_List)))
#     fruit_feature_extraction(filename)
# print("[INFOR]: Sum time: {}".format(time.time() - startTimeSum))
# print("----------------------------------DONE---------------------------------")


# close excel file
WorkBook.close()
