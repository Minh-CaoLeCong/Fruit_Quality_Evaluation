import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import mglearn
import sklearn
from sklearn.model_selection import train_test_split

print("""
[INFOR]: ----- LINEAR REGRESSION VISUALIZING -----
""")

def extractDigits(lst):
    return [[el] for el in lst]

InputFile_Path = input("[ENTER]: Input file path (../image_result/mango/Weight_AreaPixels_A.txt): ")
OuputFolder_Path = input("[ENTER]: Ouput folder path (../image_result/mango/): ")
FileName_Result = input("[ENTER]: File name result: ")
X_name = input("[ENTER]: Name of x coordinate: ")
y_name = input("[ENTER]: Name of y coordinate: ")

print("----------Start-processing----------")

file1 = open(InputFile_Path, "r")
file2 = open((OuputFolder_Path + FileName_Result + ".txt"), "w")

X_Data_list = []
y_Data_list = []

if file1.mode == 'r':
    lines = file1.readlines()
    for i in lines:
        X_Data_list.append(float(i.split()[0]))
        y_Data_list.append(float(i.split()[1]))

X_Data_np = extractDigits(X_Data_list)
X_Data_np = np.array(X_Data_np)

y_Data_np = np.array(y_Data_list)

plt.scatter(X_Data_np, y_Data_np, color='green')
plt.xlabel(X_name)
plt.ylabel(y_name)

X_Data_Train_np, X_Data_Test_np, y_Data_Train_np, y_Data_Test_np = train_test_split(X_Data_np, y_Data_np, random_state=42)

LinearRegression_Model1 = LinearRegression().fit(X_Data_Train_np, y_Data_Train_np)

coef = LinearRegression_Model1.coef_
intercept = LinearRegression_Model1.intercept_
print("[INFOR]: Coefficient: {}".format(coef))
print("[INFOR]: Intercept: {}".format(intercept))
file2.write("[INFOR]: Coefficient: " + str(coef) + "\n")
file2.write("[INFOR]: Intercept: " + str(intercept) + "\n")

training_score = LinearRegression_Model1.score(X_Data_Train_np, y_Data_Train_np)
test_score = LinearRegression_Model1.score(X_Data_Test_np, y_Data_Test_np)
print("[INFOR]: Training set score: {:.2f}".format(training_score))
print("[INFOR]: Test set score: {:.2f}".format(test_score))
file2.write("[INFOR]: Training set score: " + str(training_score) + "\n")
file2.write("[INFOR]: Test set score: " + str(test_score) + "\n")

y_pred = LinearRegression_Model1.predict(X_Data_np)

plt.plot(X_Data_np, y_pred, color="red")

plt.savefig(OuputFolder_Path + FileName_Result + ".png")
plt.savefig(OuputFolder_Path + FileName_Result + ".pdf")
# plt.show()
file1.close()
print("---------------Done---------------")









