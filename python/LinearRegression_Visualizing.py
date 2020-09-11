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

# InputFile_Path = input("[ENTER]: Input file path (../image_result/mango/Weight_AreaPixels_A.txt): ")
# X_name = input("[ENTER]: Name of x coordinate: ")
# y_name = input("[ENTER]: Name of y coordinate: ")
InputFile_Path = "../image_result/mango/Weight_AreaPixels_A.txt"
X_name = "Number of pixels"
y_name = "Weight"

file1 = open(InputFile_Path, "r")

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
print("[INFOR] Coefficient: {}".format(coef))
print("[INFOR] Intercept: {}".format(intercept))

print("Training set score: {:.2f}".format(LinearRegression_Model1.score(X_Data_Train_np, y_Data_Train_np)))
print("Test set score: {:.2f}".format(LinearRegression_Model1.score(X_Data_Test_np, y_Data_Test_np)))

y_pred = LinearRegression_Model1.predict(X_Data_np)

plt.plot(X_Data_np, y_pred, color="red")

plt.show()
file1.close() 








