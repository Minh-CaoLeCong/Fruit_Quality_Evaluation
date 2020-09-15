import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import mglearn
import sklearn
from sklearn.model_selection import train_test_split
from mpl_toolkits import mplot3d

print("""
[INFOR]: -------- LINEAR REGRESSION VISUALIZING - 3D --------
""")

InputFile_Path = input("[ENTER]: Input file path (../image_result/apple_golden/Apple_Golden_Weight_AreaPixels_1_4.txt): ")
OuputFolder_Path = input("[ENTER]: Ouput folder path (../image_result/apple_golden/linear_regression/): ")

FileName_Result = input("[ENTER]: File name result: ")

X_name = input("[ENTER]: Name of x coordinate: ")
Y_name = input("[ENTER]: Name of y coordinate: ")
Z_name = input("[ENTER]: Name of z coordinate: ")

Show_Plot_Check_str = input("[ENTER]: Show plot or not? y/n?: ")
if Show_Plot_Check_str == "y" or Show_Plot_Check_str == "Y" or\
    Show_Plot_Check_str == "yes" or Show_Plot_Check_str == "Yes":
    Show_Plot_Check_bool = True
    print("[INFOR]: Show plot: {}".format(Show_Plot_Check_bool))
else:
    Show_Plot_Check_bool = False
    print("[INFOR]: Show plot: {}".format(Show_Plot_Check_bool))


print("-------------Start-processing-------------")

file1 = open(InputFile_Path, "r")
file2 = open((OuputFolder_Path + FileName_Result + ".txt"), "w")

if file1.mode == 'r':
    lines = file1.readlines()
    X_Data_list = [[0.0] * 2 for i in range(len(lines))]
    y_Data_list = [0.0] * len(lines)
    for num, line in enumerate(lines):
        X_Data_list[num][0] = float(line.split()[0])
        X_Data_list[num][1] = float(line.split()[1])
        y_Data_list[num] = float(line.split()[2])

X_Data_np = np.array(X_Data_list)
y_Data_np = np.array(y_Data_list)


X1_Data_np, X2_Data_np = np.hsplit(X_Data_np,2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X1_Data_np, X2_Data_np, y_Data_np, c='g', marker='o')

ax.set_xlabel(X_name)
ax.set_ylabel(Y_name)
ax.set_zlabel(Z_name)


X_Data_Train_np, X_Data_Test_np, y_Data_Train_np, y_Data_Test_np =\
     train_test_split(X_Data_np, y_Data_np, random_state=42)

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


# y_pred = LinearRegression_Model1.predict(X_Data_np)


plt.savefig(OuputFolder_Path + FileName_Result + ".png")
plt.savefig(OuputFolder_Path + FileName_Result + ".pdf")

if Show_Plot_Check_bool:
    plt.show()

file1.close()

print("---------------Done---------------")









