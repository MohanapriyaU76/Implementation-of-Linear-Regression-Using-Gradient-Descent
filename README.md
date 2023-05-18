# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe. 
2. Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Mohanapriya U
RegisterNumber: 212220040091 
*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('ex1.txt',header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("population of city (10,000s)")
plt.ylabel("profit ($10,000")
plt.title("profit prediction")
def compute(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
compute(x,y,theta)
def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  j_history=[]
  for i in range(num_iters):
    predictions=x.dot(theta)
    error=np.dot(x.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    j_history.append(compute(x,y,theta))
  return theta,j_history
theta,j_history=gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")
plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$j(\Theta)$")
plt.title("cost function using Gradient Descent")
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("population of city (10,000s)")
plt.ylabel("profit ($10,000")
plt.title("profit prediction")
def  predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))
predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))

## Output:
![linear regression using gradient descent](sam.png)

1.Profit Prediction graph

![image](https://github.com/MohanapriyaU76/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/133958624/631e652e-3720-4d2a-a706-ba171afc676d)

2.Compute Cost Value

![image](https://github.com/MohanapriyaU76/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/133958624/c78d14d3-c8fb-4ece-8004-589553ae9c46)

3.h(X) Value

![image](https://github.com/MohanapriyaU76/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/133958624/fa7a7193-9d2f-4581-b83c-adc10aaaee4c)

4.Cost function using Gradient Descent Graph

![image](https://github.com/MohanapriyaU76/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/133958624/0c810b99-10ec-450c-8091-c1e6b86e07ba)

5.Profit Prediction graph

![image](https://github.com/MohanapriyaU76/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/133958624/97825406-d457-4a8f-b675-7ef1470a209c)

6.Profit for the Population 35,000

![image](https://github.com/MohanapriyaU76/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/133958624/5df0c759-deeb-4a2c-a5b1-9f78526610b1)

7.Profit for the Population 70,000

![image](https://github.com/MohanapriyaU76/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/133958624/01f4fb6e-dd5e-4e55-a5e1-f5ae4b43d2a5)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
