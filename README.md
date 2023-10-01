# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.

2. Set variables for assigning dataset values.

3. Import linear regression from sklearn.

4. Predict the values of array.

5. Calculate the accuracy, confusion and classification report b importing the required modules from sklearn.
Obtain the graph. 

## Program:
```

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: YUVARAJ B
RegisterNumber: 212222230182


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
  
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j

def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x,X)==y) 

```

## Output:

# Array value of x:

![270584113-42045068-a2c4-42e4-8913-fcde2788bdbc](https://github.com/Yuva2005raj/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343998/35f9a872-3017-4f8b-819e-55bb4d5efb4f)

# Array value of y:

![270584200-aff41d26-9c3d-4107-8531-8420da9fecae](https://github.com/Yuva2005raj/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343998/529ac4c0-b204-4970-aff6-b21a71449151)

# Exam 1 score graph:

![270584322-03720f54-b391-4090-b488-3f5a7eca6fee](https://github.com/Yuva2005raj/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343998/4f81e38b-86df-42c6-b340-5ce994c2824f)

# Sigmoid function group:

![270584434-4fa5f881-e348-410b-8d08-98b5c2bae583](https://github.com/Yuva2005raj/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343998/3dc648a4-9df8-4888-9d89-e1c7c1fbcc46)

# X_Train_ grade value:

![270584586-da4e301d-0a1d-46d9-b731-f9c9f2dee214](https://github.com/Yuva2005raj/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343998/7b92d85d-eacc-40e9-b85c-f1dc1991f10a)

# Y_Train_Grade value:

![270584731-061c8864-4e34-4ceb-aced-f287af142435](https://github.com/Yuva2005raj/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343998/42e043e9-9bc2-4da1-af80-6d4b9d95bdf5)

# print rex.x:

![270584731-061c8864-4e34-4ceb-aced-f287af142435](https://github.com/Yuva2005raj/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343998/febc9983-adf8-4f3e-b895-f60e8aa833a9)

# Decision boundary graph for exam score:

![270586288-df276009-64d1-4648-8142-18ac2e72fe08](https://github.com/Yuva2005raj/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343998/2539b632-2662-4d20-a2c8-4084fe214257)

# probability value:

![270586404-d28eff8d-4445-4866-ac9e-78d3449d63ae](https://github.com/Yuva2005raj/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343998/a991d8fb-fdce-42d8-bf23-305938d98277)

# Prediction value of mean:

![270587090-5ccd966f-e001-43fe-b6ab-7c25fc66934a](https://github.com/Yuva2005raj/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343998/dc5abf8e-d317-417b-ab93-3c5cd038ef0e)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

