import numpy as np
from sklearn.linear_model import LinearRegression

x1 = np.array([[1.1,2.2,3.1,4.5,5.4,6.2, \
                7.8,8.4,9.5,10]]).T
x2 = np.array([[10,9,8,7,6,5,4,3,2,1]]).T
y = 1.45*x1 - 2.31*x2 - 2.5 \
    + np.random.normal(0,0.1,(10,1))

X = np.append(x1,x2,axis=1)

LR = LinearRegression()
LR.fit(X,y)

print("\n w* = ",LR.coef_,"\nb* = ",LR.intercept_)

yy = LR.predict(X)

print("\n      y          Predict_y\n",
    np.append(y,yy,axis=1) )
print("\n R2 = ", LR.score(X,y))
