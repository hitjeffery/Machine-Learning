import numpy as np

class LinearRegression:
    def __init__(self):
        self.w = 0
        self.b = 0

    def fit(self,X,y):
        (m,d) = X.shape
        EX = np.append(X,np.ones([m,1]),axis=1)
        pinvX = np.matmul( np.linalg.inv(np.matmul(EX.T,EX)), EX.T )
        
        wHat = np.matmul(pinvX,y)
        self.w = wHat[0:-1]
        self.b = wHat[d-1]

    def predict(self,X):
        return(np.matmul(X,self.w) + self.b)

    def score(self,X,y):
        predict_y = np.matmul(X,self.w) + self.b
        E = ((y-predict_y)**2).sum()
        R2 = 1 - E/(((y-y.mean())**2).sum())
        return(R2)

if __name__ == '__main__':
    x1 = np.array([[1.1,2.2,3.1,4.5,5.4,6.2, 
                    7.8,8.4,9.5,10]]).T
    x2 = np.array([[10,9,8,7,6,5,4,3,2,1]]).T
    y = 1.45*x1 - 2.31*x2 - 2.5 + np.random.normal(0,0.1,(10,1))
    
    X = np.append(x1,x2,axis=1)

    LR = LinearRegression()
    LR.fit(X,y)

    print("\n w* = ",LR.w,"\nb* = ",LR.b)
    
    yy = LR.predict(X)

    print("\n      y          Predict_y\n",
        np.append(y,yy,axis=1) )
    print("\n R2 = ", LR.score(X,y))
