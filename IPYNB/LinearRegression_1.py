import numpy as np

class LinearRegression:
    def __init__(self):
        self.w = 0
        self.b = 0

    def fit(self,x,y):
        self.w = (y*(x-x.mean())).sum() / ((x**2).sum() - ((x.sum())**2)/x.size)
        self.b = (y-self.w*x).mean()

    def predict(self,x):
        return(self.w*x + self.b)

    def score(self,x,y):
        predict_y = self.w*x + self.b
        E = ((y-predict_y)**2).sum()
        R2 = 1 - E/(((y-y.mean())**2).sum())
        return(R2)

if __name__ == '__main__':
    x = np.array([1,2,3,4,5,6,7,8,9,10])
    y = 1.45*x - 2.5

    LR = LinearRegression()
    LR.fit(x,y)

    print("\n w* = ",LR.w,",  b* = ",LR.b)
    
    yy = LR.predict(x)

    print("\ny:\n",y)
    print("Predict y:\n",yy,"\n")
    print(LR.score(x,y))
