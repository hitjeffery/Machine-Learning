import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
    
def plot_decision_boundary(model,axis,ax=None):
    '''axis是x轴y轴对应的范围'''
    x0,x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape(-1,1),
        np.linspace(axis[2], axis[3], int((axis[3]-axis[2])*100)).reshape(-1,1)
    )
    x_new=np.c_[x0.ravel(),x1.ravel()]
    y_predict=model.predict(x_new)
    zz = y_predict.reshape(x0.shape)
    
    custom_cmap = ListedColormap(['peachpuff','lavender','lightyellow'])
    
    if ax is None:
        ax = plt.gca()

    ax.contourf(x0, x1, zz, cmap=custom_cmap)
