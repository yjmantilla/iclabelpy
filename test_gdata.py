from iclabelpy.utils import mergesimpts,gdatav4,mergepoints2D, multimergesimpts
from iclabelpy.mat import loadmat
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    gdataIN =loadmat(r'Y:\code\iclabelpy\data\gdataIN.mat')
    inty = gdataIN['varargin'][0]
    intx = gdataIN['varargin'][1]
    intValues = gdataIN['varargin'][2]
    yi = gdataIN['varargin'][3]
    xi = gdataIN['varargin'][4]
    gdataOUT =loadmat(r'Y:\code\iclabelpy\data\gdataOUT.mat')
    Xi = gdataOUT['sa']['Xi']
    Yi = gdataOUT['sa']['Yi']
    Zi = gdataOUT['sa']['Zi']

    xq,yq,vq = gdatav4(inty,intx,intValues,yi,xi)
    # x,y,v = mergepoints2D(data['x'],data['y'],data['v'])
    plt.imshow(vq)
    plt.colorbar()
    plt.show()
    # x = [[1,2], [1.06,2], [1.1, 2], [1.1, 2.03]]
    # multimergesimpts(x,[0.05,0.1],'','')

    print('ok')