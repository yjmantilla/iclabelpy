from iclabelpy.utils import mergesimpts,gdatav4,mergepoints2D, multimergesimpts
from iclabelpy.mat import loadmat
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    data = loadmat(r'Y:\code\iclabelpy\data\values.mat')
    locfile = loadmat(r'Y:\code\iclabelpy\data\locfile.mat')
    plotrad = loadmat(r'Y:\code\iclabelpy\data\plotrad.mat')
    temptopo = loadmat(r'Y:\code\iclabelpy\data\temptopo.mat')
    labels = loadmat(r'Y:\code\iclabelpy\data\labels.mat')
    tmpeloc = loadmat(r'Y:\code\iclabelpy\data\tmpeloc.mat')
    th = loadmat(r'Y:\code\iclabelpy\data\Th.mat')
    rd = loadmat(r'Y:\code\iclabelpy\data\Rd.mat')
    indices = loadmat(r'Y:\code\iclabelpy\data\indices.mat')
    gdataIN =loadmat(r'Y:\code\iclabelpy\data\gdataIN.mat')
    inty = gdataIN['varargin'][0]
    intx = gdataIN['varargin'][1]
    intValues = gdataIN['varargin'][2]
    yi = gdataIN['varargin'][3]
    x   i = gdataIN['varargin'][4]
    vq = gdatav4(inty,intx,intValues,yi,xi)
    # x,y,v = mergepoints2D(data['x'],data['y'],data['v'])
    # plt.imshow(vq)
    # plt.colorbar()
    # plt.show()
    # x = [[1,2], [1.06,2], [1.1, 2], [1.1, 2.03]]
    # multimergesimpts(x,[0.05,0.1],'','')

    print('ok')