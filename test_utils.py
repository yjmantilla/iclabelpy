from iclabelpy.utils import mergesimpts,gdatav4,mergepoints2D, multimergesimpts
from iclabelpy.mat import loadmat
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    x = [1, 1.1, 1.05]
    mergesimpts(x,0.1,'unique','')
    # x = list(np.arange(0,10,0.5))
    # mergesimpts(x,1,mode='')
    # x=[-0.147921891353479,0.147921891353479,-0.122450697428586,0.122721402649397,-0.245187423015011,0.245166992432308,-0.367658551026300,0.367658551026300,-0.117632487188521,0.117632487188521,-0.234883896620009,0.234883896620009,-0.351164911943686,0.351164911943686,0,-0.102262955834933,0.102173367580941,-0.203418205392725,0.203150481352913,-0.301299618644754,0.301082429428487,-0.392349849052477,0.392107755545381,-0.117632487188521,0.117632487188521,-0.234883896620009,0.234883896620009,-0.351164911943686,0.351164911943686,-0.145728126853333,0.145728126853333,0,-0.145728126853333,0.145728126853333,5.70486668071449*1e-17,-0.102262955834933,0.102173367580941,-0.203418205392725,0.203150481352913,-0.301299618644754,0.301082429428487,-0.392349849052477,0.392107755545381,-0.147921891353479,0.147921891353479,-0.281533912238001,0.281291034975421,4.27783685248524*1e-17,2.85280864407292*1e-17,-0.490196078431373,0.490196078431373,-0.464959285410511,0.464803892788837]
    # a = mergesimpts(x,3.814697265625003*1e-6,'')
    # data = loadmat('Y:\code\iclabelpy\data\gdatav4.mat')
    # data2 = loadmat('Y:\code\iclabelpy\data\mergesimpoints.mat')
    # vq = gdatav4(data2['x'],data2['y'],data2['v'],data['xq'],data['yq'])
    # x,y,v = mergepoints2D(data['x'],data['y'],data['v'])
    # plt.imshow(vq)
    # plt.colorbar()
    # plt.show()
    x = [[1,2], [1.06,2], [1.1, 2], [1.1, 2.03]]
    multimergesimpts(x,[0.05,0.1],'','')

    print('ok')