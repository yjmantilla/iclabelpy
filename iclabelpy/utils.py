import numpy as np
import copy
from iclabelpy.mat import loadmat
def mergesimpts(x,atol,mode='',ref='mean'):
    Xs = []
    Is = []
    if isinstance(x,np.ndarray):
        x = x.copy().flatten()
    clusters = []
    for i,xi in enumerate(np.sort(x)):
        if Xs == []:
            Xs.append(xi)
            Is.append(i)
        else:
            m = np.mean(Xs+[xi])
            if ref == 'mean':
                make_cluster = [np.isclose(m,xii,atol=atol,rtol=0) for xii in Xs+[xi]]
            else:
                make_cluster = [np.abs(Xs[0]-xii)<atol for xii in Xs+[xi]]
            if np.all(make_cluster):
                Xs.append(xi)
                Is.append(i)
                if i == len(x)-1:
                    clusters.append((np.mean(Xs),copy.deepcopy(Is)))
            else:
                clusters.append((np.mean(Xs),copy.deepcopy(Is)))
                Xs = [xi]
                Is = [i]
                if i == len(x)-1:
                    clusters.append((xi,[i]))

    result = []
    if mode == 'unique':
        for cl in clusters:
            result.append(cl[0])
    else:
        for cl in clusters:
            for _ in range(len(cl[1])):
                result.append(cl[0])
    return np.array(result),np.argsort(x)

def multimergesimpts(x,atol,mode='',ref='mean'):
    Xs = []
    Is = []
    if isinstance(x,np.ndarray):
        x = x.copy()
    else:
        x = copy.deepcopy(x)
        x = np.array(x)
    clusters = []
    for i in enumerate(np.argsort(x[:,0].flatten)):
        if Xs == []:
            Xs.append(x[i,:])
            Is.append(i)
        else:
            m = np.mean(np.array(Xs),axis=-1)
            if ref == 'mean':
                make_cluster = [np.isclose(m,xii,atol=atol,rtol=0) for xii in Xs+[xi]]
            else:
                make_cluster = [np.abs(Xs[0]-xii)<atol for xii in Xs+[xi]]
            if np.all(make_cluster):
                Xs.append(xi)
                Is.append(i)
                if i == len(x)-1:
                    clusters.append((np.mean(Xs),copy.deepcopy(Is)))
            else:
                clusters.append((np.mean(Xs),copy.deepcopy(Is)))
                Xs = [xi]
                Is = [i]
                if i == len(x)-1:
                    clusters.append((xi,[i]))

    result = []
    if mode == 'unique':
        for cl in clusters:
            result.append(cl[0])
    else:
        for cl in clusters:
            for _ in range(len(cl[1])):
                result.append(cl[0])
    return np.array(result),np.argsort(x)

def mergepoints2D(x,y,v):
    # Sort x and y so duplicate points can be averaged

    # Need x,y and z to be column vectors
    sz = x.size
    x = x.copy()
    y = y.copy()
    v = v.copy()
    x = np.reshape(x,(sz,1),order='F');
    y = np.reshape(y,(sz,1),order='F');
    v = np.reshape(v,(sz,1),order='F');
    myepsx = np.spacing(0.5 * (np.max(x) - np.min(x)))**(1/3);
    myepsy = np.spacing(0.5 * (np.max(y) - np.min(y)))**(1/3);
    # % look for x, y points that are indentical (within a tolerance)
    # % average out the values for these points
    if np.all(np.isreal(v)):
        x1,x_i = mergesimpts(x,myepsx,'')
        y1,y_i = mergesimpts(y,myepsy,'')
        v1,v_i = mergesimpts(v,np.inf,'')
        x2 =[]
        y2 =[]
        v2 =[]
        for i in range(len(x1)):
            x2.append(x1[np.where(x_i==i)])
            y2.append(y1[np.where(y_i==i)])
            v2.append(v1[np.where(v_i==i)])
        x2 = np.array(x2)
        y2 = np.array(y2)
        v2 = np.array(v2)
    else:
        #% if z is imaginary split out the real and imaginary parts
        x1,x_i = mergesimpts(x,myepsx,'')
        y1,y_i = mergesimpts(y,myepsy,'')
        realv1,realv_i = mergesimpts(np.real(v),np.inf,'')
        imagv1,imagv_i = mergesimpts(np.imag(v),np.inf,'')
        x2=[]
        y2=[]
        v2=[]
        for i in range(len(x1)):
            x2.append(x1[np.where(x_i==i)])
            y2.append(y1[np.where(y_i==i)])
            v2.append(realv1[np.where(realv_i==i)]+1j*imagv1[np.where(imagv_i==i)])
        x2 = np.array(x2)
        y2 = np.array(y2)
        v2 = np.array(v2)
        #% re-combine the real and imaginary parts
        #v = realv + 1j*imagv;
    #% give a warning if some of the points were duplicates (and averaged out)
    # if sz>numel(x):
    #     warning(message('MATLAB:griddata:DuplicateDataPoints'));
    # end
    return x2,y2,v2

def gdatav4(x,y,v,xq,yq):
    """
    %GDATAV4 MATLAB 4 GRIDDATA interpolation

    %   Reference:  David T. Sandwell, Biharmonic spline
    %   interpolation of GEOS-3 and SEASAT altimeter
    %   data, Geophysical Research Letters, 2, 139-142,
    %   1987.  Describes interpolation using value or
    %   gradient of value in any dimension.
    """
    #x, y, v = mergepoints2D(x,y,v);

    xy = x + 1j*y
    xy = np.squeeze(xy)
    #% Determine distances between points
    d = np.zeros((xy.shape[0],xy.shape[0]))
    for i in range(xy.shape[0]):
        for j in range(xy.shape[0]):
            d[i,j]=np.abs(xy[i]-xy[j])


    # % Determine weights for interpolation
    g = np.square(d) * (np.log(d)-1) #% Green's function.
    # % Fixup value of Green's function along diagonal
    np.fill_diagonal(g, 0)
    weights = np.linalg.lstsq(g, v)[0]

    (m,n) = xq.shape
    vq = np.zeros(xq.shape);
    #xy = np.tranpose(xy);

    # % Evaluate at requested points (xq,yq).  Loop to save memory.
    for i in range(m):
        for j in range(n):
            d = np.abs(xq[i,j] + 1j*yq[i,j] - xy);
            g = np.square(d) * (np.log(d)-1);#   % Green's function.
            #% Value of Green's function at zero
            g[np.where(d)==0] = 0;
            vq[i,j] = (np.expand_dims(g,axis=0) @ np.expand_dims(weights,axis=1))[0][0]
    return vq
