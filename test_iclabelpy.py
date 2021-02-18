import numpy as np
from scipy.interpolate import griddata,Rbf
import matplotlib.pyplot as plt
import copy
from iclabelpy.utils import mergesimpts,gdatav4
def reref(input):
    data = np.copy(input)
    nchannels = input.shape[0]
    nframes = input.shape[1]
    if len(input.shape) == 3:
        nepochs = input.shape[2]
    # Rereference to average
        data = np.reshape(data,(nchannels,nframes*nepochs),order='F')
    refmatrix = np.eye(nchannels)-np.ones(nchannels)*1/nchannels
    data = refmatrix @ data
    if len(input.shape) == 3:
        data = np.reshape(data,input.shape,order='F')
    return data


def cart2sph(x,y,z):
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    # theta,phi,r
    return azimuth, elevation, r

def sph2cart(azimuth,elevation,r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def sph2topo(theta,phi):
    az = phi
    horiz = theta
    angle  = -1*horiz
    radius = 0.5 - az
    return angle,radius

# def gdatav4(x,y,v,xq,yq):
# """
# %GDATAV4 MATLAB 4 GRIDDATA interpolation

# %   Reference:  David T. Sandwell, Biharmonic spline
# %   interpolation of GEOS-3 and SEASAT altimeter
# %   data, Geophysical Research Letters, 2, 139-142,
# %   1987.  Describes interpolation using value or
# %   gradient of value in any dimension.
# """
# [x, y, v] = mergepoints2D(x,y,v);

# xy = x(:) + 1i*y(:);

# % Determine distances between points
# d = abs(xy - xy.');

# % Determine weights for interpolation
# g = (d.^2) .* (log(d)-1);   % Green's function.
# % Fixup value of Green's function along diagonal
# g(1:size(d,1)+1:end) = 0;
# weights = g \ v(:);

# [m,n] = size(xq);
# vq = zeros(size(xq));
# xy = xy.';

# % Evaluate at requested points (xq,yq).  Loop to save memory.
# for i=1:m
#     for j=1:n
#         d = abs(xq(i,j) + 1i*yq(i,j) - xy);
#         g = (d.^2) .* (log(d)-1);   % Green's function.
#         % Value of Green's function at zero
#         g(d==0) = 0;
#         vq(i,j) = g * weights;        
#     end
# end
# return vq

if __name__ == '__main__':
    import mne
    import numpy as np
    import scipy.io as sio
    from iclabelpy.mat import loadmat

    data = loadmat('Y:\code\iclabelpy\data\sub-010317_PREP_clean_ICA.mat')
    print('ok')

    flag_autocorr = True
    EEG = data['EEG']
    icawinv = EEG['icawinv']

    #ICL_feature_extractor(EEG, flag_autocorr)
    ncomp = icawinv.shape[1]
    nchannels = icawinv.shape[0]
    average_reference = False
    if not average_reference:
        data = reref(EEG['data'])
        # Rereference ICA Matrix
        newICAchaninds = reref(icawinv)
    icaact = EEG['icaact']
    assert np.all(np.isreal(icaact)) == True # Check Activations are real

    ### TOPOPLOTFAST
    GRID_SCALE = 32
    INTERPLIMITS = 'head';
    rmax = 0.5

    raw = mne.read_epochs_eeglab('Y:\code\iclabelpy\data\sub-010317_PREP_clean.set')
    locs = raw._get_channel_positions()
    # see https://github.com/mne-tools/mne-python/blob/24377ad3200b6099ed47576e9cf8b27578d571ef/mne/io/eeglab/eeglab.py#L105
    
    X = locs[:,1]
    Y = -1*locs[:,0]
    Z = locs[:,2]
    sph = np.array([cart2sph(X[i],Y[i],Z[i]) for i in range(len(X))])
    theta = sph[:,0]
    phi = sph[:,1]
    topo = np.array([sph2topo(theta[i],phi[i]) for i in range(len(theta))])
    Rd = topo[:,1]
    Th = topo[:,0]
    allchansind = np.array(list(range(len(Th))))

    plotchans = allchansind.copy()
    for comp in range(ncomp):
        Values = icawinv[:,comp]
        ContourVals = Values
        if Values.shape[0] > 1:
            inds          = set(np.where(np.logical_not(np.isfinite(Values)))[0].tolist()) #% NaN and Inf values
            plotchans     = np.array(list(set(plotchans.tolist())- inds))
        xy = np.array([pol2cart(Rd[i],Th[i]) for i in range(len(X))])
        x = xy[:,0]
        y = xy[:,1]
        #plotchans = abs(plotchans);   % reverse indicated channel polarities
        allchansind = allchansind[plotchans];
        Th        = Th[plotchans];
        Rd        = Rd[plotchans];
        x         = x[plotchans];
        y         = y[plotchans];
        Values      = Values[plotchans];
        ContourVals = ContourVals[plotchans];

        plotrad = np.min([1.0,np.max(Rd)*1.02])#default: just outside the outermost electrode location
        plotrad = np.max([plotrad,0.5])# default: plot out to the 0.5 head boundary
        intrad = np.min([1.0,np.max(Rd)*1.02])

        if plotrad >= rmax:
            headrad = rmax #(anatomically correct)
        else: #% if plotrad < rmax
            headrad = 0 #    % don't plot head

        
        pltchans = np.where(Rd < plotrad)[0]
        intchans = np.where(np.logical_and(xy[:,0] <= intrad,xy[:,1] <= intrad))[0]
        allx = x.copy()
        ally = y.copy()
        if len(Values) == len(Th):#   if as many map Values as channel locs
            intValues      = Values[intchans]
            intContourVals = ContourVals[intchans]
            Values         = Values[pltchans]
            ContourVals    = ContourVals[pltchans]

        allchansind = allchansind[pltchans];
        intTh = Th[intchans];           #% eliminate channels outside the interpolation area
        intRd = Rd[intchans];
        intx  = x[intchans];
        inty  = y[intchans];
        Th    = Th[pltchans];              #% eliminate channels outside the plotting area
        Rd    = Rd[pltchans];
        x     = x[pltchans];
        y     = y[pltchans];

        squeezefac = rmax/plotrad;
        intRd = intRd*squeezefac; #% squeeze electrode arc_lengths towards the vertex
        Rd = Rd*squeezefac;       #% squeeze electrode arc_lengths towards the vertex
                                #% to plot all inside the head cartoon
        intx = intx*squeezefac;   
        inty = inty*squeezefac;  
        x    = x*squeezefac;    
        y    = y*squeezefac;   
        allx    = allx*squeezefac;    
        ally    = ally*squeezefac;   

        xmin = np.min([-1*rmax,np.min(intx)]); xmax = np.max([rmax,np.max(intx)]);
        ymin = np.min([-1*rmax,np.min(inty)]); ymax = np.max([rmax,np.max(inty)]);
        xi = np.linspace(xmin,xmax,GRID_SCALE);  # % x-axis description (row vector)
        yi = np.linspace(ymin,ymax,GRID_SCALE);  # % y-axis description (row vector)
        inty = np.expand_dims(inty,axis=-1)
        intx = np.expand_dims(intx,axis=-1)
        # xi = np.expand_dims(xi,axis=-1)
        # yi = np.expand_dims(yi,axis=-1)
        xv, yv = np.meshgrid(xi, yi, sparse=False, indexing='ij')
        #xyV = griddata(np.hstack([inty,intx]),intValues,(xv,yv))
        rbf = Rbf(intx, inty, intValues, epsilon=2,smooth=1)
        ZI = rbf(xv, yv)
        ZI2 = gdatav4(intx,inty,intValues,xv,yv)
        plt.imshow(ZI)
        plt.colorbar()
        plt.show()
        plt.imshow(ZI2)
        plt.colorbar()
        plt.show()
        print('next')

    #topo = np.array([sph2topo(sph[0,i])])
    
    print('ok')
