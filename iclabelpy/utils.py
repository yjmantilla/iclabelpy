import numpy as np
import copy
from iclabelpy.mat import loadmat
from scipy.interpolate import griddata,Rbf
import matplotlib.pyplot as plt
import copy
from gdatav4 import gdatav4
import fractions
import scipy.signal as sx

def reref(input):
    """
    assume nchannels,nframes,nepochs
    """
    data = np.copy(input)
    if input.ndim == 3:
        nchannels = input.shape[0]
        nframes = input.shape[1]
        nepochs = input.shape[2]
    else:
        nchannels = input.shape[0]
        nframes = input.shape[1]
        nepochs = 1

    # Rereference to average
    data = np.reshape(data,(nchannels,nframes*nepochs),order='F')
    refmatrix = np.eye(nchannels)-np.ones(nchannels)*1/nchannels
    data = refmatrix @ data
    if input.ndim == 3:
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
    radius = (np.pi/2 - az)/np.pi
    return angle,radius

def topoplotFast(locs,values,plot=False):
    ### TOPOPLOTFAST
    GRID_SCALE = 32
    INTERPLIMITS = 'head';
    rmax = 0.5

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
    Values = values
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
    #rbf = Rbf(intx, inty, intValues, epsilon=2,smooth=1)
    #ZI = rbf(xv, yv)
    XI,YI,ZI = gdatav4(intx,inty,intValues,xv,yv)
    mask = np.where(np.sqrt(np.square(XI) + np.square(YI)) > rmax); #% mask outside the plotting circle
    ZI[mask] = np.nan;                         #% mask non-plotting voxels with NaNs  
    grid = plotrad;                       #% unless 'noplot', then 3rd output arg is plotrad

    if plot:
        plt.imshow(ZI)
        plt.colorbar()
        plt.show()
    return ZI,plotrad

def eeg_rpsd(icaact,srate,demixing,nfreqs=None, pct_data=None,test=False):
    """
    assume channels,frames,epochs
    """
    if icaact.ndim == 3:
        nchannels = icaact.shape[0]
        nframes = icaact.shape[1]
        nepochs = icaact.shape[2]
    else:
        nchannels = icaact.shape[0]
        nframes = icaact.shape[1]
        nepochs = 1

    # % clean input cutoff freq
    nyquist = int(np.floor(srate / 2).astype(int));
    if nfreqs is None:
        nfreqs = nyquist;
    elif nfreqs > nyquist:
        nfreqs = nyquist;

    if pct_data is None:
        pct_data = 100;

    # % setup constants
    ncomp = demixing.shape[0];
    n_points = np.min([nframes, srate]).astype(int);
    window = np.hamming(n_points);
    cutoff = np.floor(nframes / n_points) * n_points;
    index = np.array([x+ np.arange(0,n_points) for x in np.arange(0,cutoff - n_points+1,n_points / 2)]).astype(int).T
    n_seg = index.shape[1] * nepochs;

    rng = np.random.RandomState(seed=0)
    seq=[5489 ,1301868182 ,2938499221 ,2950281878 ,1875628136 ,751856242 ,944701696 ,2243192071 ,694061057 ,219885934 ,2066767472 ,3182869408 ,485472502 ,2336857883 ,1071588843 ,3418470598 ,951210697 ,3693558366 ,2923482051 ,1793174584 ,2982310801 ,1586906132 ,1951078751 ,1808158765 ,1733897588 ,431328322 ,4202539044 ,530658942 ,1714810322 ,3025256284 ,3342585396 ,1937033938 ,2640572511 ,1654299090 ,3692403553 ,4233871309 ,3497650794 ,862629010 ,2943236032 ,2426458545 ,1603307207 ,1133453895 ,3099196360 ,2208657629 ,2747653927 ,931059398 ,761573964 ,3157853227 ,785880413 ,730313442 ,124945756 ,2937117055 ,3295982469 ,1724353043 ,3021675344 ,3884886417 ,4010150098 ,4056961966 ,699635835 ,2681338818 ,1339167484 ,720757518 ,2800161476 ,2376097373 ,1532957371 ,3902664099 ,1238982754 ,3725394514 ,3449176889 ,3570962471 ,4287636090 ,4087307012 ,3603343627 ,202242161 ,2995682783 ,1620962684 ,3704723357 ,371613603 ,2814834333 ,2111005706 ,624778151 ,2094172212 ,4284947003 ,1211977835 ,991917094 ,1570449747 ,2962370480 ,1259410321 ,170182696 ,146300961 ,2836829791 ,619452428 ,2723670296 ,1881399711 ,1161269684 ,1675188680 ,4132175277 ,780088327 ,3409462821 ,1036518241 ,1834958505 ,3048448173 ,161811569 ,618488316 ,44795092 ,3918322701 ,1924681712 ,3239478144 ,383254043 ,4042306580 ,2146983041 ,3992780527 ,3518029708 ,3545545436 ,3901231469 ,1896136409 ,2028528556 ,2339662006 ,501326714 ,2060962201 ,2502746480 ,561575027 ,581893337 ,3393774360 ,1778912547 ,3626131687 ,2175155826 ,319853231 ,986875531 ,819755096 ,2915734330 ,2688355739 ,3482074849 ,2736559 ,2296975761 ,1029741190 ,2876812646 ,690154749 ,579200347 ,4027461746 ,1285330465 ,2701024045 ,4117700889 ,759495121 ,3332270341 ,2313004527 ,2277067795 ,4131855432 ,2722057515 ,1264804546 ,3848622725 ,2211267957 ,4100593547 ,959123777 ,2130745407 ,3194437393 ,486673947 ,1377371204 ,17472727 ,352317554 ,3955548058 ,159652094 ,1232063192 ,3835177280 ,49423123 ,3083993636 ,733092 ,2120519771 ,2573409834 ,1112952433 ,3239502554 ,761045320 ,1087580692 ,2540165110 ,641058802 ,1792435497 ,2261799288 ,1579184083 ,627146892 ,2165744623 ,2200142389 ,2167590760 ,2381418376 ,1793358889 ,3081659520 ,1663384067 ,2009658756 ,2689600308 ,739136266 ,2304581039 ,3529067263 ,591360555 ,525209271 ,3131882996 ,294230224 ,2076220115 ,3113580446 ,1245621585 ,1386885462 ,3203270426 ,123512128 ,12350217 ,354956375 ,4282398238 ,3356876605 ,3888857667 ,157639694 ,2616064085 ,1563068963 ,2762125883 ,4045394511 ,4180452559 ,3294769488 ,1684529556 ,1002945951 ,3181438866 ,22506664 ,691783457 ,2685221343 ,171579916 ,3878728600 ,2475806724 ,2030324028 ,3331164912 ,1708711359 ,1970023127 ,2859691344 ,2588476477 ,2748146879 ,136111222 ,2967685492 ,909517429 ,2835297809 ,3206906216 ,3186870716 ,341264097 ,2542035121 ,3353277068 ,548223577 ,3170936588 ,1678403446 ,297435620 ,2337555430 ,466603495 ,1132321815 ,1208589219 ,696392160 ,894244439 ,2562678859 ,470224582 ,3306867480 ,201364898 ,2075966438 ,1767227936 ,2929737987 ,3674877796 ,2654196643 ,3692734598 ,3528895099 ,2796780123 ,3048728353 ,842329300 ,191554730 ,2922459673 ,3489020079 ,3979110629 ,1022523848 ,2202932467 ,3583655201 ,3565113719 ,587085778 ,4176046313 ,3013713762 ,950944241 ,396426791 ,3784844662 ,3477431613 ,3594592395 ,2782043838 ,3392093507 ,3106564952 ,2829419931 ,1358665591 ,2206918825 ,3170783123 ,31522386 ,2988194168 ,1782249537 ,1105080928 ,843500134 ,1225290080 ,1521001832 ,3605886097 ,2802786495 ,2728923319 ,3996284304 ,903417639 ,1171249804 ,1020374987 ,2824535874 ,423621996 ,1988534473 ,2493544470 ,1008604435 ,1756003503 ,1488867287 ,1386808992 ,732088248 ,1780630732 ,2482101014 ,976561178 ,1543448953 ,2602866064 ,2021139923 ,1952599828 ,2360242564 ,2117959962 ,2753061860 ,2388623612 ,4138193781 ,2962920654 ,2284970429 ,766920861 ,3457264692 ,2879611383 ,815055854 ,2332929068 ,1254853997 ,3740375268 ,3799380844 ,4091048725 ,2006331129 ,1982546212 ,686850534 ,1907447564 ,2682801776 ,2780821066 ,998290361 ,1342433871 ,4195430425 ,607905174 ,3902331779 ,2454067926 ,1708133115 ,1170874362 ,2008609376 ,3260320415 ,2211196135 ,433538229 ,2728786374 ,2189520818 ,262554063 ,1182318347 ,3710237267 ,1221022450 ,715966018 ,2417068910 ,2591870721 ,2870691989 ,3418190842 ,4238214053 ,1540704231 ,1575580968 ,2095917976 ,4078310857 ,2313532447 ,2110690783 ,4056346629 ,4061784526 ,1123218514 ,551538993 ,597148360 ,4120175196 ,3581618160 ,3181170517 ,422862282 ,3227524138 ,1713114790 ,662317149 ,1230418732 ,928171837 ,1324564878 ,1928816105 ,1786535431 ,2878099422 ,3290185549 ,539474248 ,1657512683 ,552370646 ,1671741683 ,3655312128 ,1552739510 ,2605208763 ,1441755014 ,181878989 ,3124053868 ,1447103986 ,3183906156 ,1728556020 ,3502241336 ,3055466967 ,1013272474 ,818402132 ,1715099063 ,2900113506 ,397254517 ,4194863039 ,1009068739 ,232864647 ,2540223708 ,2608288560 ,2415367765 ,478404847 ,3455100648 ,3182600021 ,2115988978 ,434269567 ,4117179324 ,3461774077 ,887256537 ,3545801025 ,286388911 ,3451742129 ,1981164769 ,786667016 ,3310123729 ,3097811076 ,2224235657 ,2959658883 ,3370969234 ,2514770915 ,3345656436 ,2677010851 ,2206236470 ,271648054 ,2342188545 ,4292848611 ,3646533909 ,3754009956 ,3803931226 ,4160647125 ,1477814055 ,4043852216 ,1876372354 ,3133294443 ,3871104810 ,3177020907 ,2074304428 ,3479393793 ,759562891 ,164128153 ,1839069216 ,2114162633 ,3989947309 ,3611054956 ,1333547922 ,835429831 ,494987340 ,171987910 ,1252001001 ,370809172 ,3508925425 ,2535703112 ,1276855041 ,1922855120 ,835673414 ,3030664304 ,613287117 ,171219893 ,3423096126 ,3376881639 ,2287770315 ,1658692645 ,1262815245 ,3957234326 ,1168096164 ,2968737525 ,2655813712 ,2132313144 ,3976047964 ,326516571 ,353088456 ,3679188938 ,3205649712 ,2654036126 ,1249024881 ,880166166 ,691800469 ,2229503665 ,1673458056 ,4032208375 ,1851778863 ,2563757330 ,376742205 ,1794655231 ,340247333 ,1505873033 ,396524441 ,879666767 ,3335579166 ,3260764261 ,3335999539 ,506221798 ,4214658741 ,975887814 ,2080536343 ,3360539560 ,571586418 ,138896374 ,4234352651 ,2737620262 ,3928362291 ,1516365296 ,38056726 ,3599462320 ,3585007266 ,3850961033 ,471667319 ,1536883193 ,2310166751 ,1861637689 ,2530999841 ,4139843801 ,2710569485 ,827578615 ,2012334720 ,2907369459 ,3029312804 ,2820112398 ,1965028045 ,35518606 ,2478379033 ,643747771 ,1924139484 ,4123405127 ,3811735531 ,3429660832 ,3285177704 ,1948416081 ,1311525291 ,1183517742 ,1739192232 ,3979815115 ,2567840007 ,4116821529 ,213304419 ,4125718577 ,1473064925 ,2442436592 ,1893310111 ,4195361916 ,3747569474 ,828465101 ,2991227658 ,750582866 ,1205170309 ,1409813056 ,678418130 ,1171531016 ,3821236156 ,354504587 ,4202874632 ,3882511497 ,1893248677 ,1903078632 ,26340130 ,2069166240 ,3657122492 ,3725758099 ,831344905 ,811453383 ,3447711422 ,2434543565 ,4166886888 ,3358210805 ,4142984013 ,2988152326 ,3527824853 ,982082992 ,2809155763 ,190157081 ,3340214818 ,2365432395 ,2548636180 ,2894533366 ,3474657421 ,2372634704 ,2845748389 ,43024175 ,2774226648 ,1987702864 ,3186502468 ,453610222 ,4204736567 ,1392892630 ,2471323686 ,2470534280 ,3541393095 ,4269885866 ,3909911300 ,759132955 ,1482612480 ,667715263 ,1795580598 ,2337923983 ,3390586366 ,581426223 ,1515718634 ,476374295 ,705213300 ,363062054 ,2084697697 ,2407503428 ,2292957699 ,2426213835 ,2199989172 ,1987356470 ,4026755612 ,2147252133 ,270400031 ,1367820199 ,2369854699 ,2844269403 ,79981964 ,624]
    pos = 623
    seq=np.array(seq[:-1])
    # subset = randperm(n_seg, ceil(n_seg * pct_data / 100)); % need to improve this
    k = np.ceil(n_seg * pct_data / 100).astype(int) #by default k = n_seg since pctdata is not set by default    
    rng.set_state(('MT19937',seq,624,0,0.0))
    subset = rng.permutation(np.arange(0,n_seg))
    if test:
        subset=np.array([124,100,175,36,9,93,92,176,55,120,171,147,97,122,52,191,166,161,149,136,141,48,167,47,134,229,114,150,108,177,37,49,104,130,187,148,151,219,81,164,172,225,217,10,212,65,99,3,7,218,220,129,1,158,183,105,132,4,146,230,197,173,43,163,213,28,165,89,83,222,76,109,206,72,145,110,203,20,86,45,139,181,34,111,195,30,21,196,143,96,22,56,31,84,231,33,224,170,85,182,228,199,126,64,131,115,67,82,14,205,127,40,69,77,209,178,180,12,189,138,50,142,119,211,117,57,162,186,91,44,210,2,61,87,60,79,75,184,51,154,59,133,174,159,95,66,155,135,190,193,53,118,18,38,29,140,156,121,168,201,116,169,152,200,88,204,32,112,80,58,128,35,25,24,17,153,13,106,192,39,103,41,216,113,101,70,185,11,123,137,78,98,27,73,207,5,74,90,63,16,54,179,157,102,226,223,62,46,26,202,23,6,107,198,94,208,8,215,68,144,125,188,227,221,15,42,71,214,19,160,194]) -1#for test
    # i tried but no luck

    # % calculate windowed spectrums
    psdmed = np.zeros((ncomp, nfreqs));
    icaact2=icaact#np.reshape(icaact,(nchannels,nframes,nepochs),order='F')
    if test:
        icaact2=loadmat('data/icaact.mat')['icaact2']
    for it in np.arange(ncomp):
        shape_=tuple(np.concatenate((np.array([1]),np.array(index.shape)*np.array([1,nepochs]))).tolist())
        temp = np.reshape(icaact2[it,index,:],shape_,order='F')
        temp = temp[:, :, subset]*window[None,:,None]; #https://stackoverflow.com/questions/23544889/converting-bsxfun-with-times-to-numpy , https://stackoverflow.com/questions/16229823/how-to-multiply-numpy-2d-array-with-numpy-1d-array
        temp = np.fft.fft(temp,n_points,axis=1)
        temp = temp * np.conj(temp)
        temp = np.real(temp)
        temp = temp[:, np.arange(1,nfreqs + 1), :] * 2 / (srate*np.sum(np.square(window)));
        if nfreqs == nyquist:
            temp[:,-1,:] = temp[:,-1,:]/2
        psdmed[it,:] = 20*np.log10(np.median(temp,2))
    return psdmed

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def eeg_autocorr_fftw(icaact_,srate_,pct_data=100,test=False):
    """
    assume channels,frames,epochs
    """
    srate= int(srate_)

    if icaact_.ndim == 3:
        nchannels = icaact_.shape[0]
        nframes = icaact_.shape[1]
        nepochs = icaact_.shape[2]
    else:
        nchannels = icaact_.shape[0]
        nframes = icaact_.shape[1]
        nepochs = 1

    nfft = next_power_of_2(2*nframes-1);

    icaact=icaact_#np.reshape(icaact_,(nchannels,nframes,nepochs),order='F')

    if test:
        icaact=loadmat('data/icaact.mat')['icaact2']

    # % calc autocorrelation
    # fftw('planner', 'hybrid');
    ncomp = icaact.shape[0]
    ac = np.zeros((ncomp, nfft));
    for it in np.arange(ncomp):
        X = np.fft.fft(icaact[it,:,:],nfft,axis=0)
        ac[it,:] = np.mean(np.square(np.abs(X)),axis=1)

    ac = np.real(np.fft.ifft(ac,axis=1));

    if nframes < srate:
        ac = np.concatenate([ac[:,:nframes],np.zeros((ac.shape[0],srate - nframes + 1))],axis=-1);
    else:
        ac = ac[:,:srate+1]

    # % normalize by 0-tap autocorrelation
    ac = ac[:,:(srate+1)]/ac[:,0][:,None]

    # % resample to 1 second at 100 samples/sec
    frac = fractions.Fraction.from_float(100/256).limit_denominator()
    p=frac.numerator
    q=frac.denominator
    desired_length = int((np.ceil(ac.shape[1]*p/q)).astype(int))
    resample= sx.resample_poly(ac, desired_length, ac.shape[1],axis=-1)
    resample = resample[:,1:]
    return resample