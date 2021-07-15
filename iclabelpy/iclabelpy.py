
from matplotlib.pyplot import flag
from numpy.core.fromnumeric import squeeze
from iclabelpy.utils import topoplotFast,reref,eeg_rpsd,loadmat,eeg_autocorr_fftw
import numpy as np
import scipy.io as sio

#import numpy.matlib as npm
import scipy.io
def iclabel(EEG,mixing,demixing=None,flag_autocorr=True,flag_reref=False,test=False):
    #% extract features
    features = ICL_feature_extractor(EEG,mixing,demixing,flag_autocorr,flag_reref,test)
    #% run ICL
    #labels = run_ICL(version, features{:});

    # % save into EEG
    # disp 'ICLabel: saving results...'
    # EEG.etc.ic_classification.ICLabel.classes = ...
    #     {'Brain', 'Muscle', 'Eye', 'Heart', ...
    #     'Line Noise', 'Channel Noise', 'Other'};
    # EEG.etc.ic_classification.ICLabel.classifications = labels;
    # EEG.etc.ic_classification.ICLabel.version = version;
    return features

def ICL_feature_extractor(EEG,mixing_,demixing_=None,flag_autocorr=True,flag_reref=False,test=False):
    raw = EEG.copy()
    mixing = mixing_.copy()
    ncomp = mixing.shape[1]
    eeg = np.transpose(raw._data*1e6, (1,2,0)) # epochs,channels,frames to channels,frames,epochs
    nchannels = eeg.shape[0]

    locs = raw._get_channel_positions()
    # see https://github.com/mne-tools/mne-python/blob/24377ad3200b6099ed47576e9cf8b27578d571ef/mne/io/eeglab/eeglab.py#L105
    
    #% assuming chanlocs are correct
    if flag_reref:
        eeg = reref(eeg)
    # Rereference ICA Matrix
        mixing = reref(mixing)

    #% calculate ica activations if missing and cast to double
    if demixing_ is None:
        demixing = np.linalg.pinv(mixing)
    else:
        demixing = demixing_.copy()
    if eeg.ndim == 3:
        nchannels = eeg.shape[0]
        nframes = eeg.shape[1]
        nepochs = eeg.shape[2]
    else:
        nchannels = eeg.shape[0]
        nframes = eeg.shape[1]
        nepochs = 1

    eeg_cont = np.reshape(eeg,(nchannels,nframes*nepochs),order='F')
    if test:
        icaact=sio.loadmat('data/icaact.mat',squeeze_me=False)['icaact2']
    else:
        #icaact = demixing.astype('float32') @ eeg_cont.astype('float32')
        icaact = demixing @ eeg_cont
        if eeg.ndim == 3:
            icaact = np.reshape(icaact,(ncomp,nframes,nepochs),order='F')
    #EEG.icaact = double(EEG.icaact);
    #% check ica is real
    assert np.all(np.isreal(icaact)) == True

    #%% calc topo
    topo = np.zeros((32, 32, 1, ncomp));
    for comp in range(ncomp):
        values = mixing[:,comp]
        temp_topo,plotrad = topoplotFast(locs,values,False)
        temp_topo[np.where(np.isnan(temp_topo))] = 0
        topo[:, :, 0, comp] = temp_topo / np.max(np.abs(temp_topo.flatten()));

    #% cast
    #topo = single(topo);

    #%% calc psd
    psd = eeg_rpsd(icaact,raw.info['sfreq'],demixing,nfreqs=100, pct_data=None,test=test);

    # % extrapolate or prune as needed
    nfreq = psd.shape[1]
    if nfreq < 100:
        psd = np.concatenate([psd,np.tile(psd[:,-1][:,None],(1,100-nfreq))],axis=-1)

    # % undo notch filter
    for linenoise_ind in [50-1,60-1]: #index offset
        linenoise_around = [linenoise_ind-1,linenoise_ind+1];
        difference = psd[:,linenoise_around] - psd[:, linenoise_ind][:,None]
        notch_ind = np.all(difference > 5, 1);
        if np.any(notch_ind):
            psd[notch_ind, linenoise_ind] = np.mean(psd[notch_ind, [linenoise_around]], 1);

    # % normalize
    psd = psd/np.max(np.abs(psd),1)[:,None]

    # % reshape and cast
    # psd = single(permute(psd, [3 2 4 1]));
    psd = np.transpose(psd[:, :, None,None], (2,1,3,0))
    if test:
        psd2_test=sio.loadmat('data/permuted_psd.mat',squeeze_me=False)['psd2']
        assert np.all(np.isclose(psd2_test,psd))


    # %% calc autocorrelation?
    if flag_autocorr:
        if nepochs == 1:
            if nframes/raw.info['sfreq'] > 5:
                pass # autocorr = eeg_autocorr_welch(EEG);
            else:
                pass # autocorr = eeg_autocorr(EEG);
        else:
            autocorr = eeg_autocorr_fftw(icaact,raw.info['sfreq'],test=test);

        #% reshape and cast
        #autocorr = single(permute(autocorr, [3 2 4 1]));
        autocorr = np.transpose(autocorr[:, :, None,None], (2,1,3,0))

    # %% format outputs
    if flag_autocorr:
        features = [0.99 * topo, 0.99 * psd, 0.99 * autocorr];
    else:
        features = [0.99 * topo, 0.99 * psd];

    return features
