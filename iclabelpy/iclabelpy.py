
from numpy.core.fromnumeric import squeeze
from iclabelpy.utils import topoplotFast,reref,eeg_rpsd,loadmat
import numpy as np
import scipy.io as sio

#import numpy.matlib as npm
import scipy.io
def iclabel(icawinv):
    #features = ICL_feature_extractor(EEG, flag_autocorr)
    pass

def ICL_feature_extractor(EEG,mixing_,flag_autocorr=False,flag_reref=False,dummy=[],test=False):
    raw = EEG.copy()
    mixing = mixing_.copy()
    ncomp = mixing.shape[1]
    nchannels = mixing.shape[0]

    locs = raw._get_channel_positions()
    # see https://github.com/mne-tools/mne-python/blob/24377ad3200b6099ed47576e9cf8b27578d571ef/mne/io/eeglab/eeglab.py#L105
    
    #% assuming chanlocs are correct
    if flag_reref:
        raw._data = reref(raw._data)
    # Rereference ICA Matrix
        mixing = reref(mixing)

    #% calculate ica activations if missing and cast to double
    demixing = np.linalg.pinv(mixing)
    if raw._data.ndim == 3:
        nchannels = raw._data.shape[1]
        nframes = raw._data.shape[2]
        nepochs = raw._data.shape[0]
    else:
        nchannels = raw._data.shape[0]
        nframes = raw._data.shape[1]
        nepochs = 1

    icaact = np.dot(demixing,np.reshape(raw._data,(nchannels,nframes*nepochs),order='F'))
    if raw._data.ndim == 3:
        icaact = np.reshape(icaact,raw._data.shape,order='F')
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
    # if flag_autocorr
    #     if EEG.trials == 1
    #         if EEG.pnts / EEG.srate > 5
    #             autocorr = eeg_autocorr_welch(EEG);
    #         else
    #             autocorr = eeg_autocorr(EEG);
    #         end
    #     else
    #         autocorr = eeg_autocorr_fftw(EEG);
    #     end

    #     % reshape and cast
    #     autocorr = single(permute(autocorr, [3 2 4 1]));
    # end

    # %% format outputs
    # if flag_autocorr
    #     features = {0.99 * topo, 0.99 * psd, 0.99 * autocorr};
    # else
    #     features = {0.99 * topo, 0.99 * psd};
    # end
