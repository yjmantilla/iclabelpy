import numpy as np
from iclabelpy.iclabelpy import ICL_feature_extractor, iclabel
from iclabelpy.utils import topoplotFast,reref
import scipy.io as sio
if __name__ == '__main__':
    import mne
    import numpy as np
    import scipy.io as sio
    from iclabelpy.mat import loadmat

    data = loadmat('Y:\code\iclabelpy\data\sub-010317_PREP_clean_ICA.mat')
    ica = loadmat('Y:\code\iclabelpy\data\ica.mat')

    print('ok')

    flag_autocorr = True
    EEG = data['EEG']
    mixing = ica['icawinv']
    demixing = ica['icaweights']@ica['icasphere']

    #ICL_feature_extractor(EEG, flag_autocorr)
    ncomp = mixing.shape[1]
    nchannels = mixing.shape[0]
    average_reference = False
    if not average_reference:
        data = reref(EEG['data'])
        # Rereference ICA Matrix
        newICAchaninds = reref(mixing)
    icaact = EEG['icaact']
    assert np.all(np.isreal(icaact)) == True # Check Activations are real

    test=True
    raw = mne.read_epochs_eeglab('Y:\code\iclabelpy\data\sub-010317_PREP_clean_ICA.set')
    features = iclabel(raw,mixing,demixing,True,True,test)
    sio.savemat('features_python.mat',{'features_python':features})
    print('ok')
