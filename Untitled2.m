clc, clear all ,close all;

cfg.path.wd = pwd;

% EEGLAB: Should have neuroscanio, biosig, prep, bvaio extensions
cfg.path.eeglab = 'Y:\code\matlab_packages\eeglab';


cd(cfg.path.eeglab)
%addpath(genpath(eeglab_path));
eeglab % eeglab says is better to start it rather than adding it manually
close all

cd(cfg.path.wd)
%%
file='Y:\code\iclabelpy\data\sub-010317_PREP_clean_ICA.set';
%file='E:\datasets\mica_release\datasets\prep\ke70_PREP.set';
EEG = pop_loadset(file);
EEG = iclabel(EEG);
iclabels = EEG.etc.ic_classification.ICLabel.classifications; 

