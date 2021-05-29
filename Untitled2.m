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
EEG.icaact=[];
EEG = iclabel(EEG);
iclabels = EEG.etc.ic_classification.ICLabel.classifications;

load('features_python.mat');
for i=1:3,features_python{i}=single(features_python{i});,end
labels_matlab = EEG.etc.ic_classification.ICLabel.classifications;
labels_python = run_ICL('default',features_python{:});
[argvalue_matlab, argmax_matlab] = max(labels_matlab,[],2);
[argvalue_python, argmax_python] = max(labels_python,[],2);
comp_values= [argvalue_matlab,argvalue_python];
comp_idx= [argmax_matlab,argmax_python];
diff_label = sum(argmax_matlab ~= argmax_python);
err = immse(argvalue_matlab,argvalue_python);
