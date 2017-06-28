close all;
warning off backtrace

addpath('Utils');
global sRoot;
tmp = mfilename('fullpath');tmp =  strsplit(tmp, '/');tmp= tmp(1:end-2);
sRoot = strjoin(tmp,'/');
setup_path


dataset_name = {'VggAffineDataset'};%,'EFDataset','WebcamDataset'

for i = 1:numel(dataset_name)
    parameters.nameDataset = dataset_name{i};%for saving at the end
    if(strcmp(parameters.nameDataset,'VggAffineDataset'))
        parameters.testsets = {'bikes','bark', 'boat','graf','leuven','trees','ubc', 'wall'};
    end
    if(strcmp(parameters.nameDataset,'EFDataset'))
        parameters.testsets = {'notredame', 'obama', 'yosemite', 'paintedladies', 'rushmore'};
    end
    if(strcmp(parameters.nameDataset,'WebcamDataset'))
        parameters.testsets = {'Panorama','Chamonix', 'StLouis', 'Courbevoie', 'Frankfurt'};
    end

    parameters.models = {'Mexico'};
    parameters.optionalTildeSuffix = 'Standard';

    parameters.numberOfKeypoints  = {200,200,200,200,200,200,200,200,200,200,200,200,200};
    parameters.repeatabilityType = 'OXFORD';
    parameters.list_method = {'TILDEP', 'TILDEP24', 'DDetPoint', 'LearnedConvolutional', 'CovariantPoint'}';
    computeKP(parameters);
end
