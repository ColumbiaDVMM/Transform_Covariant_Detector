close all;
warning off backtrace

addpath('Utils');
global sRoot;
tmp = mfilename('fullpath');
tmp =  strsplit(tmp, '/');
tmp= tmp(1:end-1);
sRoot = strjoin(tmp,'/');
disp(sRoot)
setup_path

%dataset_name = {'VggAffineDataset','EFDataset','WebcamDataset'};
dataset_name = {'VggAffineDataset'};
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
    %
   
    parameters.models = {'Mexico'};
    parameters.optionalTildeSuffix = 'Standard';

    parameters.numberOfKeypoints  = {1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000};
    parameters.repeatabilityType = 'OXFORD';
    parameters.list_method = {'CovariantPoint'}';
    Allrepeatability = computeKP(parameters);
    disp(Allrepeatability);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% evaluate_OxfordEFDataset_1000.m ends here
