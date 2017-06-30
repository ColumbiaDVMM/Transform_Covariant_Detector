function evaluate_OxfordEFDataset(dataset_name,numberOfKeypoints)
    addpath('Utils');
    global sRoot;
    tmp = mfilename('fullpath');
    tmp =  strsplit(tmp, '/');
    tmp= tmp(1:end-1);
    sRoot = strjoin(tmp,'/');
    setup_path

    parameters.nameDataset = dataset_name;
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

    parameters.repeatabilityType = 'OXFORD';
    %parameters.list_method = {'TILDEP', 'TILDEP24', 'DDetPoint', 'LearnedConvolutional', 'CovariantPoint'}';
    parameters.list_method = {'TILDEP24','CovariantPoint'}';
    parameters.numberOfKeypoints  = {};
    for i = 1:10*size(parameters.list_method,1)
        parameters.numberOfKeypoints{i} = numberOfKeypoints;
    end
    repeatability = computeKP(parameters);
    disp(mean(repeatability));
end
