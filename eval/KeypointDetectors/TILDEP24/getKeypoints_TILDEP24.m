%% getKeypoints_TILDEP24.m --- 
% 
% Filename: getKeypoints_TILDEP24.m
% Description: 
% Author: Kwang Moo Yi, Yannick Verdie
% Maintainer: Kwang Moo Yi, Yannick Verdie
% Created: Tue Jun 16 17:21:28 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:21:34 2015 (+0200)
%           By: Kwang
%     Update #: 1
% URL: 
% Doc URL: 
% Keywords: 
% Compatibility: 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%% Commentary: 
% 
% 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%% Change Log:
% 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Copyright (C), EPFL Computer Vision Lab.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%% Code:


function [keypts, score_res] = getKeypoints_TILDEP24(img_info, p)
global sRoot;

    suffix = '';
    if (isfield(p, 'optionalTildeSuffix'))
        suffix = ['_' p.optionalTildeSuffix];
    end

    trainset_name = p.trainset_name;
    testset_name = p.testset_name;
    fixed_scale = 10;%scale of the kp
    pyramid_level = 5;
    
    numFilters = 24;

    name_our_approx = ['BestFilters' suffix '/Approx/' trainset_name 'Med' num2str(numFilters) '.mat'];
    name_our_orig = ['BestFilters' suffix '/Original/' trainset_name 'Med.mat'];

    file_prefix = img_info.full_feature_prefix;
    file_suffix = ['_dump_approx' num2str(numFilters) '.mat'];
    filter_res_file_name = [file_prefix '_Train_' trainset_name '_Test_' testset_name file_suffix];        

    brun_filter = ~exist(filter_res_file_name,'file');
    score_res = cell(pyramid_level,1);
    binary_res = cell(pyramid_level,1);
    if(brun_filter)
        input_color_image = img_info.image_color;
        clear param;load([sRoot '/filters/' name_our_orig],'res');
        param = res.param;
        delta = res.delta; %backup old delta from previous res structure
        res = load([sRoot '/filters/' name_our_approx]);
        %param = res.param;
        res.param = param;
        res.delta = delta;
        % res.newFilters = newFilters
        if isfield(param, 'fMultiScaleList')
            param = rmfield(param, 'fMultiScaleList');
        end

        if(size(input_color_image,3) == 1)
            input_color_image = repmat(input_color_image, [1 1 3]);
        end
        
        for j = 1:pyramid_level
            input_preproc_image = PreProcessTrainImage(input_color_image, param);

            [score_res_t, ~] = fastELLFiltering_approx(input_preproc_image, -Inf, res);
            fs = param.nBinSize;
            [score_res_t, max_img] = ApplyNonMax2Score(score_res_t, param);
            binary_res_t = max_img .* (score_res_t > -Inf);
            % Mutiplied fs with param.fScaling to consider scaling (25/04/2014 KMYI)
            binary_res_t(1:fs,:) = 0;
            binary_res_t(end-fs+1:end,:) = 0;
            binary_res_t(:,1:fs) = 0;
            binary_res_t(:,end-fs+1:end) = 0;
            score_res{j} = score_res_t;
            binary_res{j} = binary_res_t;
            
            G = fspecial('gaussian',[5 5],sqrt(2));
            %# Filter it
            input_color_image = imfilter(input_color_image,G,'same');
            input_color_image = imresize(input_color_image, 1/sqrt(2));
            
        end
        parsavefilter(filter_res_file_name, score_res, binary_res);
    else
        %display(' -- loaded dumped filter response');
        loadres = load(filter_res_file_name);
        score_res = loadres.score_res;
        binary_res = loadres.binary_res;
    end

    
    keypts = [];
    factor = 1;
    for j = 1:pyramid_level
        score_res_t = score_res{j};
        binary_res_t = binary_res{j};
        idx = find(binary_res_t);
        if(sum(sum(~isreal(score_res_t))))
            error(['Score Result for Our Filter has imaginary parts']);
        end
        [I,J] = ind2sub(size(binary_res_t),idx);
        keypts_t = [J I zeros(size(I,1),3) repmat(fixed_scale,size(I,1),1)]';
        keypts_t = mergeScoreImg2Keypoints(keypts_t, score_res_t);
        keypts_t(1,:) = keypts_t(1,:)*factor;
        keypts_t(2,:) = keypts_t(2,:)*factor;
        keypts_t(6,:) = keypts_t(6,:)*factor;
        factor = factor*sqrt(2);
        if(isempty(keypts))
            keypts = keypts_t;
        else
            keypts = [keypts keypts_t];
        end
    end

end

function [] = parsavefilter(fname, score_res, binary_res)
    save(fname, 'score_res', 'binary_res');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% getKeypoints_TILDEP24.m ends here
