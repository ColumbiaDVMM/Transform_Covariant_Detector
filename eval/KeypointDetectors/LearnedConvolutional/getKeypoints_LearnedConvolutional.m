%% getKeypoints_LearnedConvolutional.m --- 
% 
% Filename: getKeypoints_LearnedConvolutional.m
% Description: 
% Author: Kwang Moo Yi, Yannick Verdie
% Maintainer: Kwang Moo Yi, Yannick Verdie
% Created: Tue Jun 16 17:17:08 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:18:04 2015 (+0200)
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
% Learned Convolutial Filters code using the coeeficients from the
% authors' website. See related copyright issues.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%% Code:


function [keypts] = getKeypoints_LearnedConvolutional(img_info, p)

    fixed_scale = 10;%scale of the kp
    pyramid_level = 5;
    
    h = [-127,  -53,   49,  123,  123,   53,  -49, -123;
          -92,  -39,   35,   88,   88,   39,  -35,  -85; 
          -35,  -18,   11,   32,   32,   14,  -11,  -28; 
           21,    7,  -11,  -25,  -21,   -7,   14,   28; 
           56,   21,  -25,  -56,  -56,  -21,   25,   60; 
           67,   28,  -25,  -64,  -64,  -28,   25,   64; 
           60,   28,  -21,  -53,  -56,  -25,   18,   49; 
           53,   25,  -14,  -42,  -46,  -21,   14,   39]'; % transposed since it was originally C

    learnconv_name = [img_info.full_feature_prefix '_learnconv_keypoints.mat'];
    keypts = [];
    factor = 1;
    if ~exist(learnconv_name, 'file')
        % convolve to get response map
        input_gray_image = img_info.image_gray;
        for j = 1:pyramid_level
            [score_res_t] = imfilter(double(input_gray_image), h, 'symmetric', 'conv');
            [score_res_t,binary_res_t] = ApplyNonMax2Score(score_res_t, [], true);
            
            idx = find(binary_res_t);
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
            G = fspecial('gaussian',[5 5],sqrt(2));
            %# Filter it
            input_gray_image = imfilter(input_gray_image,G,'same');
            input_gray_image = imresize(input_gray_image, 1/sqrt(2));
        end

        % safety check to prevent race condition
        if ~exist(learnconv_name, 'file')
            save(learnconv_name, 'keypts', '-v7.3');
        end
    else
        % loop to prevent race condition
        bFileReady = false;
        while (~bFileReady)
            try
                loadkey = load(learnconv_name);
                keypts = loadkey.keypts;
                bFileReady = true;
            catch
                pause(rand*5+5); % try again in 5~10 seconds
            end
        end
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% getKeypoints_LearnedConvolutional.m ends here
