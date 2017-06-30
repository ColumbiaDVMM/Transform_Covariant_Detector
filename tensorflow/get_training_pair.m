clc;close all;

working_dir = '../data/patch_set/';
dataset_name = 'mexico_tilde_p24_Mexico';
load([working_dir, 'standard_patch/' dataset_name, '_patches.mat']);

training_number = 256000;
test_number = 128;
training_offset = 0;
offset = 5000;
stride = 1;
total_number = training_number+test_number;
seed = 0;
rng(seed);

feature_dim = 2;

%generate random rotation
translation = (2*rand(total_number,2)-1)*8;
rotation = (2*rand(total_number,1)-1)*pi;
shear = (2*rand(total_number,2)-1)*0.15;
scale = 1.0 + (2*rand(total_number,2)-1)*0.15;

scale(scale<0.65) = 0.65;

patch_size = 32;
patch_size = patch_size/2;

im = uint8(zeros(total_number,3,patch_size*2,patch_size*2));
warped_im = uint8(zeros(total_number,3,patch_size*2,patch_size*2));
transform_matrix = zeros(total_number,feature_dim);
index = 1:training_number;
index = ceil(rand(training_number,1)*offset)*stride;
patches_total = patches;
disp(size(patches_total));
disp(max(index+training_offset));
patches = patches_total(index+training_offset,:,:,:);
patches(training_number+(1:test_number),:,:,:) = ...
patches_total((end-test_number+1):end,:,:,:);

parpool;
parfor i = 1:total_number
    I = patches(i,:,:,:);
    I = squeeze(I);
    I = permute(I,[2,3,1]);

    tmp_translation = round(translation(i,:));

    identity_matrix = eye(3);
    theta = rotation(i,1);
    rotation_matrix = [cos(theta),-1*sin(theta),0; sin(theta), cos(theta),0;0,0,1];
    tmp_scale = scale(i,:);
    scale_matrix = [tmp_scale(1),0,0;0,tmp_scale(2),0;0,0,1];
    tmp_shear = shear(i);
    shear_matrix = [1,0,0;tmp_shear(1),1,0;0,0,1];
    
    affine_matrix = shear_matrix*scale_matrix*rotation_matrix*identity_matrix;
    tform = affine2d(affine_matrix');
    
    %transform patches 
    J = imwarp(I,tform,'linear','FillValues',[0,0,0]);

    I_center_x = round(size(J,2)/2);
    I_center_y = round(size(J,1)/2);

    J_center_x = round(size(J,2)/2)+tmp_translation(1);
    J_center_y = round(size(J,1)/2)+tmp_translation(2);

    %check boundary
    if J_center_x < patch_size+1
        J_center_x = patch_size+1;
        tmp_translation(1) = patch_size + 1 - round(size(J,2)/2);
    end
    if J_center_x + patch_size > size(J,2)
        J_center_x = size(J,2) - patch_size;
        tmp_translation(1) = size(J,2) - patch_size - round(size(J,2)/2);
    end

    if J_center_y < patch_size+1
        J_center_y = patch_size+1;
        tmp_translation(2) = patch_size + 1 - round(size(J,1)/2);
    end
    if J_center_y + patch_size > size(J,1)
        J_center_y = size(J,1) - patch_size;
        tmp_translation(2) = size(J,1) - patch_size - round(size(J,1)/2);
    end

    %standard patch, we add some rotation and shearing to standard patch
    crop_I = J(I_center_y-patch_size+1:I_center_y+patch_size,...
        I_center_x-patch_size+1:I_center_x+patch_size,:);
    
    %transformed patch
    crop_J = J(J_center_y-patch_size+1:J_center_y+patch_size,...
        J_center_x-patch_size+1:J_center_x+patch_size,:);
    
    %gt transform
    transform_matrix(i,:) = [tmp_translation(1)./(patch_size*2/3.0),tmp_translation(2)./(patch_size*2/3.0)];

    crop_I = permute(crop_I,[3,1,2]);
    im(i,:,:,:) = crop_I;
    crop_J = permute(crop_J,[3,1,2]);
    warped_im(i,:,:,:) = crop_J;
end
delete(gcp);

total_im = im;
total_warped_im = warped_im;
total_transform_matrix = transform_matrix;

im = total_im(1:training_number,:,:,:);
warped_im = total_warped_im(1:training_number,:,:,:);
transform_matrix = total_transform_matrix(1:training_number,:);
save([working_dir, 'train_pair/' dataset_name '_train_point.mat'],'im','warped_im','transform_matrix');

index = training_number+1;
im = total_im(index:(index+test_number-1),:,:,:);
warped_im = total_warped_im(index:(index+test_number-1),:,:,:);
transform_matrix = total_transform_matrix(index:(index+test_number-1),:);

save([working_dir, 'train_pair/', dataset_name '_test_point.mat'],'im','warped_im','transform_matrix');
%fclose(fout);
