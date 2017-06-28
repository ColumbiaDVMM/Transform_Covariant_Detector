function [img_list] = load_image_list(dir_name, dataset_name)
%LOAD_IMAGE_LIST Summary of this function goes here
%   Detailed explanation goes here
%-------------------------------------------------------------------------
% Retrieve the image list and load the images and SIFT
%-------------------------------------------------------------------------

img_list = dir ([dir_name, dataset_name,'/*.jpg']);
img_list1 = dir ([dir_name, dataset_name,'/*.bmp']);
if(size(img_list1,1)>0)
    img_list = [img_list;img_list1];
end
img_list1 = dir ([dir_name, dataset_name,'/*.tif']);
if(size(img_list1,1)>0)
    img_list = [img_list;img_list1];
end
img_list1 = dir ([dir_name, dataset_name,'/*.ppm']);
if(size(img_list1,1)>0)
    img_list = [img_list;img_list1];
end
img_list1 = dir ([dir_name, dataset_name,'/*.pgm']);
if(size(img_list1,1)>0)
    img_list = [img_list;img_list1];
end
img_list1 = dir ([dir_name, dataset_name,'/*.png']);
if(size(img_list1,1)>0)
    img_list = [img_list;img_list1];
end

if ~ismac&&isunix
    img_list1 = dir ([dir_name, dataset_name,'/*.JPG']);
    if(size(img_list1,1)>0)
       img_list = [img_list;img_list1];
    end
    img_list1 = dir ([dir_name, dataset_name,'/*.BMP']);
    if(size(img_list1,1)>0)
        img_list = [img_list;img_list1];
    end
    img_list1 = dir ([dir_name, dataset_name,'/*.TIF']);
    if(size(img_list1,1)>0)
        img_list = [img_list;img_list1];
    end
    img_list1 = dir ([dir_name, dataset_name,'/*.PPM']);
    if(size(img_list1,1)>0)
        img_list = [img_list;img_list1];
    end
    img_list1 = dir ([dir_name, dataset_name,'/*.PGM']);
    if(size(img_list1,1)>0)
        img_list = [img_list;img_list1];
    end
    img_list1 = dir ([dir_name, dataset_name,'/*.PNG']);
    if(size(img_list1,1)>0)
        img_list = [img_list;img_list1];
    end
end

end
