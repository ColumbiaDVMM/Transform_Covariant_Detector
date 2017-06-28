function point_extractor_2d_NC17(datasets_name, load_feature_name, save_feature_name, point_number)

maxsize = 1024*768;
%Change this to your own vlfeat folder
addpath(genpath('~/tool/vlfeat-0.9.20/toolbox/'));

dir_name = ['../data/'];

if strcmp(datasets_name, 'VggAffineDataset')
    subsets = {'bikes', 'trees', 'graf', 'wall', 'bark', 'boat', 'leuven', 'ubc'};
elseif strcmp(datasets_name, 'EFDataset')
    subsets = {'notredame','obama','paintedladies','rushmore','yosemite'};
elseif strcmp(datasets_name, 'WebcamDataset')
    subsets = {'Chamonix','Courbevoie','Frankfurt','Mexico','Panorama','StLouis'};
end

pyramid_level = 5;
real_scale = 10;

image_list = load_image_list_NC17([dir_name datasets_name '/'], datasets_name);
[s, mess, messid] = mkdir([dir_name datasets_name '/' save_feature_name '/']);

for set_index = 1:numel(subsets)
    subset = subsets{set_index};
    image_list = load_image_list([dir_name 'datasets/' datasets_name '/'], subset);
    [s, mess, messid] = mkdir([dir_name save_feature_name '/' datasets_name '/' subset '/']);
    for i = 1:numel(image_list)
        
        feature = [];
        score = []; 
        
        try
            image = imread([dir_name 'datasets/' datasets_name '/' subset '/' image_list(i).name]);
            x = load([dir_name load_feature_name '/' datasets_name '/' subset '/' image_list(i).name(1:end-4) '.mat']);
        catch
            disp(image_list(i).name);
            save([dir_name save_feature_name '/' datasets_name '/' subset '/' image_list(i).name(1:end-4) '.mat'],'feature','score');
            continue;
        end
    
        if numel(x.output_list) == 0
            disp(image_list(i).name);
            save([dir_name save_feature_name '/' datasets_name '/' subset '/' image_list(i).name(1:end-4) '.mat'],'feature','score');
            continue;
        end
        
        scale = 1.0;
        if size(image,1)*size(image,2)>maxsize
            scale = sqrt(maxsize/(size(image,1)*size(image,2)));
            image = imresize(image,scale);
        end
        
        if size(image,3)>3
            image = image(:,:,1:3);
        end
        
        if size(image,3)==1
            image = repmat(image,1,1,3);
        end
        
        output = x.output_list;
        if numel(output)==0
            disp(image_list(i).name);
            save([dir_name save_feature_name '/' datasets_name '/' subset '/' image_list(i).name(1:end-4) '.mat'],'feature','score');
            continue;
        end
    
        for j = 1:numel(output)
            output{j} = permute(output{j},[3,1,2]);
        end
        clear x;
    
        for p = 1:pyramid_level
            if size(output{p},1)==0 
                break
            end
            output_t = output{p};
            output{p} = zeros(6,size(output{p},2),size(output{p},3));
            output{p}(1,:,:) = 1;
            output{p}(5,:,:) = 1;
            if size(output_t,1)==6
                output{p}(3,:,:) = output_t(3,:,:);
                output{p}(6,:,:) = output_t(6,:,:);
            else
                output{p}(3,:,:) = output_t(1,:,:);
                output{p}(6,:,:) = output_t(2,:,:);
            end
            
            radius_factor = sqrt(2)^(p-1);
            
            outputs = permute(output{p},[2,3,1]);
            outputs(:,:,3) = outputs(:,:,3)*16*2/3;
            outputs(:,:,6) = outputs(:,:,6)*16*2/3;
            offset_x = (size(image,2)/radius_factor-size(outputs,2)*4)/2;
            offset_y = (size(image,1)/radius_factor-size(outputs,1)*4)/2;
    
            output_width = size(outputs,2);
            output_height = size(outputs,1);
            
            grid_x = (1:output_width)';
            grid_x = repmat(grid_x,1,output_height)*4+offset_x;
            grid_x = grid_x';
    
            grid_y = (1:output_height)';
            grid_y = repmat(grid_y,1,output_width)*4+offset_y;
            grid_x = grid_x - outputs(:,:,3);
            grid_y = grid_y - outputs(:,:,6);
            
            vote = zeros(size(grid_x));
            for j = 1:size(grid_x,1)
                for k = 1:size(grid_x,2)
                    index_x = k-ceil(outputs(j,k,3)/4);
                    index_y = j-ceil(outputs(j,k,6)/4);
                    frac_x = ceil(outputs(j,k,3)/4) - outputs(j,k,3)/4;
                    frac_y = ceil(outputs(j,k,6)/4) - outputs(j,k,6)/4;
                    if (index_x)>=1&&(index_x+1)<=output_width&&(index_y)>=1&&(index_y+1)<=output_height
                        vote(index_y+1,index_x+1) = vote(index_y+1,index_x+1)+frac_x*frac_y;
                        vote(index_y+1,index_x) = vote(index_y+1,index_x)+frac_y*(1-frac_x);
                        vote(index_y,index_x+1) = vote(index_y,index_x+1)+(1-frac_y)*frac_x;
                        vote(index_y,index_x) = vote(index_y,index_x)+(1-frac_x)*(1-frac_y);
                    end
                end
            end
            [vote, binary_img] = ApplyNonMax2Score(vote);
            binary_img = binary_img.*(vote>1.2);
    
            vote = reshape(vote,1,output_width*output_height);
            grid_x = reshape(grid_x,1,output_width*output_height);
            grid_y = reshape(grid_y,1,output_width*output_height);
            real_output = reshape(outputs,output_width*output_height,6);
            binary_img = reshape(binary_img,1,output_width*output_height);
         
            vote(~binary_img) = [];
            grid_x(~binary_img) = [];
            grid_y(~binary_img) = [];
            real_output(~binary_img,:) = [];
            
            [~,idx] = sort(vote,'descend');
            
            grid_x = grid_x(idx(1:min(size(idx,2),round(point_number/radius_factor^2))));
            grid_y = grid_y(idx(1:min(size(idx,2),round(point_number/radius_factor^2))));
            real_output = real_output(idx(1:min(size(idx,2),round(point_number/radius_factor^2))),:);
            score_t = vote(idx(1:min(size(idx,2),round(point_number/radius_factor^2))))';
            if(isempty(grid_x))
                continue;
            end
            
            grid_x = grid_x.*radius_factor;
            grid_y = grid_y.*radius_factor;
            real_output = real_output.*radius_factor;
            
            real_output(:,3) = grid_x;
            real_output(:,6) = grid_y;
            feature_t = real_output;
            feature_t(:,[1,2,4,5]) = feature_t(:,[1,2,4,5])*real_scale;
            if(isempty(feature))
                feature = feature_t;
                score = score_t;
            else
                feature = [feature;feature_t];
                score = [score;score_t];
            end
        end
        disp(size(feature))
        save([dir_name save_feature_name '/' datasets_name '/' subset '/' image_list(i).name(1:end-4) '.mat'],'feature','score');
    end
end 
