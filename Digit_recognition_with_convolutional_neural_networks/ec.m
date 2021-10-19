% read the images and recognize the handwritten numbers.
layers = get_lenet();
load lenet.mat

for k = 1: 4
    % loading the images in order
    img_i = sprintf('../images/image%d.jpg', k);
    img_i = rgb2gray(imread(img_i));
    img_i = imcomplement(img_i); % grey-scale
    img_i = imbinarize(img_i);   % thresholding
    label = bwlabel(img_i,8);  % for assigning the labels for different parts
    se = strel('line',1,1);
    connected = bwconncomp(img_i).NumObjects; % connected digits  
    
    for i = 1: connected
        if k == 1           % image 1
            padding = 100;
        elseif k == 2       % image 2
            padding = 100;
        elseif k == 3       % image 3
            padding = 100;
        elseif k == 4       % image 4
            padding = 10;
        end    
        [temp1,temp2] = find(label == i);
        digit = [temp1 temp2];
        total = size(digit,1);
        
        counter1 = max(digit(:,1))-min(digit(:,1));
        counter2 = max(digit(:,2))-min(digit(:,2));
        
        if counter1 > counter2  % In the new dimension,choose the max side                                                   
            created_digit = zeros([counter1 + padding , counter1 + padding]);
            row = round(size(created_digit,1)/2 - counter1/2);
            col = round(size(created_digit,2)/2 - counter2/2);
        else
            created_digit = zeros([counter2 + padding , counter2 + padding]);
            row = round(size(created_digit,1)/2 - counter1/2);
            col = round(size(created_digit,2)/2 - counter2/2);
        end
        
        for j = 1:total    % plot all digits to new dimension                                               
            row_index = digit(j,1)-min(digit(:,1))+1;
            col_index = digit(j,2)-min(digit(:,2))+1;
            created_digit(row_index + row, col_index + col) = 1;
        end
        
        img_i = imresize(created_digit,[28 28]); % resize to 28 * 28

        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
% For this part, please comment the other k values for each image
        if k == 1               % for image 1
            subplot(1,10,i);
%         if k == 2               % for image 2
%             subplot(1,10,i);
%         if k == 3             % for image 3
%             subplot(1,5,i);
%         if k == 4               % for image 4
%             subplot(5,10,i);
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      

        imshow(img_i)
     
        img_i = img_i'; % transpose the prediction                                                     
        img_i = reshape(img_i, 28*28, 1);
        % passing the data via the network
        [output, P] = convnet_forward(params, layers, img_i);
        [np,count] = max(P);
        output = count - 1;
        title([num2str(output)])   
    end
end
