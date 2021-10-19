layers = get_lenet();
load lenet.mat
% load data
% Change the following value to true to load the entire dataset.
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);
xtrain = [xtrain, xvalidate];
ytrain = [ytrain, yvalidate];
m_train = size(xtrain, 2);
batch_size = 64;
 
layers{1}.batch_size = 1;
img = xtest(:, 1);
img = reshape(img, 28, 28);
imshow(img')
figure
 
% [cp, ~, output] = conv_net_output(params, layers, xtest(:, 1), ytest(:, 1));
output = convnet_forward(params, layers, xtest(:, 1));
output_1 = reshape(output{1}.data, 28, 28);

layer_normal = output{1,3}.data ; %ReLu
layer_normal = (layer_normal-min(output{1,2}.data(:,1)))/(max(output{1,2}.data(:,1))-min(output{1,2}.data(:,1))); %normalization

layer_2 = reshape(output{1,2}.data(:,1),24,24,20) ; % CONV layer
layer_3 = reshape(output{1,3}.data(:,1),24,24,20) ; % ReLU layer 
layer_norm = reshape(layer_normal(:,1),24,24,20);  % normalizing ReLU layer

for i = 1:20             % CONV layer
    subplot(4,5,i) ;
    imshow(layer_2(:,:,i)')    
end
figure

for j = 1:20             % ReLU layer
    subplot(4,5,j) ;
    imshow(layer_3(:,:,j)')   
end
figure

for k = 1:20         % normalized for ReLU layer    
    subplot(4,5,k) ;   
    imshow(layer_norm(:,:,k)')
end