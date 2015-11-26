%main grayscale

clear;

printf("Begining Execution, press enter to continue");
pause;
printf("\n");

imsize = 32;

patchDim   = 16;        % patch dimension
numPatches = 15000;   % number of patches

visibleSize = patchDim * patchDim;  % number of input units 
outputSize  = visibleSize;   % number of output units
hiddenSize  = 64;           % number of hidden units 

sparsityParam = 0.035; % desired average activation of the hidden units.
lambda = 3e-3;         % weight decay parameter       
beta = 5;              % weight of sparsity penalty term 

load cifar-10-batches-mat/data_batch_1.mat

addpath gray/;

% Use minFunc to minimize the function
addpath minFunc/

data = data';

%disp(data);

figure 1;
dispcf(1,data,imsize);
data = toGrayScale(data,imsize);
figure 2;
dispgi(1,data,imsize);

testGradients();

theta = initializeParameters(hiddenSize, visibleSize);

patches = selectPatches(data, patchDim, numPatches, imsize);
patches = normalizeData(patches);

optTheta = batchLearn(visibleSize, hiddenSize, lambda, sparsityParam, beta, patches, theta);

W = reshape(optTheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
b = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
displayNetwork(W');