%main grayscale

clear;

printf("Begining Execution, press enter to continue");
pause;
printf("\n");

%width
imsize1 = 32;
%height
imsize2 = 32;

patchDim   = 16;        % patch dimension
numPatches = 6000;   % number of patches

visibleSize = patchDim * patchDim;  % number of input units 
outputSize  = visibleSize;   % number of output units
hiddenSize  = 400;           % number of hidden units 

sparsityParam = 0.01; % desired average activation of the hidden units.
lambda = 0.0001;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term 

load cifar-10-batches-mat/data_batch_1.mat
data = data';

addpath gray/;

% Use minFunc to minimize the function
addpath minFunc/

%load imageprocess/imdata.mat;

%	disp(size(data));
%	figure 1;
%	dispcf(1,data,imsize1,imsize2);
	data = toGrayScale(data,imsize1,imsize2);
%	figure 2;
%	dispgi(1,data,imsize1,imsize2);
%	disp(size(data));
%	pause;

%testGradients();

theta = initializeParameters(hiddenSize, visibleSize);

patches = selectPatches(data, patchDim, numPatches, imsize1, imsize2);

t1 = getMillis();

optTheta = batchLearn(visibleSize, hiddenSize, lambda, sparsityParam, beta, patches, theta);

printf("Time: %f seconds\n", (getMillis()-t1));

W = reshape(optTheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
b = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
displayNetwork(W');