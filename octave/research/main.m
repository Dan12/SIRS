%Main octave file for research project
%Copywrite 2015 Daniel Weber

printf("Begining Execution, press enter to continue");
pause;
printf("\n");

%t1 = getMillis();

%for i = 1:3
%    a = rand(1000,1000);
%    b = rand(1000,1000);
%    c = a*b;
%end

%printf("Time: %f seconds\n", (getMillis()-t1));

%setup variables
imageChannels = 3;     % number of channels (rgb, so 3)

patchDim   = 8;        % patch dimension
numPatches = 2000;   % number of patches

visibleSize = patchDim * patchDim * imageChannels;  % number of input units 
outputSize  = visibleSize;   % number of output units
hiddenSize  = 400;           % number of hidden units 

sparsityParam = 0.035; % desired average activation of the hidden units.
lambda = 3e-3;         % weight decay parameter       
beta = 5;              % weight of sparsity penalty term 

epsilon = 0.1;	       % epsilon for ZCA whitening

addpath color/;

%testGradients();

load cifar-10-batches-mat/data_batch_1.mat

%figure(1);
%dispcf(3,data',32);

printf("3rd image\n");
pause;

patches = selectPatches(data, patchDim, numPatches);
%range [0-1]
patches = patches./(max(max(patches)));

[patches, ZCAWhite] = ZCAWhiten(patches, numPatches, epsilon);

%figure(2);
%dispcf(3,patches,patchDim);

printf("patches\n");
pause;

%% STEP 2c: Learn features
%  You will now use your sparse autoencoder (with linear decoder) to learn
%  features on the preprocessed patches. This should take around 45 minutes.

theta = initializeParameters(hiddenSize, visibleSize);

% Use minFunc to minimize the function
addpath minFunc/

t1 = getMillis();

%batch
[optTheta, cost] = batchLearn(visibleSize, hiddenSize, lambda, sparsityParam, beta, patches, theta);

%online
%[optTheta, cost] = onlineLearn(visibleSize, hiddenSize, lambda, sparsityParam, beta, data, numPatches, patchDim, theta, ZCAWhite);

printf("Time: %f seconds\n", (getMillis()-t1));

W = reshape(optTheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
b = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
displayColorNetwork( (W*ZCAWhite)');