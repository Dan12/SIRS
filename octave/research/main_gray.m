%main grayscale

clear;

printf("Begining Execution, press enter to continue");
pause;
printf("\n");

patchDim   = 16;        % patch dimension
numPatches = 3000;   % number of patches

visibleSize = patchDim * patchDim;  % number of input units 
outputSize  = visibleSize;   % number of output units
hiddenSize  = 400;           % number of hidden units 

sparsityParam = 0.01; % desired average activation of the hidden units.
lambda = 0.0001;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term 

load cifar-10-batches-mat/data_batch_1.mat
data = data';

%load imageprocess/imdata.mat;

%width
imsize1 = 32;
%height
imsize2 = 32;

addpath gray/;

% Use minFunc to minimize the function
addpath minFunc/

%	disp(size(data));
%	figure 1;
	h = dispcf(1,data,imsize1,imsize2); 
	%set image handler
	get(h);
	set(gcf,'doublebuffer','on'); 
	set(h,'erasemode','xor');
	%grayscale data
	data = toGrayScale(data,imsize1,imsize2);
%	figure 2;
%	dispgi(1,data,imsize1,imsize2);
%	disp(size(data));
%	pause;

%testGradients();

theta = initializeParameters(hiddenSize, visibleSize);

patches = selectPatches(data, patchDim, numPatches, imsize1, imsize2);

t1 = getMillis();

optTheta = batchLearn(visibleSize, hiddenSize, lambda, sparsityParam, beta, patches, theta, h);

printf("Time: %f seconds\n", (getMillis()-t1));

W = reshape(optTheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
b = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
displayNetwork(W',h);