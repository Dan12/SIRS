%main grayscale

clear;

printf("Begining Execution, press enter to continue");
pause;
printf("\n");

patchDim   = 12;        % patch dimension
numPatches = 10000;   % number of patches

visibleSize = patchDim * patchDim;  % number of input units 
outputSize  = visibleSize;   % number of output units
hiddenSize  = 121;           % number of hidden units 

sparsityParam = 0.01; % desired average activation of the hidden units.
lambda = 0.0001;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term 

load cifar-10-batches-mat/data_batch_1.mat
data = data';

%load imageprocess/imdata2.mat;

%width (100; 57 ; 32)
imsize1 = 32;
%height (56; 32 ; 32)
imsize2 = 32;

addpath gray/;

% Use minFunc to minimize the function
addpath minFunc/

%	disp(size(data));
	imnum = ceil(rand(1)*size(data,2));
	figure 1;
	dispcf(imnum,data,imsize1,imsize2);
	%grayscale data
	data = toGrayScale(data,imsize1,imsize2);
	figure 2;
	dispgi(imnum,data,imsize1,imsize2);
%	disp(size(data));
%	pause;

%testGradients();

theta = initializeParameters([visibleSize,hiddenSize,visibleSize]);

patches = selectPatches(data, patchDim, numPatches, imsize1, imsize2);

%dispgi(1,patches,patchDim,patchDim);

printf("View Images\n");
pause;

t1 = getMillis();

h = imagesc(rand(patchDim,patchDim)); 
%set image handler
%get(h);
%set(gcf,'doublebuffer','on'); 
%set(h,'erasemode','xor');
axis square;

%displayNetwork(patches,h);

%optTheta = batchLearn(visibleSize, hiddenSize, lambda, sparsityParam, beta, patches, theta, h);

alpha = 1e-3;
convergeAlpha = 1;
%these values should converge to alpha of convergeAlpha with random switching
%		min,    max, pos-add,            neg-mult]
alrs = [0.00001, 30,  convergeAlpha*0.05, 0.95];

%how many images to train on at once and slider step size in pixels
fLearn = 3;
slideStep = 4;
%iterations per patchsample
iterP = 20;
%optTheta = sequSGD(theta, alpha, visibleSize, hiddenSize, lambda, sparsityParam, beta, data, 100, 1, h, alrs, fLearn, slideStep, imsize1, imsize2, patchDim, iterP);

optTheta = groupLearn(visibleSize, hiddenSize, lambda, sparsityParam, beta, patches, theta, h);

printf("Time: %f seconds\n", (getMillis()-t1));

W = reshape(optTheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
b = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
displayNetwork(W',h);