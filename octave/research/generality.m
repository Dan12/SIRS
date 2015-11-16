function generality()
	%variables
	patchDim = 8;
	numPatches = 1000;
	epsilon = 0.1;
	alpha = 30;
	visibleSize = patchDim * patchDim * 3;  % number of input units 
	
	%load dataset
	load cifar-10-batches-mat/data_batch_1.mat
	
	
	patches = selectPatches(data, patchDim, numPatches);
	%patches = toGrayScale(patches);	
	%range [0-1]
	maxPatches = max(max(patches));
	patches = patches./maxPatches;
	
	[patches, ZCAWhite] = ZCAWhiten(patches, numPatches, epsilon);
	
	%subtract mean
	%meanPatch = mean(patches, 2);  
	%patches = bsxfun(@minus, patches, meanPatch);		


	maxPatches = max(max(patches));
	minPatches = min(min(patches));
	
	r  = sqrt(6) / sqrt(400+visibleSize+1);
	
	theta = rand(4, visibleSize) * 2 * r - r;
	disp(theta);
	printf("Theta, enter to cont.\n");
	printf("%f %f %f\n",maxPatches,minPatches,max(max(theta)));
	pause;randPerm = randperm(numPatches);
	for i=1:numPatches
		z = theta*patches(:,randPerm(i));
		a = round(sigmoid(z));
		if(a == 1)
			t = sigmoid(theta);
			d = (a*ones(1,size(t,2))-t).*(alpha/i);
			printf("%d %f %f %f\n",i,z,a,d(1));
			%displayColorNetwork((theta*ZCAWhite)');
			theta = theta+d;
			figure(1);
			%dispcf(patches(:,i),patchDim);
			figure(2);
			%display_network((theta*ZCAWhite)');
			%pause;
		else
			printf(" %d ",i);
		end
	end
	printf("Iterns");
	pause;
	disp(theta);
	displayColorNetwork((theta*ZCAWhite)');
	%display_network((theta*ZCAWhite)');
endfunction

%sigmoid function
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
endfunction

%grayscale
function gray = toGrayScale(x)
	a = size(x,1)/3;
	b = x(1:a,:);
	c = x(a+1:a*2,:);
	d = x(a*2+1:end,:);
	gray = (b+c+d)./3;
endfunction

function dispgf(patch, imSize)
	a = patch;
	b = reshape(a,imSize,imSize);
	b(:,:) = flipud(b(:,:)');
	imagesc(b);
	axis square;
endfunction

function dispcf(patch, imSize)
	a = patch;
	b = reshape(a,imSize,imSize,3);
	b(:,:,1) = flipud(b(:,:,1)');
	b(:,:,2) = flipud(b(:,:,2)');
	b(:,:,3) = flipud(b(:,:,3)');
	imagesc(b);
	axis square;
endfunction

function initCase(theta,patches)
	for i = 1:size(theta,1)
		
	end
endfunction