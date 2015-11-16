function [optTheta, cost] = onlineLearn(visibleSize, hiddenSize, ...
									   lambda, sparsityParam, ...
									   beta, data, numPatches, patchDim, ...
									   theta)

alpha = .1;

interns = 20;

options = struct;
options.Method = 'lbfgs'; 
options.maxIter = interns;
options.display = 'on';
option.useMex = 0;

rowLen = patchDim*patchDim*3;
colLen = size(data,2)/(patchDim*patchDim*3);
rows = sqrt(colLen);

epsilon = 0.1;

batchSize = 200;

for i = 1:numPatches/batchSize
	printf("starting\n");

	patches = zeros(rowLen,colLen*batchSize);
	for k = 1:batchSize
		curImg = reshape(data((i-1)*batchSize+k,:,:),32,32,3);
		for j = 1:colLen
			r = ceil(colLen/rows);
			c = mod(colLen,rows)+1;
			patches(:,(k-1)*batchSize+j) = curImg((r-1)*patchDim+1:r*patchDim,(c-1)*patchDim+1:c*patchDim,:)(:);
		end
	end

	printf("patches1\n");

	patches = patches./(max(max(patches)));

	[patches, ZCAWhite] = ZCAWhiten(patches, colLen*batchSize, epsilon);

	printf("patches2\n");

	[optTheta, cost] = minFunc( @(p) SpLinAeCostGrad(p, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches), theta, options);

	%[optTheta, cost] = gradientDescent(theta, alpha, interns, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches);

	theta = optTheta;
	%disp(cost);
	printf("--------%d %f-------\n",i,cost(size(cost,1)));
end
	
endfunction