function optTheta = groupLearn(visibleSize, hiddenSize, lambda, sparsityParam, beta, patches, theta, h)

	donutDim = floor(sqrt(hiddenSize));
	poolDim = 3;

	groupMatrix = zeros(hiddenSize, donutDim, donutDim);

	groupNum = 1;
	for row = 1:donutDim
	    for col = 1:donutDim
	        groupMatrix(groupNum, 1:poolDim, 1:poolDim) = 1;
	        groupNum = groupNum + 1;
	        groupMatrix = circshift(groupMatrix, [0 0 -1]);
	    end
	    groupMatrix = circshift(groupMatrix, [0 -1, 0]);
	end

	groupMatrix = reshape(groupMatrix, hiddenSize, hiddenSize);

	groupMatrix = eye(hiddenSize);

	%disp(groupMatrix);
	%disp(sum(groupMatrix,1));
	%disp(sum(groupMatrix,2));

	numIters = 600;

	options = struct;
	options.Method = 'cg'; 
	options.maxIter = numIters;
	options.display = 'on';
	option.useMex = 0;

	printf("Ready to learn\n");
	pause;

	%optTheta = theta;

	%batch learning
	[optTheta, cost] = minFunc( @(p) groupCost(p, visibleSize, hiddenSize, lambda, sparsityParam,beta, patches, groupMatrix), theta, options, h, visibleSize, hiddenSize);
endfunction