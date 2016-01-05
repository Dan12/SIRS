function optTheta = groupLearn(visibleSize, hiddenSize, lambda, sparsityParam, beta, patches, theta, h)

	donutDim = floor(sqrt(hiddenSize));
	poolDim = 3;
	poolDimSqrd = poolDim*poolDim-1;

	groupMatrix = zeros(hiddenSize, donutDim, donutDim, poolDimSqrd);

	groupNum = 1;
	for row = 1:donutDim
	    for col = 1:donutDim
	        groupMatrix(groupNum, 1, 2, 1) = -1;
			groupMatrix(groupNum, 1, 3, 2) = -1;
			groupMatrix(groupNum, 2, 1, 3) = -1;
			groupMatrix(groupNum, 2, 2, 4) = -1;
			groupMatrix(groupNum, 2, 3, 5) = -1;
			groupMatrix(groupNum, 3, 1, 6) = -1;
			groupMatrix(groupNum, 3, 2, 7) = -1;
			groupMatrix(groupNum, 3, 3, 8) = -1;
		
	    	groupMatrix(groupNum, 1, 1, :) = ones(1,poolDimSqrd);
	        groupNum = groupNum + 1;
	        groupMatrix = circshift(groupMatrix, [0 -1 0 0]);
	    end
	    groupMatrix = circshift(groupMatrix, [0 0 -1 0]);
	end

	%disp(groupMatrix);

	groupMatrix = reshape(groupMatrix, hiddenSize, hiddenSize, poolDimSqrd);

	%groupMatrix = eye(hiddenSize);

	%disp(groupMatrix);
	%disp(sum(groupMatrix,1));
	%disp(sum(groupMatrix,2));

	numIters = 1000;

	options = struct;
	options.Method = 'lbfgs'; 
	options.maxIter = numIters;
	options.display = 'on';
	option.useMex = 0;

	printf("Ready to learn\n");
	pause;

	optTheta = theta;

	%groupCost(theta, visibleSize, hiddenSize, lambda, sparsityParam,beta, patches, groupMatrix);

	%batch learning
	[optTheta, cost] = minFunc( @(p) groupCost(p, visibleSize, hiddenSize, lambda, sparsityParam,beta, patches, groupMatrix), theta, options, h, visibleSize, hiddenSize);
endfunction