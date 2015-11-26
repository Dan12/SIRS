function optTheta = batchLearn(visibleSize, hiddenSize, lambda, sparsityParam, beta, patches, theta)

	numIters = 800;

	options = struct;
	options.Method = 'lbfgs'; 
	options.maxIter = numIters;
	options.display = 'on';
	option.useMex = 0;

	printf("Ready to learn\n");
	pause;

	%batch learning
	%[optTheta, cost] = minFunc( @(p) SpAeCostGrad(p, visibleSize, hiddenSize, lambda, sparsityParam,beta, patches), theta, options);

	alpha = .3;

	[optTheta,cost_hist] = gradientDescent(theta, alpha, numIters, visibleSize,hiddenSize,lambda,sparsityParam,beta,patches);

endfunction