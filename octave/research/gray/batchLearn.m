function optTheta = batchLearn(visibleSize, hiddenSize, lambda, sparsityParam, beta, patches, theta)

	numIters = 600;

	options = struct;
	options.Method = 'lbfgs'; 
	options.maxIter = numIters;
	options.display = 'on';
	option.useMex = 0;

	printf("Ready to learn\n");
	pause;

	%batch learning
	%[optTheta, cost] = minFunc( @(p) SpAeCostGrad(p, visibleSize, hiddenSize, lambda, sparsityParam,beta, patches), theta, options);

	%repetative stochastic gradient descent
	optTheta = repsgd(visibleSize, hiddenSize, lambda, sparsityParam, beta, patches, theta);
endfunction