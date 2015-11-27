function optTheta = batchLearn(visibleSize, hiddenSize, lambda, sparsityParam, beta, patches, theta, h)

	numIters = 800;

	options = struct;
	options.Method = 'lbfgs'; 
	options.maxIter = numIters;
	options.display = 'on';
	option.useMex = 0;

	printf("Ready to learn\n");
	pause;

	%batch learning
	[optTheta, cost] = minFunc( @(p) SpAeCostGrad(p, visibleSize, hiddenSize, lambda, sparsityParam,beta, patches), theta, options);

	%repetative stochastic gradient descent
	%optTheta = repsgd(visibleSize, hiddenSize, lambda, sparsityParam, beta, patches, theta,h);

	%regular sgd
	alpha = .6;
	batchSize = 25;
	numIters = 8000;
	%[optTheta, cost] = sgd(theta, alpha, numIters, visibleSize,hiddenSize,lambda,sparsityParam,beta,patches,batchSize,10,0,h);
endfunction