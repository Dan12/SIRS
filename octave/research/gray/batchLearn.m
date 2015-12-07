function optTheta = batchLearn(visibleSize, hiddenSize, lambda, sparsityParam, beta, patches, theta, h)

	numIters = 400;

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
	%optTheta = repsgd(visibleSize, hiddenSize, lambda, sparsityParam, beta, patches, theta,h);

	%regular sgd
	alpha = 1e-3;
	batchSize = 15;
	numIters = 90000;
	%[optTheta, cost] = sgd(theta, alpha, numIters, visibleSize,hiddenSize,lambda,sparsityParam,beta,patches,batchSize,100,0,h);
	convergeAlpha = .8;
	%these values should converge to alpha of convergeAlpha with random switching
	%		min,    max, pos-add,            neg-mult]
	alrs = [0.00001, 30,  convergeAlpha*0.05, 0.95];
	[optTheta, cost] = sgdALR(theta, alpha, numIters, visibleSize,hiddenSize,lambda,sparsityParam,beta,patches,batchSize,100,0,h,alrs);
endfunction