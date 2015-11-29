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
	alpha = .001;
	batchSize = 5;
	numIters = 50000;
	%these values should converge to alpha of 0.7 with random switching
	%		min,   max, pos-add,  neg-mult]
	alrs = [0.0001,10,  0.3*0.05, .95];
	%[optTheta, cost] = sgd(theta, alpha, numIters, visibleSize,hiddenSize,lambda,sparsityParam,beta,patches,batchSize,100,0,h);
	[optTheta, cost] = sgdALR(theta, alpha, numIters, visibleSize,hiddenSize,lambda,sparsityParam,beta,patches,batchSize,1,0,h,alrs);
endfunction