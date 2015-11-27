function optTheta = repsgd(visibleSize, hiddenSize, lambda, sparsityParam, beta, patches, theta,h)
	batchSize = 25;

	baseItern = 60;
	baseAlpha = 0.008;
	dispPeriod = 10;

	[optTheta,cost_hist] = sgd(theta, baseAlpha, baseItern, visibleSize,hiddenSize,lambda,sparsityParam,beta,patches,batchSize,dispPeriod,0,h);

	theta = optTheta;

	[optTheta,cost_hist] = sgd(theta, baseAlpha*10, baseItern*10, visibleSize,hiddenSize,lambda,sparsityParam,beta,patches,batchSize,dispPeriod,0,h);

	theta = optTheta;

	[optTheta,cost_hist] = sgd(theta, baseAlpha*100, baseItern*100, visibleSize,hiddenSize,lambda,sparsityParam,beta,patches,batchSize,dispPeriod,0,h);
endfunction