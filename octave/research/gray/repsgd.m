function optTheta = repsgd(visibleSize, hiddenSize, lambda, sparsityParam, beta, patches, theta,h)
	batchSize = 200;

	baseItern = 1000;
	baseAlpha = 0.008;
	dispPeriod = 100;
	dispNeurons = 1;

	[optTheta,cost_hist] = sgd(theta, baseAlpha, baseItern, visibleSize,hiddenSize,lambda,sparsityParam,beta,patches,batchSize,dispPeriod,dispNeurons,h);

	theta = optTheta;

	[optTheta,cost_hist] = sgd(theta, baseAlpha*10, baseItern*10, visibleSize,hiddenSize,lambda,sparsityParam,beta,patches,batchSize,dispPeriod,dispNeurons,h);

	theta = optTheta;

	[optTheta,cost_hist] = sgd(theta, baseAlpha*100, baseItern*100, visibleSize,hiddenSize,lambda,sparsityParam,beta,patches,batchSize,dispPeriod,dispNeurons,h);
endfunction