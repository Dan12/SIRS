function [optTheta, cost] = batchLearn(visibleSize, hiddenSize, lambda, sparsityParam, beta, patches, theta)

numIters = 600;

options = struct;
options.Method = 'lbfgs'; 
options.maxIter = numIters;
options.display = 'on';
option.useMex = 0;

%batch learning
%[optTheta, cost] = minFunc( @(p) SpLinAeCostGrad(p, visibleSize, hiddenSize, lambda, sparsityParam,beta, patches), theta, options);

alpha = 0.1;

[optTheta,cost_hist] = gradientDescent(theta, alpha, numIters, visibleSize,hiddenSize,lambda,sparsityParam,beta,patches);

endfunction