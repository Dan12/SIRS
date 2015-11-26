function [theta, J_history] = sgd(theta, alpha, num_iters, ...
                                              visibleSize, hiddenSize, ...
                                              lambda, sparsityParam, ...
                                              beta, data, batchsize, dispPeriod)

	%Stochastic Gradient Descent Performs gradient descent to learn theta
    %   theta = GRADIENTDESENT(theta, alpha, num_iters) updates theta by 
    %   taking num_iters gradient steps with learning rate alpha
    %	using random selections to form data of size batchsize

    % Initialize some useful values
    J_history = zeros(num_iters, 1);

    for iter = 1:num_iters

    	randdata = data(:,randi(size(data,2),batchsize,1));
		
        [cost, grad] = SpAeCostGrad(theta, visibleSize, hiddenSize, ...
                                     lambda, sparsityParam, beta, randdata);

        theta = theta-alpha.*grad;
        
        % Save the cost J in every iteration    
        J_history(iter) = cost;
	
		if(mod(iter,dispPeriod) == 0)
        	printf("%d %f\n", iter, cost);
        endif

    end


endfunction