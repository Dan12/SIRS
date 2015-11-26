function [theta, J_history] = gradientDescent(theta, alpha, num_iters, ...
                                              visibleSize, hiddenSize, ...
                                              lambda, sparsityParam, ...
                                              beta, data)

    %GRADIENTDESCENT Performs gradient descent to learn theta
    %   theta = GRADIENTDESENT(theta, alpha, num_iters) updates theta by 
    %   taking num_iters gradient steps with learning rate alpha

    % Initialize some useful values
    J_history = zeros(num_iters, 1);

    for iter = 1:num_iters
		
        [cost, grad] = SpLinAeCostGrad(theta, visibleSize, hiddenSize, ...
                                     lambda, sparsityParam, beta, data);

        theta = theta-alpha.*grad;
        
        % Save the cost J in every iteration    
        J_history(iter) = cost;
	
	printf("%d %f\n", iter, cost);

    end

end
