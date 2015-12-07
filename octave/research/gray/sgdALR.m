function [theta, J_history] = sgdALR(theta, alpha, num_iters, ...
                                              visibleSize, hiddenSize, ...
                                              lambda, sparsityParam, ...
                                              beta, data, batchsize, dispPeriod, draw, h, alrs)

	%Stochastic Gradient Descent Performs gradient descent to learn theta
    %   theta = GRADIENTDESENT(theta, alpha, num_iters) updates theta by 
    %   taking num_iters gradient steps with learning rate alpha
    %	using random selections to form data of size batchsize

    % Initialize some useful values
    J_history = zeros(num_iters, 1);
    alphaG = ones(size(theta))*alpha;
    prevGrad = zeros(size(theta));

    %Adjustable Learing Rate Parameters
    minG = alrs(1);
    maxG = alrs(2);
    add = alrs(3);
    mult = alrs(4);

   	for iter = 1:num_iters

    	randdata = data(:,randi(size(data,2),batchsize,1));
		
        [cost, grad] = SpAeCostGrad(theta, visibleSize, hiddenSize, ...
                                     lambda, sparsityParam, beta, randdata);

        theta = theta-alphaG.*grad;

       	if(isnan(mean(mean(grad))))
       		disp(cost);
        	disp(mean(mean(alphaG)));
        	disp(max(max(alphaG)));
        	disp(min(min(alphaG)));
        	disp(mean(mean(grad)));
        	disp(mean(mean(prevGrad)));
        	disp(max(max(prevGrad)));
        	disp(min(min(prevGrad)));
        	break;
    	endif

        gradSign = sign(prevGrad.*grad);
        gradSignPlus = gradSign;
        gradSign(gradSign < 0) = mult;
        gradSign(gradSign == 0) = 1;
        gradSignPlus(gradSignPlus < 0) = 0;
        alphaG = alphaG + gradSignPlus*add;
        alphaG = alphaG.*gradSign;

        alphaG = max(alphaG,minG);
        alphaG = min(alphaG,maxG);

        prevGrad = grad;
        
        % Save the cost J in every iteration    
        J_history(iter) = cost;
	
		if(mod(iter,dispPeriod) == 0)
        	printf("%d %f\n", iter, cost);
        	if(draw == 1)
	        	W = reshape(theta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
	        	displayNetwork(W',h);
	        	drawnow;
        	endif
            fflush(stdout);
        endif

    end


endfunction