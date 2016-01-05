function [theta] = sequSGD(theta, alpha, ...
                           visibleSize, hiddenSize, ...
                           lambda, sparsityParam, ...
                           beta, data, dispPeriod, draw, h, alrs, ...
                           fLearn, slideStep, imSize1, imSize2, patchDim, iterP)


	% Initialize some useful values
    alphaG = ones(size(theta))*alpha;
    prevGrad = zeros(size(theta));

    %Adjustable Learing Rate Parameters
    minG = alrs(1);
    maxG = alrs(2);
    add = alrs(3);
    mult = alrs(4);

    rIterns = floor((imSize1-patchDim)/slideStep)+1;
    cIterns = floor((imSize2-patchDim)/slideStep)+1;
    rStart = floor((imSize1-(rIterns-1)*slideStep-patchDim)/2)+1;
    cStart = floor((imSize2-(cIterns-1)*slideStep-patchDim)/2)+1;

    num_iters = floor(size(data,2)/fLearn)-1;

    disp(size(data));
    disp(rIterns);
    disp(cIterns);
    disp(rStart);
    disp(cStart);
    disp(fLearn*rIterns*cIterns);

  	J_history = zeros(num_iters, 1);

   	for iter = 1:num_iters

   		%select patches
   		patches = zeros(patchDim*patchDim,fLearn*rIterns*cIterns);

    	for pNum = 1:fLearn
    		img = reshape(data(:,iter*fLearn+pNum),imSize1,imSize2);
    		for r = 1:rIterns
    			for c = 1:cIterns
    				%disp((pNum-1)*cIterns*rIterns + (r-1)*cIterns + c);
    				patches(:,(pNum-1)*cIterns*rIterns + (r-1)*cIterns + c) = fliplr(rot90(img(rStart+(r-1)*slideStep:rStart+(r-1)*slideStep-1+patchDim,cStart+(c-1)*slideStep:cStart+(c-1)*slideStep-1+patchDim)))(:);
				end
			end
		end
		patches = normalizeData(patches);

		%disp(size(patches));
		%displayNetwork(patches,h);
		%drawnow;
		%pause;
		
		for i = 1:iterP

	        [cost, grad] = SpAeCostGrad(theta, visibleSize, hiddenSize, ...
	                                     lambda, sparsityParam, beta, patches);

	    	%if something goes wrong
	       	if(isnan(mean(mean(grad))))
	       		printf("Grad is NaN");
	       		disp(mean(mean(theta)));
	       		disp(mean(mean(patches)));
	       		printf("%f %f %f %f %f", visibleSize, hiddenSize, lambda, sparsityParam, beta);
	       		pause;
	        	break;
	    	endif

	        theta = theta-alphaG.*grad;

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
		
			if(mod(iter,dispPeriod) == 0 && i == iterP)
	        	printf("%d %d %f\n", iter, i, cost);
	        	if(draw == 1)
		        	W = reshape(theta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
		        	%figure 1;
		        	%displayNetwork(patches);
		        	%figure 2;
		        	displayNetwork(W',h);
		        	drawnow;
		        	%printf("Stop");
		            imageOutName = ["media/tempImgs/img-" mat2str(iter/dispPeriod) ".png"];
		            print(imageOutName,"-dpng");
		            %pause;
	        	endif
	            fflush(stdout);
	        endif

        end

    end

endfunction