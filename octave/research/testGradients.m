%setup dummy network to test gradients
function testGradients()
    testLambda = 3e-3;
    testSparityParam = 0.035;
    testBeta = 5;

    debugHiddenSize = 10;
    debugvisibleSize = 20;
    debugM = 10;
    patches = rand(debugvisibleSize, debugM);
    theta = initializeParameters(debugHiddenSize, debugvisibleSize);
    
    [cost, grad] = SpAeXECostGrad(theta, debugvisibleSize, debugHiddenSize, ...
                                               testLambda, testSparityParam, testBeta, ...
                                               patches);
   
    %compute numerical gradients, reference cost/gradient function
    numGrad = computeNumericalGradient( @(x) SpAeXECostGrad(x, debugvisibleSize, debugHiddenSize, ...
                                                  testLambda, testSparityParam, testBeta, ...
                                                  patches), theta);
                                                  
    %compare gradients
    printf("Columns of numerical Gradient and Backprop Gradient (should be similar)");
    disp([numGrad grad]); 
    
    %compute difference between caluculated and approximated
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    
    printf('%f\nNorm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n', diff);
    
    printf("Press enter to view ideal image for theta");
    pause;
    
    %disp(calcOptImage(reshape(theta(1:debugHiddenSize*debugvisibleSize),debugHiddenSize,debugvisibleSize)));
endfunction