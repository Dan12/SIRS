function theta = initializeParameters(layerSizes)

    % Initialize parameters randomly based on layer sizes.
    sumLayerSizes = 0;
    numLayers = size(layerSizes,2);
    for i = 2:numLayers
         sumLayerSizes = sumLayerSizes + layerSizes(i);
    end
    r  = sqrt(6) / sqrt(sumLayerSizes+1);   % we'll choose weights uniformly from the interval [-r, r]
    
    theta = [];

    for i = 1:numLayers-1
        tempW = rand(layerSizes(i+1),layerSizes(i)) * 2 * r - r;
        theta = [theta ; tempW(:)];
    end
    for i = 1:numLayers-1
        tempB = zeros(layerSizes(i+1),1);
        theta = [theta ; tempB(:)];
    end

    %3 layer case
    %W1 = rand(hiddenSize, visibleSize) * 2 * r - r;
    %W2 = rand(visibleSize, hiddenSize) * 2 * r - r;
    
    %b1 = zeros(hiddenSize, 1);
    %b2 = zeros(visibleSize, 1);
    
    % Convert weights and bias gradients to the vector form.
    % This step will "unroll" (flatten and concatenate together) all 
    % your parameters into a vector, which can then be used with minFunc. 
    %theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];

end