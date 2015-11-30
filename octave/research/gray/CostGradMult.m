%calculate cost, J, and gradients of weights vecotrized, grad

%theta:vectorized weights
%visibleSize: input vecotor size
%hiddenSize: hidden layer size
%lambda: regularization parameter
%sparsityParam: sparsityParam parameter
%beta: weight of sparsity
%data: n*m matrix of inputs
%n: input parameters
%m: number of training examples

function [cost, grad] = CostGradMult(theta, layerSize, isSparse, ...
                                     lambda, sparsityParam, beta, ...
                                     input, output)


    Weights = {};
    Biases = {};
    WeightGrads = {};
    BiasGrad = {};
    Zs = {};
    As = {};

    numLayers = size(layerSize,2);
    startPos = 1;
    %reshape weights from weight vector
    for i = 1:numLayers-1;
        Weights{i} = reshape(theta(startPos:startPos+layerSize(i+1)*layerSize(i)-1), layerSize(i+1), layerSize(i));
        startPos = startPos + layerSize(i+1)*layerSize(i);
    end
    for i = 1:numLayers-1;
        Biases{i} = theta(startPos:startPos+layerSize(i+1)-1);
        startPos = startPos + layerSize(i+1);
    end
    
    %set m
    m = size(input,2);

    Zs{1} = Weights{1}*input+repmat(Biases{1},1,m);
    As{1} = sigmoid(Zs{1});
    for i = 2:numLayers-1
        Zs{i} = Weights{i}*As{i-1}+repmat(Biases{i},1,m);
        As{i} = sigmoid(Zs{i});
    end
    
    %cost
    %squared error cost
    cost = sum(sum((As{numLayers-1}-output).^2,2),1)./(2*m);
    
    %regularization
    sumWeights = 0;
    for i = 1:numLayers-1
        sumWeights = sumWeights + sum((Weights{i}.^2)(:));
    end
    cost = cost+(lambda/2)*sumWeights;

    %sparsity and sparsity derivatives
    Phats = {};
    SparseDeltas = {};
    for i = 1:numLayers-1
        if isSparse(i) == 1
            Phats{i} = sum(As{i},2)./m;
            SparseDeltas{i} = -sparsityParam./Phats{i}+(1-sparsityParam)./(1-Phats{i});
            cost = cost + beta*sum(sparsityParam*log(sparsityParam./Phats{i})+(1-sparsityParam)*log((1-sparsityParam)./(1-Phats{i})),1);
        else
            Phats{i} = zeros(size(As{i},1),1);
            SparseDeltas{i} = Phats{i};
        endif
    end
    
    %Errors of each weight matrix
    Deltas = {};
    Deltas{numLayers-1} = -(output-As{numLayers-1}-beta*SparseDeltas{numLayers-1}*ones(1,size(As{numLayers-1},2))).*psigmoid(As{numLayers-1});
    for i = numLayers-2:-1:1
        Deltas{i} = (Weights{i+1}'*Deltas{i+1}+beta*SparseDeltas{i}*ones(1,size(As{i},2))).*psigmoid(As{i});
    end

    %gradient of each weight matrix
    WeightGrads{1} = (Deltas{1}*input')./m + lambda*Weights{1};
    BiasGrad{1} = sum(Deltas{1},2)./m;
    for i = 2:numLayers-1
        WeightGrads{i} = (Deltas{i}*As{i-1}')./m + lambda*Weights{i};
        BiasGrad{i} = sum(Deltas{i},2)./m;
    end
    
    %create gradient vector
    grad = WeightGrads{1}(:);
    for i = 2:numLayers-1
        grad = [grad ; WeightGrads{i}(:)];
    end
    for i = 1:numLayers-1
        grad = [grad ; BiasGrad{i}(:)];
    end
    
endfunction

%sigmoid function
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
endfunction

%inverse sigmoid function for z vecotor, inputs should be activation vecotor
function psigm = psigmoid(x)
    psigm = x.*(1-x);
endfunction