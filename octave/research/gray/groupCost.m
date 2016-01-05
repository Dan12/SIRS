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

function [cost, grad] = groupCost(theta, visibleSize, hiddenSize, ...
                                     lambda, sparsityParam, beta, data, groupMatrix)

    %reshape theta vecotor into weight vectors
    %[hiddenSize,visibleSize]
    W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
    %[visibleSize,hiddenSize]
    W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
    %[hiddenSize,1]
    b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
    %[visibleSize,1]
    b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
    
    %set m
    m = size(data,2);
    
    %calculate activation vector of hidden layer (l2)
    z2 = W1*data+repmat(b1,1,m);
    a2 = sigmoid(z2);
    a2 = constrA(a2);
    
    %calculate activation vector of output layer (l3)
    z3 = W2*a2+repmat(b2,1,m);
    a3 = sigmoid(z3);
    a3 = constrA(a3);
    
    %cost
    %squared error cost
    cost = sum(sum((a3-data).^2,2),1)./(2*m);
    
    %regularization
    cost = cost+(lambda/2)*(sum((W1.^2)(:))+sum((W2.^2)(:)));

    %sparsity
    phat = sum(a2,2)./m;
    cost = cost+beta*sum(sparsityParam*log(sparsityParam./phat)+(1-sparsityParam)*log((1-sparsityParam)./(1-phat)),1);
    
    %grLambda = 5e-6;
    grLambda = 0;
    %grouping
    %disp(W1);
    groupError = zeros(size(W1,1),size(W1,2),size(groupMatrix,3));
    %disp(size(groupError));
    for i = 1:size(groupMatrix,3)
        groupError(:,:,i) = groupMatrix(:,:,i)*W1;
    end
    %disp(groupError);

    %disp(groupError); 
    cost = cost+(grLambda/2)*sum((groupError.^2)(:));
    %grouping gradient
    groupGrad = 2*grLambda*sum(groupError,3);

    %gradients
    %sparsity partial derivative
    sparstity_delta = -sparsityParam./phat+(1-sparsityParam)./(1-phat);
    
    %error of outputs
    delta3 = -(data-a3).*psigmoid(a3);
    
    %error of hidden layer
    delta2 = (W2'*delta3+beta*sparstity_delta*ones(1,size(z2,2))).*psigmoid(a2);
    
    %gradient of W1
    W1grad = (delta2*data')./m + lambda*W1+groupGrad;
    
    %gradient of b1
    b1grad = sum(delta2,2)./m;
    
    %gradient of W2
    W2grad = (delta3*a2')./m + lambda*W2;
    
    %gradient of b2
    b2grad = sum(delta3,2)./m;
    
    %create gradient vector
    grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
    
endfunction

%sigmoid function
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
endfunction

%inverse sigmoid function for z vecotor, inputs should be activation vecotor
function psigm = psigmoid(x)
    psigm = x.*(1-x);
endfunction

function a = constrA(x)
    a = max(x,1e-305);
endfunction