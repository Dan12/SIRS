function [cost, grad] = convNN(theta, visibleSize, hiddenSize, ...
                               lambda, sparsityParam, beta, data, ...
                               convSize, convLayers)

Convs = zeros(convSize,convSize,convLayers);
Biases = zeros(1,1,convLayers);

m = size(data,2);

startPos = 1;
for i = 1:convLayers
	Convs(:,:,i) = reshape(theta(startPos:startPos+convSize*convSize-1),convSize,convSize);
	startPos += convSize*convSize;
end
%one bias per layer
for i = 1:convLayers
	Biases(:,:,i) = reshape(theta(startPos),convSize,convSize);
	startPos += 1;
end

ConvZs = zeros();
ConvAs = {};



endfunction