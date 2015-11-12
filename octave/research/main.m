%Main octave file for research project
%Copywrite 2015 Daniel Weber

printf("Begining Execution, press enter to continue");
pause;
printf("\n");

t1 = getMillis();

for i = 1:3
    a = rand(1000,1000);
    b = rand(1000,1000);
    c = a*b;
end

printf("Time: %f seconds\n", (getMillis()-t1)/1000);

%setup variables
imageChannels = 3;     % number of channels (rgb, so 3)

patchDim   = 8;        % patch dimension
numPatches = 100000;   % number of patches

visibleSize = patchDim * patchDim * imageChannels;  % number of input units 
outputSize  = visibleSize;   % number of output units
hiddenSize  = 400;           % number of hidden units 

sparsityParam = 0.035; % desired average activation of the hidden units.
lambda = 3e-3;         % weight decay parameter       
beta = 5;              % weight of sparsity penalty term 

testGradients();