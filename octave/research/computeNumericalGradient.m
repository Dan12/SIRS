%calculate gradients of theta using epsilon and cost function J
%this is an approximation of the gradient to check the
%implementation of backpropegation
function numgrad = computeNumericalGradient(J, theta)
    
    %set numgrad to size theta
    numgrad = zeros(size(theta));
    %set perturb to size theta
    perturb = zeros(size(theta));
    %set epsilon
    e = 1e-4;
    
    %go through each weight and calculate difference in cost for 
    %small pertubation in either direction
    for p = 1:numel(theta)
        % Set perturbation matrix
        perturb(p) = e;
        loss1 = J(theta - perturb);
        loss2 = J(theta + perturb);
        % Compute Numerical Gradient
        numgrad(p) = (loss2 - loss1) / (2*e);
        perturb(p) = 0;
    end
    
endfunction