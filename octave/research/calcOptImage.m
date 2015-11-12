function optImg = calcOptImage(theta)
    optImg = theta./(sum(theta.^2,2)*ones(1,size(theta,2)));
endfunction