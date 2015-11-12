%Test Prject

clear; close all; clc;

fprintf("Delay test, enter to continue \n");

pause;

patchsize = 128;

[i, m, a, r] = testFunction("testimg.jpg", patchsize);

fprintf("Convolutions, press enter to continue\n");
pause;

mconv1 = [-1,-1,-1;
		  -1, 8,-1;
		  -1,-1,-1];

mconv2 = [1/9,1/9,1/9;
		  1/9,1/9,1/9;
		  1/9,1/9,1/9];

mconvegx = [-1/8,0,1/8;
			-1/4,0,1/4;
			-1/8,0,1/8];

mconvegy = mconvegx';

blurimg = convolution(reshape(r(:,1),patchsize,patchsize,3), patchsize, mconv2);
sobelx = convolution(blurimg, patchsize, mconvegx);
sobely = convolution(sobelx, patchsize, mconvegy);
image(mean(sobelx,3));