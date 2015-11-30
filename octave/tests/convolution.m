function convimg = convolution(img, patchsize, mconv)
	disp(size(img));
	convimg = convn(img,mconv,'valid');
	imagesc(img);
	disp("enter to continue");
	pause;
	imagesc(convimg);
	disp("enter to continue");
	pause;
endfunction