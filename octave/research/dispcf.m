function dispcf(i, data, imSize1, imSize2)
	a = data(:,i);
	b = reshape(a,imSize1,imSize2,3);
	c = zeros(imSize2,imSize1,3);
	c(:,:,1) = flipud(fliplr(b(:,:,1)'));
	c(:,:,2) = flipud(fliplr(b(:,:,2)'));
	c(:,:,3) = flipud(fliplr(b(:,:,3)'));
	imagesc(c,[0,255]);
	axis square;
endfunction