function dispgi(i, data, imSize1,imSize2)
	a = data(:,i);
	b = reshape(a,imSize1,imSize2);
	c = zeros(imSize2,imSize1);
	c(:,:) = flipud(fliplr(b(:,:)'));
	colormap(gray);
	imagesc(c);
	axis square;
endfunction