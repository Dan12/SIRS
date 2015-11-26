function dispgi(i, data, imSize1,imSize2)
	a = data(:,i);
	b = reshape(a,imSize1,imSize2);
	c = zeros(imSize2,imSize1);
	c(:,:) = flipud(b(:,:)');
	colormap(gray);
	imagesc(c,[0 255]);
	axis square;
endfunction