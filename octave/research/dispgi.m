function dispgi(i, data, imSize)
	a = data(:,i);
	b = reshape(a,imSize,imSize);
	b(:,:) = flipud(b(:,:)');
	colormap(gray);
	imagesc(b,[0 255]);
	axis square;
endfunction