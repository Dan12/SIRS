function dispcf(i, data, imSize)
	a = data(:,i);
	b = reshape(a,imSize,imSize,3);
	b(:,:,1) = flipud(b(:,:,1)');
	b(:,:,2) = flipud(b(:,:,2)');
	b(:,:,3) = flipud(b(:,:,3)');
	imagesc(b);
	axis square;
endfunction