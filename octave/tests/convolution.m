function convimg = convolution(img, patchsize, mconv)
	disp(size(img));
	convimg = zeros(size(img,1)-2,size(img,2)-2,3);
	for i = 1:size(img,1)-2
		for j = 1:size(img,2)-2
			for k = 1:size(img,3)
				convimg(i,j,k) = sum(sum(img(i:i+2,j:j+2,k).*mconv));
			end
		end
	end
	imagesc(img);
	disp("enter to continue");
	pause;
	imagesc(convimg);
	disp("enter to continue");
	pause;
endfunction