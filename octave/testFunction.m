function [img, map, alpha, imgrgb] = testFunction(imgsrc)
	[img, map, alpha] = imread(imgsrc);
	imgrgb = otherFunc(img, map, alpha);
endfunction

function retimgArr = otherFunc(img, map, alpha)
	fprintf("%f \n %f \n %f \n", size(img), size(map), size(alpha));
	%rgb colors
	%fprintf("%f \n", img(1:10, end-10:end, 1:3));
	retimgArr = 0;
	image(img);
	%order matters, ij not working
	axis("equal", "ij");
	pt = getPatches(img);
	rebuild(pt);
endfunction

function patches = getPatches(img)
	patchsize = 32;
	patches = zeros(patchsize*patchsize*3,1);
	actr = (size(img,1)-patchsize);
	actc = (size(img,2)-patchsize);
	rand = randperm(actr*actc);

	for i = 1:10
		r = floor(rand(i)/actr);
		c = mod(rand(i), actc);
		tempPatch = img(r:r+patchsize-1,c:c+patchsize-1,:)(:);
		patches = [tempPatch patches];
	end
endfunction

function rebuild(patches)
	patchsize = 32;
	img = zeros(3*patchsize,3*patchsize,3);
	%fprintf("%f\n", patches);
	fprintf("press enter to continue");
	pause;

	for i = 1:3
		for j = 1:3
			fprintf("%f %f\n", i, j);
			img(i*patchsize-patchsize+1:i*patchsize,j*patchsize-patchsize+1:j*patchsize,:) = reshape(patches(:,(i-1)*3+j),patchsize,patchsize,3)
		end
	end
	imagesc(img);
endfunction