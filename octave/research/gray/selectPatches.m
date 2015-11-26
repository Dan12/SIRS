function patches = selectPatches(data, patchDim, numPatches, imsize)
	patches = zeros(patchDim*patchDim,numPatches);
	for i=1:numPatches
		rc = ceil(rand()*(imsize-patchDim)*(imsize-patchDim));
		r = ceil(rc/(imsize-patchDim));
		c = mod(rc,(imsize-patchDim))+1;
		im = ceil(rand(1)*size(data,2));
		patches(:,i) = reshape(data(:,im),imsize,imsize)(r:r+patchDim-1,c:c+patchDim-1)(:);
	end
endfunction