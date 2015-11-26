function patches = selectPatches(data, patchDim, numPatches, imsize1, imsize2)
	patches = zeros(patchDim*patchDim,numPatches);
	for i=1:numPatches
		rc = ceil(rand()*(imsize1-patchDim)*(imsize2-patchDim));
		r = ceil(rc/(imsize1-patchDim));
		c = mod(rc,(imsize1-patchDim))+1;
		im = ceil(rand(1)*size(data,2));
		patches(:,i) = reshape(data(:,im),imsize2,imsize1)(r:r+patchDim-1,c:c+patchDim-1)(:);
	end
	patches = normalizeData(patches);
endfunction