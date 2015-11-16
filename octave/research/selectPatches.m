function patches = selectPatches(data, patchDim, numPatches)
	patches = zeros(patchDim*patchDim*3,numPatches);
	for i=1:numPatches
		rc = ceil(rand()*(32-patchDim)*(32-patchDim));
		r = ceil(rc/(32-patchDim));
		c = mod(rc,(32-patchDim))+1;
		patches(:,i) = reshape(data(i,:,:),32,32,3)(r:r+7,c:c+7,:)(:);
	end
endfunction