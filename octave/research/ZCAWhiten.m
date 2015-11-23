function [patches, ZCAWhite] = ZCAWhiten(patches, numPatches, epsilon)
	% Subtract mean patch (hence zeroing the mean of the patches)
	meanPatch = mean(patches, 2);  
	patches = bsxfun(@minus, patches, meanPatch);

	% Apply ZCA whitening
	sigma = patches * patches' / numPatches;
	[u, s, v] = svd(sigma);
	ZCAWhite = u * diag(1 ./ sqrt(diag(s) + epsilon)) * u';
	%patches = ZCAWhite * patches;
endfunction