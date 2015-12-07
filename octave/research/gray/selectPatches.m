function patches = selectPatches(data, patchDim, numPatches, imsize1, imsize2)
	patches = zeros(patchDim*patchDim,numPatches);
	for i=1:numPatches
		rc = ceil(rand()*(imsize1-patchDim)*(imsize2-patchDim));
		r = ceil(rc/(imsize2-patchDim));
		c = mod(rc,(imsize2-patchDim))+1;
		im = ceil(rand(1)*size(data,2));
		patches(:,i) = (reshape(data(:,im),imsize1,imsize2)(r:r+patchDim-1,c:c+patchDim-1))(:);
		
		if false
			figure 1;
			dispgi(im,data,imsize1,imsize2);
			pause;
			disp(r);
			disp(c);
			figure 2;
			dispgi(i,patches,patchDim,patchDim);
			pause;
		end
	end
	patches = normalizeData(patches);
endfunction