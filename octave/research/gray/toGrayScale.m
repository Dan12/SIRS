function grayimg = toGrayScale(data, imsize1, imsize2)
	newImSize = imsize1*imsize2;
	a = cast(data(1:newImSize,:),"double");
	b = cast(data(1+newImSize:newImSize*2,:),"double");
	c = cast(data(1+newImSize*2:end,:),"double");
	grayimg = cast((a+b+c)./3,"uint8");
endfunction