%Process Images

function improccess()
	imgdir = "/Users/Danweb/Desktop/sequen/";
	imgprefix = "sequ-";

	numImgs = 2187;
	temp = imread([imgdir imgprefix "0001.png"]);
	disp(size(temp));
	data = zeros(size(temp,1),size(temp,2)*numImgs,size(temp,3));

	for i=1:numImgs
		filename = [imgdir imgprefix getNumbering(i) ".png"];
		%disp(filename);
		data(:,(i-1)*size(temp,2)+1:(i)*size(temp,2),:) = imread(filename);
	end

	%disp(I);

	%imagesc(I);

	data = cast(data,"uint8");

	save imdata.mat data -mat7-binary;

endfunction

function [n] = getNumbering(i)
	n = i;
	if (i/1000 >= 1)
		n = mat2str(i);
	elseif (i/100 >= 1)
		n = ["0" mat2str(i)];
	elseif (i/10 >= 1)
		n = ["00" mat2str(i)];	
	else
		n = ["000" mat2str(i)];
	endif
endfunction