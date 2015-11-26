%Process Images

%100*56 image files
function improcess()
	imgdir = "/Users/Danweb/Desktop/Machine Learning/videoSequen/";
	imgprefix = "sequ-";

	numImgs = 2187;
	temp = imread([imgdir imgprefix "0001.png"]);
	disp(size(temp));
	data = zeros(size(temp,1)*size(temp,2)*size(temp,3),numImgs);

	temp2 = zeros(size(temp,2),size(temp,1),size(temp,3));

	for i=1:numImgs
		filename = [imgdir imgprefix getNumbering(i) ".png"];
		%disp(filename);
		temp3 = imread(filename);
		temp2(:,:,1) = flipud(temp3(:,:,1)');
		temp2(:,:,2) = flipud(temp3(:,:,2)');
		temp2(:,:,3) = flipud(temp3(:,:,3)');
		data(:,i) = temp2(:);
	end

	%disp(I);

	%imagesc(I);

	data = cast(data,"uint8");

	save imdata.mat data -binary;

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