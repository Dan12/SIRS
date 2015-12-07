%Process Images

%ffmpeg command: ffmpeg -i file_location -vf scale=-1:32,fps=2 -sws_flags 'lanczos' folder_output/sequ-%05d.png

%output image files
function improcess()
	imgdir = "/Users/Danweb/Desktop/Machine Learning/videoSequences/videoSequen4/";
	imgprefix = "sequ-";

	%1-2187 (100x56), 2-482 (100x56), 3-10414 (57x32), 4-14312 (57x32)
	numImgs = 14312;
	temp = imread([imgdir imgprefix "00001.png"]);
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

	save imdata4.mat data -binary;

endfunction

function [n] = getNumbering(i)
	n = i;
	if (i/10000 >= 1)
		n = mat2str(i);
	elseif (i/1000 >= 1)
		n = ["0" mat2str(i)];
	elseif (i/100 >= 1)
		n = ["00" mat2str(i)];
	elseif (i/10 >= 1)
		n = ["000" mat2str(i)];	
	else
		n = ["0000" mat2str(i)];
	endif
endfunction