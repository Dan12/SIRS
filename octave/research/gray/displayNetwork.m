function displayNetwork(A)
	
	% display receptive field(s) or basis vector(s) for image patches
	%
	% A         the basis, with patches as column vectors

	% In case the midpoint is not set at 0, we shift it dynamically
    A = A - mean(A(:));

	cols = round(sqrt(size(A, 2)));

	channel_size = size(A,1);
	dim = sqrt(channel_size);
	dimp = dim+1;
	rows = ceil(size(A,2)/cols);

	%split image into three color channels
	B = A(1:channel_size,:);

	%normalize [-1,1]
	B=B./(ones(size(B,1),1)*max(abs(B)));
	%disp(B);
	% Initialization of the image
	I = ones(dim*rows+rows-1,dim*cols+cols-1);

	%Transfer features to this image matrix
	for i=0:rows-1
	  for j=0:cols-1
	      
	    if i*cols+j+1 > size(B, 2)
	        break
	    end
	    
	    % This sets the patch
	    I(i*dimp+1:i*dimp+dim,j*dimp+1:j*dimp+dim,1) = reshape(B(:,i*cols+j+1),[dim dim]);

	  end
	end

	I = I + 1;
	I = I / 2;
	colormap(gray);
	imagesc(I); 
	axis equal
	axis off

endfunction