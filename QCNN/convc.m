function C = convc(A,B,shape)
%CONVN  N-dimensional clifford convolution.
%   C = CONVN(A, B) performs the N-dimensional clifford convolution of
%   matrices A and B. If nak = size(A,k) and nbk = size(B,k), then
%   size(C,k) = max([nak+nbk-1,nak,nbk]);
%
%   C = CONVN(A, B, 'shape') controls the size of the answer C:
%     'full'   - (default) returns the full N-D convolution
%     'same'   - returns the central part of the convolution that
%                is the same size as A.
%     'valid'  - returns only the part of the result that can be
%                computed without assuming zero-padded arrays.
%                size(C,k) = max([nak-max(0,nbk-1)],0).
%
%   Class support for inputs A,B: clifford
%   Note: The function is waiting for update when the dimension of data is
%   greater than 3. 
%   Copyright: Liu Bin

global clifford_descriptor;

if nargin < 3
  shape = 'full';
end

if isa(A,'clifford')
    A = clifford(A);
end

if isa(B,'clifford')
    B = clifford(B);
end

[A_row,A_col,A_page] = size(A);
sizeA = [A_row,A_col,A_page];
[k_row,k_col] = size(B);
sizeK = [k_row,k_col,1];
if A_row < k_row || A_col < k_col
    fprintf('Error: please checkout the dimention of input data!');
    return;
end

if strcmp(shape,'full')
    pad = clifford(zeros(A_row+2*(k_row-1),A_col+2*(k_col-1),A_page));
    for j = 1:clifford_descriptor.m
        t = zeros(A_row+2*(k_row-1),A_col+2*(k_col-1),A_page);
        t(k_row:A_row+k_row-1,k_col:A_col+k_col-1,:) = part(A, j);%component
        pad = put(pad, j, t);
    end
    kernel_array = repmat(B, 1, 1, A_page); % replicate kernel to all input pages.
    C = clifford(zeros(sizeA+sizeK-1));
    for i = 1 : A_row + k_row - 1
        for j = 1 : A_col + k_col - 1
            tmp_out = sum(sum(pad(i:i+k_row-1,j:j+k_col-1,:).*kernel_array(:,:,:),1),2);
            C(i,j,:) = reshape(squeeze(tmp_out),[1 1 A_page]);
        end
    end
    return;
elseif strcmp(shape,'valid')
    kernel_array = repmat(B, 1, 1, A_page); % replicate kernel to all input pages.
    C = clifford(zeros(sizeA-sizeK+1));
    for i = 1 : A_row - k_row + 1
        for j = 1 : A_col - k_col + 1
            t = A(i:i+k_row-1,j:j+k_col-1,:);
            tmp_out = squeeze(sum(sum( t.* kernel_array, 1 ),2 ));
            C(i, j, :) = reshape(squeeze(tmp_out),[1 1 A_page]);
        end
    end
    return;
else
    pad = zeros(sizeA+sizeK-[0 0 1]);
    pad(ceil(k_row/2):ceil(k_row/2)+A_row-1,ceil(k_col/2):ceil(k_col/2)+A_col-1,:) = A(:,:,:);
    kernel_array = repmat(B, 1, 1, A_page); % replicate kernel to all input pages.
    C = zeros(sizeA);
    for i = 1 : A_row
        for j = 1 : A_col
            tmp_out = squeeze(sum(sum(pad(i:i+k_row-1,j:j+k_col-1,:) .* kernel_array, 1 ),2 ));
            C(i, j, :) = reshape(squeeze(tmp_out),[1 1 A_page]);
        end
    end
end

end
