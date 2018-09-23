function C = convnq(A,B,shape)
%CONVN  N-dimensional quaternion convolution.
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

if nargin < 3
  shape = 'full';
end

if isa(A,'quaternion')
    A = quaternion(A);
end

if isa(B,'quaternion')
    B = quaternion(B);
end

[Ar,Ac,Ap] = size(A);
sizeA = [Ar,Ac,Ap];
if length(size(B)) == 2
    [kr,kc] = size(B);
    kp = 1;
    sizeK = [kr,kc,kp];
    kernel_array = repmat(B, 1, 1, Ap); % replicate kernel to all input pages.
else
    [kr,kc,kp] = size(B);
    sizeK = [kr,kc,kp];
    kernel_array = B;
end    
    
if Ar < kr || Ac < kc || Ap < kp
    fprintf('Error: please checkout the dimention of input data!');
    return;
end

if strcmp(shape,'full')
    pad = zerosq(Ar + 2*(kr - 1),Ac + 2*(kc - 1),Ap);
    T = substruct('()', {kr:Ar+kr-1,kc:Ac+kc-1,1:size(pad,3)});
    pad = subsasgn(pad, T, A);
    C = zerosq(Ar+kr-1,Ac-kc+1,Ap);
    for i = 1 : Ar + kr - 1
        for j = 1 : Ac + kc - 1
            T = substruct('()', {i:i+kr-1,j:j+kc-1,1:size(pad,3)});
            tmp_pad = subsref(pad, T);
%             tmp_out = sum(sum(tmp_pad.*kernel_array,1),2);    %%权重左乘
            tmp_out = sum(sum(kernel_array.*tmp_pad,1),2);  %%权重右乘
            tmp_out2 = reshape(squeeze(tmp_out),[1 1 Ap]);
            indexc = substruct('()',{i,j,1:size(C,3)});
            C = subsasgn(C,indexc,tmp_out2);
        end
    end
    return;
elseif strcmp(shape,'valid')
    C = zerosq(Ar-kr+1,Ac-kc+1,Ap);
    for i = 1 : Ar - kr + 1
        for j = 1 : Ac - kc + 1            
            T = substruct('()', {i:i+kr-1,j:j+kc-1,1:size(A,3)});
            t = subsref(A, T);
%             tmp_out = squeeze(sum(sum( t.* kernel_array, 1 ),2 ));    %%权重左乘
            tmp_out = squeeze(sum(sum( kernel_array.*t , 1 ),2 ));    %%权重右乘
            tmp_out2 = reshape(squeeze(tmp_out),[1 1 Ap]);
            indexc = substruct('()',{i,j,1:size(C,3)});
            C = subsasgn(C,indexc,tmp_out2);
        end
    end
    return;
else     %  Care for:   The followed program maybe accure fault!
    pad = zerosq(sizeA);     
    T = substruct('()', {ceil(kr/2):ceil(kr/2)+Ar-1,ceil(kc/2):ceil(kc/2)+Ac-1,1:Ap});
    pad = subsasgn(pad, T, A);
    C = zerosq(sizeA);
    for i = 1 : Ar
        for j = 1 : Ac
            tmp_out = squeeze(sum(sum(pad(i:i+kr-1,j:j+kc-1,:) .* kernel_array, 1 ),2 ));
            tmp_out2 = reshape(squeeze(tmp_out),[1 1 Ap]);
            indexc = substruct('()',{i,j,1:size(C,3)});
            C = subsasgn(C,indexc,tmp_out2);
        end
    end
end

end
