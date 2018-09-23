function [ B ] = normer( A )
% NORM   Each element's norm of Matrix or vector .
% (Quaternion overloading of standard Matlab function, with some limitations.)

% Copyright (c) 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

narginchk(1, 1), nargoutchk(0, 1) 

% We have to handle two different cases, depending on whether A is a vector
% or a matrix. The limiting case of a vector (a single quaternion) is handled
% using the vector case.

dim = length(size(A));

if dim == 1
    B = norm(A);
elseif dim == 2
    B = zeros(size(A));
    for i = 1 : size(A,1)
        for j = 1 : size(A,2)
            B(i,j) = norm(A(i,j));
        end
    end
elseif dim == 3
    B = zeros(size(A));
    for i = 1 : size(A,1)
        for j = 1 : size(A,2)
            for k = 1 : size(A,3)
                B(i,j,k) = norm(A(i,j,k));
            end
        end
    end
else
    B = zeros(size(A));
    p = zeros(dim);
    for i = dim
        p(i) = size(A,dim);
    end
    for i = 1 : dim
        for j = 1 : p(i)
            B(i,j) = A(i,j);
            disp(' The function is waiting for update!');
        end
    end
end
end

