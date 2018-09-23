function varargout = size(q, dim)
% SIZE   Size of matrix.
% (Quaternion overloading of standard Matlab function.)

% Copyright (c) 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

% In what follows, we use the size of the x component of the quaternion,
% not the size of the scalar part, since this could be empty. The choice of
% x, y or z is arbitrary, since they must all be the same size - if not the
% error lies in the code that constructed q. Earlier versions of this code
% did check the sizes, but the time taken to do this is significant and the
% check was eliminated in the interests of greater speed.

switch nargout
    case 0
        switch nargin
            case 1
                data = q.x;
                size(data)
            case 2
                data = q.x;
                size(data, dim)
            otherwise
                error('Incorrect number of input arguments.')
        end
    case 1
        switch nargin
            case 1
                data = q.x;
                varargout{1} = size(data);
            case 2
                data = q.x;
                varargout{1} = size(data,dim);
            otherwise
                error('Incorrect number of input arguments.')
        end
    case 2
        switch nargin
            case 1
                data = q.x;
                [varargout{1}, varargout{2}] = size(data);
            case 2
                error('Unknown command option.'); % Note 1.
            otherwise
                error('Incorrect number of input arguments.')
        end
    case 3
        switch nargin
            case 1
                data = q.x;
                [varargout{1}, varargout{2}, varargout{3}] = size(data);
            case 2
                error('Unknown command option.'); % Note 1.
            otherwise
                error('Incorrect number of input arguments.')
        end
    otherwise
        switch nargin
            case 1
                data = q.x;
                d = size(data);      
                for k = 1:length(d)
                    varargout{k} = d(k);
                end
            case 2
                error('Unknown command option.'); % Note 1.                
            otherwise
                error('Incorrect number of input arguments.')
        end
end

% Note 1. Size does not support the calling profile [r, c] = size(q, dim),
% or [d1, d2, d3, .... dn] = size(q, dim). The error raised is the same as
% that raised by the built-in Matlab function.


% $Id: size.m,v 1.7 2016/06/15 13:19:43 sangwine Exp $
