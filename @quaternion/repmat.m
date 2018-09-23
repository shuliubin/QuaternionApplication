function b = repmat(a, m, n,k)
% REPMAT Replicate and tile an array.
% (Quaternion overloading of standard Matlab function.)

% Copyright (c) 2005, 2008 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

narginchk(2, 4), nargoutchk(0, 1) 

if nargin == 2
    b = overload(mfilename, a, m);
elseif nargin == 3
    b = overload(mfilename, a, m, n);
elseif nargin == 4
    b = overload(mfilename, a, m, n, k);
else
    disp('The function only support the parameters less 4')
end

% $Id: repmat.m,v 1.6 2016/06/15 13:19:43 sangwine Exp $

