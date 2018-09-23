function tf = isreal(A)
% ISREAL True for real (quaternion) array.
% (Quaternion overloading of standard Matlab function.)

% Copyright (c) 2005, 2010 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

% This function returns true if all the components of A are real, that is,
% A is a quaternion with real coefficients (a real quaternion).

narginchk(1, 1), nargoutchk(0, 1)

dataw = A.w;
datax = A.x;
datay = A.y;
dataz = A.z;

if isempty(dataw)
    tf = isreal(datax) & isreal(datay) & isreal(dataz);
else
    tf = isreal(dataw) & isreal(vee(A));
end

% $Id: isreal.m,v 1.7 2016/06/15 13:19:43 sangwine Exp $