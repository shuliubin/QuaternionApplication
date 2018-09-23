function h = homogeneous(p)
% HOMOGENEOUS - function to construct or normalise a full quaternion in
% homogeneous coordinates. Given a pure quaternion argument, inserts unity
% scalar part to give a full quaternion result in homogeneous coordinates.
% Given a full quaternion argument, normalises so that the scalar part is
% unity by dividing by the scalar part.

% Copyright (c) 2006, 2016 Stephen J. Sangwine.
% See the file : Copyright.m for further details.

narginchk(1, 1), nargoutchk(0, 1)

if ispure(p)
    
    % Insert scalar part(s) of ones to convert the argument to homogeneous
    % coordinates.
    
    h = ones(size(p)) + p;
else
    
    % Normalise the argument to have unit scalar part(s) and return the
    % vector part only.
    
    h = v(p) ./ s(p);
end

end

% Implementation note: there is the possibility of divide by zero errors in
% the case of a full quaternion argument if any of the scalar parts are very
% small. For the moment we do nothing about this, but it may be necessary
% to add some code to deal with it later.

% $Id: homogeneous.m,v 1.1 2016/08/10 13:37:08 sangwine Exp $