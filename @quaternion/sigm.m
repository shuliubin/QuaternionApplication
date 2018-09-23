function b = sigm(a)

% updated.â€”â? 2018.   By Liu Bin.
nargoutchk(0, 1)

b = a;
b.w = 1./(1+exp(-a.w));
b.x = 1./(1+exp(-a.x));
b.y = 1./(1+exp(-a.y));
b.z = 1./(1+exp(-a.z));

end