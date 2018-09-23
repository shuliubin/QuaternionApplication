sourceImg=imread('./lenna.tif');
sourceImg=double(sourceImg);
qua_sourceImg=quaternion(sourceImg(:,:,1),sourceImg(:,:,2),sourceImg(:,:,3));
[U, S, V]=svd(qua_sourceImg);
Qrecov=U*S*V.';
Irecov=quaToimg(Qrecov);
imshow(Irecov);


