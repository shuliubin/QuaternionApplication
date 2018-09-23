function setup_data

%% R^3中5个三维内嵌流行
%{
%f1(t)=[x1(t),y1(t),z1(t)]=[z1.*cos(a).*sin(a),z1.*sin(a).*sin(a),z1.*cos(a)]
%f2(t)=[x2(t),y2(t),z2(t)]=[z2.*cos(a).*sin(a),z2.*sin(a).*sin(a),z2.*cos(a)]
%f3(t)=[x3(t),y3(t),z3(t)]=[z3.*cos(a).*sin(a),z3.*sin(a).*sin(a),z3.*cos(a)]
%f4(t)=[x4(t),y4(t),z4(t)]=[z4.*cos(a).*sin(a),z4.*sin(a).*sin(a),z4.*cos(a)]
%f5(t)=[x5(t),y5(t),z5(t)]=[z5.*cos(a).*sin(a),z5.*sin(a).*sin(a),z5.*cos(a)]
clc
a=linspace(0,2*pi,32);
dz=[1,10,20,30,40,50];
f=zeros(5,3,length(a));
for i = 1 : 5
    z0=4*linspace(0,10,32)+dz(i);
    x=z0.*cos(a).*sin(a);
    y=z0.*sin(a).*sin(a);
    z=z0.*cos(a);
    f(i,1,:)=x(:)';
    f(i,2,:)=y(:)';
    f(i,3,:)=z(:)';
    plot3(squeeze(f(i,1,:)),squeeze(f(i,2,:)),squeeze(f(i,3,:)));
    hold on
    scatter3(f(i,1,:),f(i,2,:),f(i,3,:));
end
grid on
%}

%% 三维曲面
%{
[a,b,c]=peaks(50);
figure(1)
mesh(a,b,c)
axis tight

x=linspace(-8,8,32);
y=x;
[X,Y]=meshgrid(x,y);
R=sqrt(X.^2+Y.^2)+eps;
Z=sin(R)./R;
mesh(X,Y,Z);
%}

%% 四元数实验数据集拆分为四维矩阵
load('data\data_quaternion_function.mat')
trainW=train_x.w;
trainX=train_x.x;
trainY=train_x.y;
trainZ=train_x.z;
train_x_matrix={trainW,trainX,trainY,trainZ};
train_y_real=normer(train_y);
train_y_real(train_y_real>0)=1;

testW=test_x.w;
testX=test_x.x;
testY=test_x.y;
testZ=test_x.z;
test_x_matrix={testW,testX,testY,testZ};
test_y_real=normer(test_y);
test_y_real(test_y_real>0)=1;

save 'data\correspondingReal_quaternion_simu.mat' 'train_x_matrix' -v7.3 'train_y_real' -v7.3 'test_x_matrix' -v7.3 'test_y_real' -v7.3

end