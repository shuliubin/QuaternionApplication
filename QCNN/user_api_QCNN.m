function user_api_QCNN



clc
fprintf('\n程序运行中...');
%% 数据加载与格式转换以及归一化处理
%{
%模型采用的数据库为MNIST
%含有70000 个手写数字样本，其中的6000个作为训练样本，10000个作为测试样本
load mnist_uint8;
train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');
%}

%% 实验1：仿真实验
%{
%% 构建二维四元数函数图形：patch1、patch2
rng(0)
patch_size = 32;
num_kind = 3;
patch1 = zerosq(patch_size,patch_size);
patch2 = zerosq(patch_size,patch_size);
patch3 = zerosq(patch_size,patch_size);
for x = 1 : patch_size
    for y = 1 : patch_size
        patch1(x,y) =  quaternion(x*x+y*y,x*x+y*y,x*x+y*y,2*x*x+y*y);
        patch2(x,y) =  quaternion(x*x-y*y,x*x-y*y,x*x-y*y,x*x+2*y*y);
        patch3(x,y) =  quaternion(x*x+y*y,x*x-y*y,x*x+y*y,x*x-y*y);
    end
end

%% 标签标识
outflag = [quaternion(0.1,0.5,0.1,0.5)  quaternion(0.5,0.2,0.1,0.5) quaternion(0.1,0.5,0.2,0.1)]';

%% 训练集与标签+randq(1,1,train_per);
train_per = 500; % 每一类多少个样本
train_x = zerosq(patch_size,patch_size,train_per*num_kind);
train_y = zerosq(num_kind,train_per*num_kind);
for i = 1 : train_per
    train_x(:,:,(i-1)*num_kind+1) = patch1+randq(patch_size,patch_size);
    train_x(:,:,(i-1)*num_kind+2) = patch2+randq(patch_size,patch_size);
    train_x(:,:,(i-1)*num_kind+3) = patch3+randq(patch_size,patch_size);
    train_y(:,(i-1)*num_kind+1) = [outflag(1) 0 0]';
    train_y(:,(i-1)*num_kind+2) = [0 outflag(2) 0]';
    train_y(:,(i-1)*num_kind+3) = [0 0 outflag(3)]';
end

%% 测试集与标签
test_per = 200;
test_x = zerosq(patch_size,patch_size,test_per*num_kind);
test_y = zerosq(num_kind,test_per*num_kind);
for i = 1 : test_per
    test_x(:,:,(i-1)*num_kind+1) = patch1+randq(patch_size,patch_size);
    test_x(:,:,(i-1)*num_kind+2) = patch2+randq(patch_size,patch_size);
    test_x(:,:,(i-1)*num_kind+3) = patch3+randq(patch_size,patch_size);
    test_y(:,(i-1)*num_kind+1) = [outflag(1) 0 0]';
    test_y(:,(i-1)*num_kind+2) = [0 outflag(2) 0]';
    test_y(:,(i-1)*num_kind+3) = [0 0 outflag(2)]';
end
save 'data\data_quaternion_function.mat' 'train_x' -v7.3 'test_x' -v7.3 'train_y' -v7.3 'test_y' -v7.3
%}
load 'data_quaternion_function.mat';%'3D_shape.mat';%
train_x0 =train_x;
train_x=zeros(32,32,4,1500);
train_x(:,:,1,:)=train_x0.w;
train_x(:,:,2,:)=train_x0.x;
train_x(:,:,3,:)=train_x0.y;
train_x(:,:,4,:)=train_x0.z;
train_y=zeros(3,1500);
for i = 1 : 500
    train_y(:,(i-1)*3+1) = [1 0 0]';
    train_y(:,(i-1)*3+2) = [0 1 0]';
    train_y(:,(i-1)*3+3) = [0 0 1]';
end

test_x0 =test_x;
test_x=zeros(32,32,4,600);
test_x(:,:,1,:)=test_x0.w;
test_x(:,:,2,:)=test_x0.x;
test_x(:,:,3,:)=test_x0.y;
test_x(:,:,4,:)=test_x0.z;
test_y=zeros(3,600);
for i = 1 : 200
    test_y(:,(i-1)*3+1) = [1 0 0]';
    test_y(:,(i-1)*3+2) = [0 1 0]';
    test_y(:,(i-1)*3+3) = [0 0 1]';
end
save 'data\quaterniondata_2_realmatrix.mat' 'train_x' -v7.3 'test_x' -v7.3 'train_y' -v7.3 'test_y' -v7.3
% 
% %% 设置网络结构及训练参数
% % ex1 Train a 6c-2s-12c-2s Convolutional neural network 
% % will run 1 epoch in about 200 second and get around 11% error. 
% % With 100 epochs you'll get around 1.2% error
% 
% rand('state',0)
% 
% qcnn.layers = {
%     struct('type', 'i') %input layer
%     struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
%     struct('type', 's', 'scale', 4) %sub sampling layer
% %     struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
% %     struct('type', 's', 'scale', 2) %subsampling layer
% };
% 
% % 训练选项，alpha为学习速率（取值一般为在区间(0，1]范围内），
% % batchsize为批量训练中样本的数量，numepoches为迭代次数。
% opts.alpha = 1; %学习率
% opts.batchsize = 15; %每次挑出一个batchsize的batch来训练，
%                      %也就是每用batchsize个样本就调整一次权值，
%                      %而不是把所有样本都输入了，计算所有样本的误差才调整一次权值。
% opts.numepochs = 1; %训练次数
% 
% %% 初始化网络；对数据进行批训练；验证模型准确率
% qcnn = qcnnsetup(qcnn, train_x, train_y);  %初始化卷积核、偏置
% qcnn = qcnntrain(qcnn, train_x, train_y, opts); %训练网络：前向传播、后向传播、参数更新
% [er, bad] = qcnntest(qcnn, test_x, test_y); %测试当前模型准确率
% 
% %% plot mean squared error （绘制均方误差曲线）
% % err_epoch1 = qcnn.rL;
% % save 'data\err_epoch1_sigm' 'err_epoch1'
% figure(2); plot(qcnn.rL);
% xlabel('QCNN 训练次数');
% ylabel('Train MSE');
% %assert(er<0.12, 'Too big error');
% disp([num2str(er*100),'%error']);
% 
% end
