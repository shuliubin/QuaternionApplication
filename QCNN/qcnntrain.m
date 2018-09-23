function net = qcnntrain(net, x, y, opts)
%% 函数作用：训练CNN；
%生成随机序列，每次选取一个batch（50）个样本进行训练；
%批训练：计算50个随机样本的梯度，求和之后一次性更新到模型权重中。
% 在批训练过程中调用：
% 	gacnnff.m 完成前向过程
% 	gacnnbp.m 完成误差传导和梯度计算
% 	gacnnapplygrads.m 吧计算出来的梯度加到原始模型上去
    
%% 训练网络
    m = size(x, 3); % m为图片样本的数量；size(x)=【28*28*60000】
    numbatches = m / opts.batchsize; % batchsize为批训练时，一批所含图片样本数（=60000/50=1200）。
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = []; % rL是最小均方误差的平滑序列，绘图要用。
    for i = 1 : opts.numepochs  %迭代训练
        %显示训练到第几个epoch，一共多少个epoch
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        tic;
        %randperm(n)：产生1到n的整数的无重复随机排列，可得到无重复的随机数。
        %生成1到m(图片数量)整数的随机无重复排列，用于打乱训练顺序。
        kk = randperm(m);
        for l = 1 : numbatches  %训练每一个batch
            %得到训练信号，一个样本是一层x(:,:,sampleorder)，每次训练，取50个样本
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            %类标签，一个样本对应的标签为一列
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            
            %NN信号前向传导过程
            net = qcnnff(net, batch_x);
            %计算误差并反向传播，计算梯度
            net = qcnnbp(net, batch_y);
            %应用梯度，更新模型参数
            net = qcnnapplygrads(net, opts);
            %net.L为模型的costFunction，即最小均方误差mse
            %net.rL是最小均方误差的平滑序列
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            net.rL(end + 1) = net.L;%0.99 * net.rL(end) + 0.01 * net.L;
        end
        toc;
    end
    
end
