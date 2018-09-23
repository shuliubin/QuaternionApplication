function net = qcnnbp(net, y)
    %% 函数作用：计算并传递误差，计算梯度    
    
    %% 
    n = numel(net.layers);
    %   error
    net.e = net.o - y;
    %  loss function
    net.L = 1/2* sum(normer(net.e(:))) / size(net.e, 2);

    %%  backprop deltas 计算尾部感知器的误差
    net.od = net.e .* (net.o .* (onesq(size(net.o)) - net.o));   %  output delta，sigmoid的误差
%     net.fvd = (net.ffW' * net.od); %权重左乘             %  feature vector delta，特征向量误差，上一层收集下一层误差，size=[192,50]
    net.fvd = (net.od' * net.ffW)'; %权重右乘
    %如果MLP的前一层（特征抽取的最后一层）是卷积层，卷积层的输出经过sigmoid函数的处理，error需要求导
    %在数字识别这个网络中不需要用到
    if strcmp(net.layers{n}.type, 'c')         %  only conv layers has sigm function
        %对于卷积层，它的 特征向量误差 再求导(net.fv.*(1-net.fv))
%         net.fvd = net.fvd .* (net.fv .* (onesq(size(net.fv)) - net.fv));    %权重左乘
        net.fvd = ((onesq(size(net.fv)) - net.fv) .* net.fv) .* net.fvd ;    %权重右乘
    end

    %% 把单层感知器的输入层fetureVector的误差矩阵，恢复为subFeatureMap的4*4二维矩阵形式
    %  reshape feature vector deltas into output map style
    sa = size(net.layers{n}.a{1}); %size(a{1})=[4*4*50]，一共有a{1}~a{12}；
    fvnum = sa(1) * sa(2); %一个图所含有的特征向量数量，4*4
    for j = 1 : numel(net.layers{n}.a) %subFeatureMap2的数量，1:12
        %net最后一层的delta，由特征向量delta，依次取一个featureMap大小，然后组合为一个featureMap的形状
        %在fvd里面保存的是所有样本的特征向量（在cnnff.m函数中用特征map拉成的），这里需要重新变回来特征map的形式。
        %net.fvd(((j-1)*fvnum+1),:)，一个featureMap的误差d。
        net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
        %size(net.layers{numlayers}.d{j})=[4*4*50]
        %size(net.fvd)=[193*50]
    end
    
    
    %% 误差在特征提取网络（卷积降采样层）的传播
        %如果本层是卷积层，它的误差从后一层（降采样层）传过来，误差传播实际上是用降采样的反过程，
        %也就是降采样层的误差复制为2*2=4份。卷积层的输入是经过sigmoid处理的，从降采样层扩充来的
        %误差要经过sigmoid求导处理。
        %如果本层是降采样层，其误差从后一层（卷积层）传过来，误差传播实际是用卷积的反向过程，也
        %就是卷积层的误差，反卷积（卷积核转180度）卷积层的误差。
    for l = (n - 1) : -1 : 1
        if strcmp(net.layers{l}.type, 'c')
            % 如果是卷积层，误差从下一层（降采样层）传来，采用从后往前均摊的方式传播误差，上层误差内摊2倍，再除以4
            % 卷积层的灵敏度传输
            for j = 1 : numel(net.layers{l}.a)
                % net.layers{l}.a{j}.*(1-net.layers{l}.a{j})为sigmoid导数。
                % expand(,)多项式展开相乘。
                net.layers{l}.d{j} = net.layers{l}.a{j} .* (onesq(size(net.layers{l}.a{j})) - net.layers{l}.a{j}) .* (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) / net.layers{l + 1}.scale ^ 2);
            end
        elseif strcmp(net.layers{l}.type, 's')
            % 如果是降采样层，误差从一层（卷积层）传来，采用卷积的方式得到、
            for i = 1 : numel(net.layers{l}.a) % 第l层输出map的数量
                z = zeros(size(net.layers{l}.a{1})); % 得到featuremap大小的零矩阵
                for j = 1 : numel(net.layers{l + 1}.a) % 从l+1层收集误差
                    % net.layers{l+1}.d{j} 下一层（卷积层）的灵敏度
                    % net.layers{l+1}.k{i}{j} 下一层（卷积层）的卷积核
                     z = z + convnq(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                end
                net.layers{l}.d{i} = z;
            end
        end
    end

    %%  calc gradients 计算特征抽取层和单层感知器的梯度
    % 计算特征抽取层（卷积+降采样）的梯度
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c') % 卷积层
            for j = 1 : numel(net.layers{l}.a) % l层featureMap的数量
                for i = 1 : numel(net.layers{l - 1}.a) % l-1层featureMap的数量
                    %卷积核的修改量=输入图像*输出图像的delta
                    net.layers{l}.dk{i}{j} = sum(convnq(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid'),3) ./ size(net.layers{l}.d{j}, 3);
                end
                % net.layers{l}.d{j](:)是一个24*24*50的矩阵，db仅除以50
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
        end
    end
    % 计算单层感知器的梯度
    % sizeof(net.od)=[10,50]
    %修改量，求和除以50(batch大小)
%     net.dffW = net.od * (net.fv)' / size(net.od, 2);    % 权重左乘
    net.dffW = (net.fv * net.od')' / size(net.od, 2);    % 权重右乘
    net.dffb = mean(net.od, 2); %取第二维均值

    function X = rot180(X)
        X = flip(flip(X, 1), 2);
    end

    function X=flipall(X)
        for dim=1:ndims(X)
            X = flip(X,dim);
        end
    end
end
