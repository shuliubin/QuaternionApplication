function net = qcnnapplygrads(net, opts)
    %% 函数作用：更新网络参数
    %特征抽取层（卷积降采样）的权重更新
    for l = 2 : numel(net.layers)  %从第2层开始
        if strcmp(net.layers{l}.type, 'c') %对于每个卷积层
            for j = 1 : numel(net.layers{l}.a) %枚举该层的每个输出
                %枚举所有卷积核 net.layers{l}.k{ii}{j}
                for ii = 1 : numel(net.layers{l - 1}.a) %枚举上层的每个输入
                    net.layers{l}.k{ii}{j} = net.layers{l}.k{ii}{j} - opts.alpha * net.layers{l}.dk{ii}{j};
                end
                %修改bias
                net.layers{l}.b{j} = net.layers{l}.b{j} - opts.alpha * net.layers{l}.db{j};
            end
        end
    end
    
    %单层感知器的参数更新
    net.ffW = net.ffW - opts.alpha * net.dffW;
    net.ffb = net.ffb - opts.alpha * net.dffb;
end
