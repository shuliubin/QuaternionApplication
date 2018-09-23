function net = qcnnff(net, x)
    %% 取得GACNN的输入
    n = numel(net.layers);
    net.layers{1}.a{1} = x; % a是输入map，是一个28*28*50的 clifford矩阵
    inputmaps = 1;
    
    %% 两次卷积和降采样处理
    for l = 2 : n   %  for each layer
        if strcmp(net.layers{l}.type, 'c')
            %  !!below can probably be handled by insane matrix operations
            for j = 1 : net.layers{l}.outputmaps   %  for each output map
                %  create temp output map
                %  z=zeros([28,28,50]-[4,4,0])=seros([24,24,50]);
                z = zerosq(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                for i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convnq(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                end
                %  add bias, pass through nonlinearity
                [z1, z2, z3] = size(z);
                bias = repmat(net.layers{l}.b{j},z1,z2,z3);
                net.layers{l}.a{j} = sigm(z + bias);
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;
        elseif strcmp(net.layers{l}.type, 's')
            %  downsample
            for j = 1 : inputmaps
                % 卷积一个[0.25,0.25;0.25,0.25]的卷积核，然后降采样  ?????：有重复计算
                z = convnq(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) ./ (net.layers{l}.scale ^ 2), 'valid');   %  !! replace with variable
                %设置scale为步长，窗花滑动的偏移量=scale   ?????：有计算浪费
                net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
            end
        end
    end
    
    %% 单层感知器的数据处理，需要把subFeatureMap2连接成为一个(4*4)*12=192的向量，
       %但由于采用了50样本批训练的方法，subFeatureMap2被拼接合成为一个192*50的特征向量fv；
       %fv作为单层感知器的输入，以全连接的方式连接到输出层。
    %  concatenate all end layer feature maps into vector
    net.fv = [];
    for j = 1 : numel(net.layers{n}.a) %fv每次拼合入subFeatureMap2[j]，[包含50个样本]
        sa = size(net.layers{n}.a{j}); % size(a)=[4,4,50]；得到Sfm2的一个输入图的大小
        %reshape(A,m,n)：把矩阵A变为m行n列的矩阵（元素个数不变，原矩阵按列排成一队，再按行排成若干队）
        %把net.layers{numLayers}.a{j}(一个Sfm2)排列成[4*4行，1列]的向量。
        %把所有Sfm2拼合成为一个列向量fv，[net.layers{numLayers}.a{j},4*4,50]
        %最后的fv是一个[(16*12)*50=192*50]的矩阵
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
    end
    %  feedforward into output perceptrons
    %net.ffW是一个[10,192]的权重矩阵
    %net.ffW * net.fv是一个[10,50]的矩阵
    %remat(net.ffb,1,size(net.fv,2))把bias复制成50份排开
    col = size(net.fv,2);   % page
%     net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, col));%%权重左乘
    net.o = sigm( ((net.fv')*(net.ffW'))' + repmat(net.ffb, 1, col));%%权重左乘

end
