function net = qcnnsetup(net, x, y)
%% 函数作用：初始化GACNN的参数——设置各层mapsize的大小；__以下说明参考实数CNN
%                                初始化卷积层的卷积核、偏置；
%                                网络尾部单层感知机的参数设置
% 具体设置：
% 偏置（bias）：一般统一设置为0；
% 权重设置：区间[-1,1]之间的随机数/sqrt(6/(输入神经元数量+输出神经元数量)）
% 卷积核权重：输入、输出为fan_in，fan_out；
% 卷积核初始化，C1层有1*6个卷积核，C3层有6*12=72个卷积核。
%   fan_in=numInputmaps * net.layers{1}.kernelsize^2；
%   fin=1*25  or  6*25      【fin表示该层的一个输出map，所对应的所有卷积核，包含的神经元的总数。1*25，6*25】
%   fan_out=net.layers{1}.outputmaps * net.layers{1}.kernelize^2。
%   fout=1*6*25  or  6*12*25
%   net.layers{1}.k{i]{j}=(randu(net.layers{1}.kernelsize)-0.5*oner(net.layers{1}.kernelsize))*2*sqrt(6/fain_in+fan_out);

%%  版本检测
    assert(~isOctave() || compare_versions(OCTAVE_VERSION, '3.8.0', '>='), ['Octave 3.8.0 or greater is required for CNNs as there is a bug in convolution in previous versions. See http://savannah.gnu.org/bugs/?39314. Your version is ' myOctaveVersion]);   
    
%% 卷积降采样的参数初始化    
    inputmaps = 1;  % 输入层 map
    mapsize = size(x(:, :, 1)); % 网络输入矩阵的大小
% 以下通过传入net这个结构体来逐层构建CNN
    for l = 1 : numel(net.layers)   %  每一层
        if strcmp(net.layers{l}.type, 's')  % 降采样层 or 池化层
            mapsize = mapsize / net.layers{l}.scale; % 输出矩阵大小
            assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);
            for j = 1 : inputmaps   % 偏置初始化
                net.layers{l}.b{j} = zeros(mapsize);
            end
        end
        if strcmp(net.layers{l}.type, 'c')  % 卷积层
            mapsize = mapsize - net.layers{l}.kernelsize + 1; % 输出矩阵大小
            fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2; % 权重参数 数量 or 连接数
            for j = 1 : net.layers{l}.outputmaps  %  每一个输出通道 map
                fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;
                for i = 1 : inputmaps  %  每一个输入矩阵 map
                    net.layers{l}.k{i}{j} = randq(net.layers{l}.kernelsize) * 2 * sqrt(6 / (fan_in + fan_out)); % 初始化卷积核
                end
                net.layers{l}.b{j} = zeros(1); % 偏置初始化
            end
            inputmaps = net.layers{l}.outputmaps;
        end
    end
    
   %% 尾部单层感知器（全连接）的参数（权重和偏置）设置：
    % 'onum' is the number of labels, that's why it is calculated using size(y, 1). If you have 20 labels so the output of the network will be 20 neurons.
    % 'fvnum' is the number of output neurons at the last layer, the layer just before the output layer.
    % 'ffb' is the biases of the output neurons.
    % 'ffW' is the weights between the last layer and the output neurons. Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)
    fvnum = prod(mapsize) * inputmaps;  %prod():数组元素的乘积(矩阵A(m,n),m>=2,n>=2时按列乘，结果为行向量)。
    onum = size(y, 1);  %输出节点的数量

    net.ffb = zeros(onum, 1);
    net.ffW = randq(onum, fvnum) * 2 * sqrt(6 / (onum + fvnum));
end
