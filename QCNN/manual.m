maxsample_perclass = [1000];  % 每一类 设置的最大样本数
traintest_ratio = [6 1];    % 训练 - 测试 样本数比例
path = 'data\';
mode = {'simul'};%,'mnist','cifar10'};
I = length(mode);
noise = {'gauss' 'rand'};
J = length(noise);
volume = {'8s' '4s' '2s' '1s' '8h' '4h' '2h'};
K = length(volume);
ratio = {[1 3] [1 1] [3 1]};
L = length(ratio);
data_name = cell(I,J,K,L);

for i = 1 : I
    for j = 1 : J
        for k = 1 : K
            for l = 1 : L
                data_name{i,j,k,l} = [path,mode{i},'_',noise{j},'_',volume{k},'_',num2str(ratio{l}(1)),'-',num2str(ratio{l}(2)),'.mat'];
            end
        end
    end
end

for i = 1 : numel(data_name)
    %fprintf(data_name{i});
    disp([data_name{i}]);
    %fprintf('\n');
end

disp([i]);
%save 'data\simul_gauss_7s_1-6.mat' 'train_x' -v7.3 'train_y' -v7.3 'test_x' -v7.3 'test_y' -v7.3