function [er, bad] = qcnntest(net, x, y)
    %% 验证样本的准确率
    %  feedforward
    net = qcnnff(net, x);
    [~, h] = max(normer(net.o));
    [~, a] = max(normer(y));
    % find(x) Find indices of nonzero elements.
    bad = find(h ~= a); %计算预测错误的样本数量
    er = numel(bad) / size(y, 2); %计算错误率
    
end
