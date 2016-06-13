function P = getPs()
%load P (points found by tsne to represent our 100 genes data)
%from the file system
P = cell(20,1);
    for i = 1:20
        a = importdata(['../../../datasets/clusterPoints/', num2str(i), '/point.in'], ',');
        P{i} = [a(2:2:end),a(3:2:end)];
    end
end