function clusterMats = getClusterMats()
%load P (points found by tsne to represent our 100 genes data)
%from the file system
clusterMats = cell(20,1);
    for i = 1:20
        a = importdata(['../../../datasets/cluster/cluster', num2str(i), '.dat'], ',');
        clusterMats{i} = a(:)+1;
    end
end