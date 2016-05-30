function dissimilarity = compPCs(pointCloudA, pointCloudB)
% COMPPCS(a,b) returns a measure of dissimilarity of pointcloud a and b.

    assert(all(size(pointCloudA) == size(pointCloudB)));

    Adist = pdist(pointCloudA);
    Adist = Adist./sum(Adist);

    Bdist = pdist(pointCloudB);
    Bdist = Bdist./sum(Bdist);

    dissimilarity = sum(abs(Adist-Bdist));
end
