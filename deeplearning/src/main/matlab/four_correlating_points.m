absCorr = abs(corr(Gene_Expression));
point = ceil(size(absCorr,1)*rand());

column = absCorr(:,point);
[sortedValues, sortedIndex] = sort(column,'descend');

fourpoints = sortedIndex(1:4);
avg_act = mean(Gene_Expression(:,fourpoints),2);

dataGen(Gene_Expression(:,sortedIndex(1)), CancerTypeIndex-1, 'name', 'gene1', 'height', 1, 'width', 1);
dataGen(Gene_Expression(:,sortedIndex(2)), CancerTypeIndex-1, 'name', 'gene2', 'height', 1, 'width', 1);
dataGen(Gene_Expression(:,sortedIndex(3)), CancerTypeIndex-1, 'name', 'gene3', 'height', 1, 'width', 1);
dataGen(Gene_Expression(:,sortedIndex(4)), CancerTypeIndex-1, 'name', 'gene4', 'height', 1, 'width', 1);
dataGen(avg_act, CancerTypeIndex-1, 'name', 'geneAVG', 'height', 1, 'width', 1);

