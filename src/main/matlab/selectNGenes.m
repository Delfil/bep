function GE = selectNGenes(N, Gene_Expression, ~)
if ~exist('Gene_Expression','var') || ~exist('CancerTypeIndex','var')
    fprintf('Loading the database.\n');
    load('GE.mat');
end
assert(N <=size(Gene_Expression,2));

[~,select] = sort(std(Gene_Expression));
select = sort(select);
GE = Gene_Expression(:,select(1:N));
end