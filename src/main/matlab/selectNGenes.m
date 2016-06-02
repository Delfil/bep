function GE = selectNGenes(N, Gene_Expression, ~)
if ~exist('Gene_Expression','var')
    fprintf('Loading the database.\n');
    load('GE.mat');
end
assert(N <=size(Gene_Expression,2));


[~,select] = sort(std(Gene_Expression),'descend');
select = sort(select(1:N));
GE = Gene_Expression(:,select);
end