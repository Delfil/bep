%select the 100 genes with the highest standard deviation calculate
%correlations and run tsne on it

if ~exist('Gene_Expression','var')
    disp('Loading the database.');
    load('GE.mat');
end

    ges = std(Gene_Expression);
    select = ges >1.2982;
    GE = Gene_Expression(:,select);
    corrs = corr(GE);
    tsne_p(corrs);