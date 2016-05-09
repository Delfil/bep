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
    P = tsne_p(corrs);
    
    M = map_d(P);
    genes = M(:)';
    data = GE(:,genes(genes~=0));
    
    sample_data(0,0,data, CancerTypeIndex, '100_Genes');
