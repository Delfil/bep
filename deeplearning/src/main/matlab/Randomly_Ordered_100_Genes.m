%Generate randomly ordered gene-expresion data
%select the 100 genes with the highest standard deviation calculate
%correlations and run tsne on it

if ~exist('Gene_Expression','var') || ~exist('CancerTypeIndex','var')
    disp('Loading the database.');
    load('GE.mat');
end

    ges = std(Gene_Expression);
    select = ges >1.2982; %hardcoded, assumes that the database stays the
    % same and that Gene_Expression and CancerTypeIndex don't get
    % overridden
    GE = Gene_Expression(:,select);
    
    
    data = Gene_Expression(:,randperm(size(Gene_Expression,2)));
    
    dataGen(data, CancerTypeIndex, 'name', 'Randomly_Ordered_100_Genes', 'height', 1, 'width', size(data,2));
