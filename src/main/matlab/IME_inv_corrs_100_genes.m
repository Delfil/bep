function IME_inv_corrs_100_genes(in_matlab_folder)
    %select the 100 genes with the highest standard deviation calculate
    %correlations and run tsne on the inverse. then make a matrix from that and
    %
    name = 'IME_inv_corrs_100_genes';
    
    if ~exist('Gene_Expression','var') || ~exist('CancerTypeIndex','var')
        disp('Loading the database.');
        load('GE.mat');
    end
    
    if ~exist('in_matlab_folder', 'var')
        in_matlab_folder = false;
    end

    GE = selectNGenes(100,Gene_Expression);
    corrs = corr(GE);
    
    inverseCorrs = -1*corrs;
    P = tsne_p(inverseCorrs);
    M = map_d(P);
    genes = M(:)';
    
    GEt = [zeros(size(GE,1),1), GE];
    data = GEt(:,genes+1);
    clear GEt;
    
    %navigate to right folder
    if in_matlab_folder
        path = sprintf('../../../datasets/%s/', name);
    else
        path = sprintf('%s/', name);
    end

    mkdir(path);
    
    dataGen(data, CancerTypeIndex,...
        'name', [path, name],...
        'height', size(M,1), 'width', size(M,2));
end