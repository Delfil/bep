function TSNE_1D_100_genes(in_matlab_folder)
    %select the 100 genes with the highest standard deviation.
    %calculate correlations and run tsne on the inverse.
    %run 1d t-sne
    name = 'TSNE_1D_inv_corrs_100_genes';

    if ~exist('Gene_Expression','var') || ~exist('CancerTypeIndex','var')
        disp('Loading the database.');
        load('GE.mat');
    end
    
    if ~exist('in_matlab_folder', 'var')
        in_matlab_folder = false;
    end

    % take the 100 genes with highest standard deviation
    GE = selectNGenes(100,Gene_Expression);
    
    % calculate correlations
    corrs = corr(GE);
    
    %reduce to one dimension
    P = tsne_p(corrs, [], 1);
    
    %sort the points
    [~,ind] = sort(P);
    
    %navigate to right folder
    if in_matlab_folder
        path = sprintf('../../../datasets/%s/', name);
    else
        path = sprintf('%s/', name);
    end

    mkdir(path);
    
    dataGen(GE(:,ind), CancerTypeIndex, 'height', 1, 'width', numel(ind),...
        'name', [path, name])
end