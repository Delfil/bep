function corrGenes(N)
    if ~exist('Gene_Expression','var') || ~exist('CancerTypeIndex','var')
        fprintf('Loading the database.\n');
        load('GE.mat');
    end

    GE = selectNGenes(100, Gene_Expression, CancerTypeIndex);
    geneInd = selectCorrelatedGenes(N,GE);
    writeCorrelatedGenes(N,GE,CancerTypeIndex,geneInd);

end