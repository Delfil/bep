function writeCorrelatedGenes(N)
    if ~exist('Gene_Expression','var') || ~exist('CancerTypeIndex','var')
        fprintf('Loading the database.\n');
        load('GE.mat');
    end
    
    GE = selectNGenes(100);
    geneInd = selectCorrelatedGenes(N);
    
    for i = 1:N
        dataGen([mean(GE(:,geneInd(i,[1,3])),2), mean(GE(:,geneInd(i,[2,4])),2)], CancerTypeIndex,...
            'name', ['genes_', sprintf('%i+%i_%i+%i',geneInd(i,[1,3,2,4]))]);
        
        dataGen([mean(GE(:,geneInd(i,[1,4])),2), mean(GE(:,geneInd(i,[2,3])),2)], CancerTypeIndex,...
            'name', ['genes_', sprintf('%i+%i_%i+%i',geneInd(i,[1,4,2,3]))]);
    end

end