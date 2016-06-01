function writeCorrelatedGenes(N)
    if ~exist('Gene_Expression','var') || ~exist('CancerTypeIndex','var')
        fprintf('Loading the database.\n');
        load('GE.mat');
    end
    
    GE = selectNGenes(100);
    geneInd = selectCorrelatedGenes(N);

    path = '../../../datasets';
    
    for i = 1:N
        dir1 = sprintf('genes_A=%i; a=%i; B=%i; b=%i; Aa&Bb',geneInd(i,[1,3,2,4]));
        dir2 = sprintf('genes_A=%i; a=%i; B=%i; b=%i; Ab&Ba',geneInd(i,[1,3,2,4]));
        mkdir(path,dir1);
        mkdir(path,dir2);
        
        dataGen([mean(GE(:,geneInd(i,[1,3])),2), mean(GE(:,geneInd(i,[2,4])),2)], CancerTypeIndex,...
           'name', [path, '/', dir1, '/', dir1]);
       
       dataGen([mean(GE(:,geneInd(i,[1,4])),2), mean(GE(:,geneInd(i,[2,3])),2)], CancerTypeIndex,...
           'name', [path, '/', dir2, '/', dir2]);
    end
end