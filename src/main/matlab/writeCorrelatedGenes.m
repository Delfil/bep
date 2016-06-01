function writeCorrelatedGenes(N, GE, CancerTypeIndex, geneInd)
    path = '../../../datasets';
    
    for i = 1:N
        dir1 = sprintf('genes_A=%i; a=%i; B=%i; b=%i;',geneInd(i,[1,3,2,4]));
        mkdir(path,dir1);
        
        dataGen(GE(:,geneInd(i,[1,3,4,2])),...
            CancerTypeIndex,...
           'name', [path, '/', dir1, '/', dir1],...
           'height', 2,...
           'width', 2);
    end
end