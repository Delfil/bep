function [meann, maxx] = matJacc(correlations, Matrix, correlatingGenes, genesToCheck)

    assert(size(correlations,1) == size(correlations,2));
    assert(correlatingGenes>0);
    assert(correlatingGenes<size(correlations,1))
    if ~exist('genesToCheck', 'var')
        genesToCheck = size(correlations,1);
    end    
    assert(genesToCheck<=size(correlations,1));
    
    genes = randperm(size(correlations,1),genesToCheck);
    
    similarities = zeros(genesToCheck,1);
    
    for iter = 1:genesToCheck
        [~,I] = sort(correlations(genes(iter),:),2,'descend');
                
        setTarget = I(2:correlatingGenes+1);
        
        setMat = Matrix(matNClosest(Matrix,Matrix == I(1),correlatingGenes));
        setMat = setMat(setMat ~= 0);
        
        
        % the values in setTarget and setMat are already unique and contain
        % no zeros, so we can use this to compute the jaccard similarity.
        inters = numel(intersect(setTarget,setMat));
        total = numel(setTarget)+numel(setMat)-inters;
        
        similarities(iter) = inters/total;
    end
    meann = mean(similarities);
    maxx = max(max(similarities));
end