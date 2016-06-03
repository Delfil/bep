function [meann, maxx] = matJacc(correlations, Matrix, halfKSize, correlatingGenes, genesToCheck)
    assert(halfKSize == round(halfKSize));
    assert(halfKSize > 0);
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
        
        [i,j] = find(Matrix == I(1));
        
        setTarget = I(2:correlatingGenes+1);
        submat = zeros(3*halfKSize);
        
        for y = max(1,i-halfKSize):min(size(Matrix,1),i+halfKSize)
            for x = max(1,j-halfKSize):min(size(Matrix,2),j+halfKSize)
                submat(y+halfKSize,x+halfKSize) = Matrix(y,x);
            end 
        end
        
        setMat = submat(submat ~= I(1));
        setMat = setMat(:);
        setMat = setMat(setMat~=0);
        
        % the values in setTarget and setMat are already unique and contain
        % no zeros, so we can use this to compute the jaccard similarity.
        inters = numel(intersect(setTarget,setMat));
        total = numel(setTarget)+numel(setMat)-inters;
        
        similarities(iter) = inters/total;
    end
    meann = mean(similarities);
    maxx = max(max(similarities));
end