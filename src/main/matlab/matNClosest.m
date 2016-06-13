function Nclosest = matNClosest(M,I, N)
    assert(N<numel(find(M)), 'looking for the N closest cells in matrix M, but the number of elements in M is smaller than N.');
    [x,y] = find(I);
    
    Md = Inf(size(M));
    
    for i = 1:size(M,1)
        for j = 1:size(M,2)
            Md(i,j) = sqrt((i-x)^2 + (j-y)^2);            
        end
    end
    [~,indClosest] = sort(Md(:));
    indClosest = indClosest(2:end);
    
    Nclosest = indClosest(1:N);
    
end