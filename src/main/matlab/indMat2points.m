function P = indMat2points(IndexMatrix)
    ma = max(max(IndexMatrix));
    P = zeros(ma,2);
    
    for i = 1:ma;
        [y,x] = find(IndexMatrix == i);
        P(i,:) = [x,y];
    end
end