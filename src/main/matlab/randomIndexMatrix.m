function M = randomIndexMatrix(~, P,~)
    m = randperm(size(P,1));
    sizz = ceil(sqrt(size(P,1)));
    m = [m, zeros(sizz^2 - numel(m))];
    M = reshape(m, ceil(sqrt(size(P,1))), ceil(sqrt(size(P,1))));
end