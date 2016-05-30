function M = map_r(Points)
    siz = ceil(sqrt(size(Points,1)));
    a = [1:size(Points,1) zeros(1,siz^2 - size(Points,1))];
    b = randperm(numel(a));
    M = reshape(a(b),siz,siz);
end