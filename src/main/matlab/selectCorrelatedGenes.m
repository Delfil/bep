function genesI = selectCorrelatedGenes(N)
GE = selectNGenes(100);
corrs = corr(GE);

genesI = zeros(N,4);
for i = 1:N
    %we always want to run once, but there is not do while loop in matlab
    not_done = true;
    while not_done

        a = randperm(size(GE,2),2);
        b = [0,0];
        x = 1;
        for j = a
            [~,ind] = sort(corrs(:,j),'descend');
            b(x) = ind(2);
            x = x + 1;
        end
        
        % as long as there is at least one index selected more than once,
        % we have to run again
        not_done = sum(sum(ismember(a,b))) > 0 && b(1) ~= b(2);
        
        %not really pretty to do this here, but it works (it is written
        %multiple times rather than once when the above condition is false)
        genesI(i,:) = [a,b];
    end
end