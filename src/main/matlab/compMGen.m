function evaluation = compMGen(Points,genMatrix,N,evalMethod)

    if ~exist('N', 'var')
        N = 10;
    end
    if ~exist('evalMethod', 'var')
        evalMethod = @compPCs;
    end

    evals = zeros(N,1);

    for i = 1:N
        evals(i) = compMatTSNE(Points, genMatrix(Points),evalMethod);
    end
    
    evaluation = mean(evals);
end