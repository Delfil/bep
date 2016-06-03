function evaluation = evalMGen(corrs, Points,genMatrix,N,evalMethod)

    if ~exist('N', 'var')
        N = 100;
    end
    if ~exist('evalMethod', 'var')
        evalMethod = @matJacc;
    end

    evals = zeros(N,1);

    for i = 1:N
        evals(i) = evalMatrix(corrs,genMatrix(Points),evalMethod);
    end
    
    evaluation = mean(evals);
end