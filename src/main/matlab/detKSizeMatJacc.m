function [evaluation, interpretation, results] = detKSizeMatJacc(correlations, Matrix, correlatingGenes, genesToCheck)
    if ~exist('genesToCheck', 'var')
        genesToCheck = 100;
    end

    results = zeros(numel(correlatingGenes),1);

    i = 1;
    for NGenes = correlatingGenes;        
        [results(i),~] = matJacc(correlations, Matrix, NGenes, genesToCheck);

        i = i + 1;
    end

    evaluation =  mean(results);
    best = correlatingGenes(results == evaluation & results ~= 0);
    


    interpretation = struct('best_value_for_correlatingGenes', best);
end