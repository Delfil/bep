function [evals,results] = testMatrices(correlations, MatrixGenerations, correlatingGenes, genesToCheck, N)

    if ~exist('genesToCheck', 'var')
        genesToCheck = size(correlations,1);
    end
    if ~exist('N', 'var')
        N = 20;
    end
    assert(N <=20, 'N cannot exceed 20, we only have 20 tsne runs.');
    
    P = getPs();
    CM = getClusterMats();
    evals = struct();
    results = cell(numel(MatrixGenerations)+1,1);
    for i = 1:numel(MatrixGenerations)
        disp(MatrixGenerations{i});
        meanEval = 0;
        results{i} = zeros(numel(correlatingGenes),N);
        for x = 1:N
            [meann, ~, results{i}(:,x)] = detKSizeMatJacc(correlations, MatrixGenerations{i}(correlations,P{x},CM{x}), correlatingGenes, genesToCheck);
            meanEval = meanEval*(x-1)/x+ (1/x) * meann;
        end
        evals.(func2str(MatrixGenerations{i})) = meanEval;
    end
    
    % compare to tsne's results

    disp('tsne');
    meanEval = 0;
    for x = 1:N
        [meann, ~, results{4}(:,x)] = detKSizeMatJacc_P(correlations, P{x}, correlatingGenes, genesToCheck);
        meanEval = meanEval*(x-1)/x + (1/x) * meann;
    end
    evals.('tsne') = meanEval;
end