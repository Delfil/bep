function [evaluation, interpretation, results] = detKSizeMatJacc(correlations, Matrix, halfKernelSize, correlatingGenes, genesToCheck)
    if ~exist('halfKernelSize', 'var')
        halfKernelSize = 1:3;
    end
    if ~exist('correlatingGenes', 'var')
        correlatingGenes = 1:49;
    end
    if ~exist('genesToCheck', 'var')
        genesToCheck = 100;
    end

    results = zeros(numel(halfKernelSize),numel(correlatingGenes));

    maxx = 0;
    
    i = 1;
    for halfKSize = halfKernelSize;
        fprintf('Calculating for kernelSize: %i\n', 2*halfKSize+1);
        j = 1;
        for corrGenes = correlatingGenes
            fprintf('\t... %i correlating gene(s)\n', corrGenes);
            [results(i,j),x] = matJacc(correlations, Matrix, halfKSize, corrGenes, genesToCheck);
            if x > maxx
                maxx = x;
            end
            j = j + 1;
        end
        i = i + 1;
    end

    evaluation =  max(max(results));
    [kernelSize,nrCorrGenes] = find(results == evaluation & results ~= 0);
    kernelSize = halfKernelSize(kernelSize)*2 + 1;
    nrCorrGenes = correlatingGenes(nrCorrGenes);


    interpretation = struct('kernelSize', kernelSize, ...
        'correlatingGenes', nrCorrGenes);
    maxx
end