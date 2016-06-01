function genCorrVars(CorrelationMatrix,n)
    if ~exist('n', 'var')
        n = 10000;
    end
    
    C = [   1,.7,.1,.2
            .7,1,.15,.1
            .1,.15,1,.6
            .2,.1,.6,1 ];

    if sum(sum(CorrelationMatrix == CorrelationMatrix')) == numel(CorrelationMatrix)
        C = CorrelationMatrix;
    end

    [e,lamb] = eig(C);
    V = e * sqrt(lamb);
    rands = rand(n,4);
    corrRands = rands*V';
    corr(corrRands)

end