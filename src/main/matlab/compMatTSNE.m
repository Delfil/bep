function evaluation = compMatTSNE(Points, Matrix, evalMethod)
    evaluation = evalMethod(Points,indMat2points(Matrix));
end