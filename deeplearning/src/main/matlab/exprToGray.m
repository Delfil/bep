function exprToGray(matrix)

A = zeros(size(matrix,1),size(matrix,2),'uint32');

for i = 0:size(matrix,1)
    for j = 0:size(matrix,2)
       if round(matrix(i,j)) > 2
           A(i,j) = 255;
       elseif round(matrix(i,j)) < -2
           A(i,j) = 0;
       else 
           A(i,j) = 128;
       end
    end
end

end

    