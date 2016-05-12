%File of points we would like to create
file_points = fopen('100points.in','w');

%Number of points printed
fprintf(file_points,'%i\n', size(P,1));

%Get all the points and print on each line
for i = 1:size(P,1)
    fprintf(file_points,'%f ', P(i,1:end-1));
    fprintf(file_points,'%f\n', P(i,end));
end

fclose(file_points);

%File of points we would like to create
file_patients = fopen('geneact.in','w');

%Print the number of labels as first line
fprintf(file_patients, '%i\n', size(CancerTypeList,1));
%Get all the gene information into the file
for i = 1:size(GE,1)
    fprintf(file_patients,'%f ', GE(i,1:end-1));
    fprintf(file_patients,'%f\n', GE(i,end));
end

fclose(file_patients);



