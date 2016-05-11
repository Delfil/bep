file_points = fopen('100points.in','w');

fprintf(file_points,'%i\n', size(P,1));

for i = 1:size(P,1)
    fprintf(file_points,'%f ', P(i,1:end-1));
    fprintf(file_points,'%f\n', P(i,end));
end

fclose(file_points);

file_patients = fopen('geneact.in','w');

fprintf(file_patients, '%i\n', size(CancerTypeList,1));
for i = 1:size(GE,1)
    fprintf(file_patients,'%f,', GE(i,1:end-1));
    fprintf(file_patients,'%f\n', GE(i,end));
end

fclose(file_patients);



