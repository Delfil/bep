<<<<<<< HEAD:src/main/matlab/point_patient.m
file_points = fopen('100points.in','w');
=======
%File of points we would like to create
file_points = fopen('point.in','w');
>>>>>>> origin:src/main/matlab/point_patient.m

%Print the number of points on the first line
fprintf(file_points,'%i\n', size(P,1));

%Print the coordinates of the points row for row.
for i = 1:size(P,1)
    fprintf(file_points,'%f ', P(i,1:end-1));
    fprintf(file_points,'%f\n', P(i,end));
end

fclose(file_points);

<<<<<<< HEAD:src/main/matlab/point_patient.m
file_patients = fopen('geneact.in','w');

fprintf(file_patients, '%i\n', size(CancerTypeList,1));
=======
%File of gene activation we would like to create
file_patients = fopen('patients.in','w');

%Gene activation per patient per row.
>>>>>>> origin:src/main/matlab/point_patient.m
for i = 1:size(GE,1)
    fprintf(file_patients,'%f ', GE(i,1:end-1));
    fprintf(file_patients,'%f\n', GE(i,end));
end

fclose(file_patients);



