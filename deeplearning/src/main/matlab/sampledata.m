function sampledata(people)

    if ~exist('Gene_Expression','var')
        load('GE.mat');
        sample = randi(size(Gene_Expression, 2), 1, 200);
    elseif ~exist('sample', 'var')
        sample = randi(size(Gene_Expression, 2), 1, 200);
    end

    fileid = fopen('sampleData.txt', 'w');

    for person = randi(size(Gene_Expression, 1), 1, people);
        fprintf(fileid, '%.4f,' ,Gene_Expression(person, sample));

        fprintf(fileid, '%d\r\n' , CancerTypeIndex(person));
    end
    fclose(fileid);

end
