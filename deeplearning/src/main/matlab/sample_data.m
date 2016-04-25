function sample_data(N, NGenes, Data, Labels, file_name)

    if ~exist('file_name', 'var')
        file_name = 'Sample_data';
    end

    people = randperm(size(Data,1));
    genes = randperm(size(Data,2));
    selection_genes = genes(1:NGenes);
    
    data_selection = people(1:N);
        
    write_files(file_name , data_selection, selection_genes, Data, Labels);
        
end

function write_files(file_name, N, genes, Data, Label)

    write_data_file(file_name, N, genes, Data, Label);
    write_meta_file(file_name, N, genes, Data, Label);
    
end

function write_data_file(file_name, N, genes, Data, Label)

    fileid = fopen([file_name '.txt'], 'w');
    
    for person = N;
        fprintf(fileid, '%u,' ,Data(person, genes));

        fprintf(fileid, '%u\r\n' , Label(person));
    end
    fclose(fileid);

end

function write_meta_file(file_name, N, genes, ~, Label)

    fileid = fopen([file_name '.meta'], 'w');
    
    fprintf(fileid, '%s\n' , [file_name '.txt']);
    fprintf(fileid, '%i\n' , size(N,2));
    fprintf(fileid, '%i\n' , size(genes,2));
    fprintf(fileid, '%i\n' ,  numel(unique(Label)));
    
    fclose(fileid);

end