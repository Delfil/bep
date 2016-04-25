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

    write_data_files(file_name, N, genes, Data, Label);
    write_meta_file(file_name, N, genes, Data, Label);
    
end

function write_data_files(file_name, N, genes, Data, Label)

    data_file = fopen([file_name '.dat'], 'w');
    label_file = fopen([file_name '.lab'], 'w');
    
    for person = N;
        fprintf(label_file, '%u\r\n' , Label(person));
        fprintf(data_file, '%.4f,' ,Data(person, genes));
        fprintf(data_file, '%.4f' ,Data(person, genes(end:end)));
        fprintf(data_file, '\r\n');
    end
    fclose(data_file);
    fclose(label_file);

end

function write_meta_file(file_name, N, genes, ~, Label)

    fileid = fopen([file_name '.meta'], 'w');
    
    fprintf(fileid, '%s\n' , [file_name '.txt']);
    fprintf(fileid, '%i\n' , size(N,2));
    fprintf(fileid, '%i\n' , size(genes,2));
    fprintf(fileid, '%i\n' ,  numel(unique(Label)));
    
    fclose(fileid);

end