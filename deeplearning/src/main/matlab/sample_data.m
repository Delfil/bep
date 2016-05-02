function sample_data(N, NGenes, Data, Labels, file_name)
% SAMPLE_DATA constructs a random subset from the data
% (or all of it, if N=0 and NGenes=0) and writes it to files
%
% SAMPLE_DATA() will write the GE data to file
%
% N = the amount of rows you want
% NGenes = the amount of columns you want
% Data = the observed gene activations
% Labels = the labels corresponding to the row in Data
% file_name = the name of the .dat, .lab and .meta files

    if ~exist('Data', 'var') || ~exist('Labels', 'var')
        if ~exist('Gene_Expression', 'var') || ~exist('CancerTypeIndex', 'var')
            load('GE.mat');
            Data = Gene_Expression;
            Labels = CancerTypeIndex-1;
        end
    end

    if ~exist('file_name', 'var')
        file_name = 'Sample_data';
    end

    if exist('N', 'var') && N~=0
        people = randperm(size(Data,1));
        data_selection = people(1:N);
    else
        data_selection = 1:size(Data,1);
    end
    
    if exist('N', 'var') && NGenes ~= 0 
        genes = randperm(size(Data,2));
        selection_genes = genes(1:NGenes);
    else
        selection_genes = 1:size(Data,2);
    end
    
        
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