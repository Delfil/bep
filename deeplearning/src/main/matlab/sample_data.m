function sample_data(N, NGenes, Data, Labels, file_name, image_size)

    if ~exist('file_name', 'var')
        file_name = 'Sample_data';
    end

    people = randperm(size(Data,1));
    
    
    if NGenes > 0 && NGenes <= size(Data,2)
        genes = randperm(size(Data,2));
        selection_genes = genes(1:NGenes);
    else
        selection_genes = 1:size(Data,2);
    end
    
    if size(N,2)>1 && min(N)>0 && max(N) < size(Data,1)
        data_selection = N;
    elseif N>0 && N<size(Data,1)
        data_selection = people(1:N);
    else
        data_selection = 1:size(Data,1);
    end
    
    write_files(file_name , data_selection, selection_genes, Data, Labels, image_size);
        
end

function write_files(file_name, N, genes, Data, Label, image_size)
	write_data_files(file_name, N, genes, Data, Label);
	write_meta_file(file_name, N, genes, Data, Label, image_size);
end
function write_data_files(file_name, N, genes, Data, Label)
	data_file = fopen([file_name '.dat'], 'w');
	label_file = fopen([file_name '.lab'], 'w');

	for person = N;
		fprintf(label_file, '%u\r\n' , Label(person));
		fprintf(data_file, '%.4f,' ,Data(person, genes));
		fprintf(data_file, '%.4f' ,Data(person, genes(end)));
		fprintf(data_file, '\r\n');
	end
	fclose(data_file);
	fclose(label_file);
end

function write_meta_file(file_name, N, ~, ~, Label, image_size)
    if size(image_size,2) ==  2
        height = image_size(1);
        width = image_size(2);
    else
        error('the image size is wrong');
    end

    fileid = fopen([file_name '.meta'], 'w');
    %write amount of rows, width and height of the image and the number of
    %classifications
    fprintf(fileid, '%i\n%i\n%i\n%i\n' , size(N,2),width, height, numel(unique(Label)));
    fclose(fileid);

end