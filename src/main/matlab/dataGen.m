function dataGen(data, labels, varargin)
% DATAGEN(data, labels, 'name', 'file_name') creates .dat, .lab, and .meta 
% files based on the data, labels and additional options provided.
% 
% DO NOT CHANGE THIS FILE, COPY IT INSTEAD AND CHANGE THE VERSION

    
%     load the default values
    a = defaults;
    a.observations = size(data,1);
    
    assert(min(labels,1) == 1)
    a.classes = max(labels);
    labels = labels-1;
    
    
    options = fieldnames(a);
    
    %# count arguments
    nArgs = length(varargin);
    if ~isequal(varargin,{{}})
        if rem(nArgs,2) == 0
            for pair = reshape(varargin,2,[]) %# pair is {propName;propValue}
                inpName = lower(pair{1}); %# make case insensitive
                
                if any(strcmp(inpName,options))
                    a.(inpName) = pair{2};
                else
                    error('%s is not a recognized parameter name',inpName)
                end
            end
        else
            error('DATAGEN needs propertyName/propertyValue pairs')
        end
    end
    
    if a.width * a.height ~= size(data,2)
        a.width = size(data,2);
        a.height = 1;
        warning ('Make sure to define width and height correctly. Using width = %i and height = %i for now.', a.width, a.height);
    end
    write_meta_file(version, a);
    write_data_files(a.name, data, labels);
end

function v = version
    v = 1; % yeah it's hardcoded it's supposed to be
end

function d = defaults
    d = struct('name', 'Data', 'width', 0, 'height', 0, 'trainpercent', .70, 'batchsize', 50);
end

function write_meta_file(version, values)

    fileid = fopen([values.name '.meta'], 'w');
    
    fprintf(fileid, '%i\n' , version);
    fprintf(fileid, '%i' , round(clock));
    fprintf(fileid, '\n%i\n' , values.observations);
    fprintf(fileid, '%i\n' , values.width);
    fprintf(fileid, '%i\n' , values.height);
    fprintf(fileid, '%i\n' , values.classes);
    fprintf(fileid, '%f\n' , values.trainpercent);
    fprintf(fileid, '%i\n' , values.batchsize);
    
    fclose(fileid);
end

function write_data_files(file_name, data, labels)
% write data and label file (.dat and .lab)

% make sure the amount of labels corresponds to the amount of observations
    assert(size(data,1) == size(labels,1) && size(labels,1) == numel(labels))
    
    N = 1:size(data,1);
    genes = 1:size(data,2);
    
    
	data_file = fopen([file_name '.dat'], 'w');
	label_file = fopen([file_name '.lab'], 'w');

    if size(data,2) ~=1
        for person = N;
            fprintf(label_file, '%u\r\n' , labels(person));
            fprintf(data_file, '%.4f,' ,data(person, genes(1:end-1)));
            fprintf(data_file, '%.4f' ,data(person, genes(end)));
            fprintf(data_file, '\r\n');
        end
    else
        for person = N;
            fprintf(label_file, '%u\r\n' , labels(person));
            fprintf(data_file, '%.4f\r\n' ,data(person, genes(end)));
        end
    end
	fclose(data_file);
	fclose(label_file);
end

% 
% 
% function res = normdata(Data)
%     %NORMDATA normalises DATA assuming that it consists of rows of observations
%     %and columns of features.
% 
%     %res = bsxfun(@minus,Data,mean(Data));
%     %TODO
% 
% end
% 
% function smthn()
%     Bool_Too_Large = Gene_Expression > 1.5;
%     Bool_Too_Small = Gene_Expression < -1.5;
%     Gene_Expression1 = Gene_Expression;
%     Gene_Expression1(Bool_Too_Large) = 255;
%     Gene_Expression1(Bool_Too_Small) = 0;
%     Gene_Expression1(~Bool_Too_Large & ~Bool_Too_Small) = (Gene_Expression1(~Bool_Too_Large & ~Bool_Too_Small) + 1.5)*255/3;
%     Gene_Expression1 = round(Gene_Expression1);
%     uint8(Gene_Expression1);
% end
