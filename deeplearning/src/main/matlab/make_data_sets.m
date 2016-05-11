function make_data_sets(N_test, N_train, N_genes, Data, Labels, file_name)

    if N_test + N_train < size(Data,1) && N_genes < size(Data,2) && size(Labels,1) == size(Data,1)
        sample_data(N_test, N_genes, Data, Labels, [file_name '.test']);
        sample_data(N_train, N_genes, Data, Labels, [file_name '.train']);
    end
end