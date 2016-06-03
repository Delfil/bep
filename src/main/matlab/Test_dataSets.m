function [M, data] = Test_dataSets(TwoD, a, b)

if ~exist('a', 'var') || ~exist('b','var')

path = '../../../datasets';

if TwoD
    datasets_path = struct(  'berend_final', [path, '/', 'berend_final/berend_final.dat'],...
                        'random_final', [path, '/', 'random_final/random_final.dat']);
else
    datasets_path = struct(...
                        'custer', [path, '/', '1dim_cluster/1dim_cluster.dat'],...
                        'random', [path, '/', '1dim_random/Randomly_Ordered_100_Genes.dat'],...
                        'tsne', [path, '/', 'TSNE_1D_100_genes/TSNE_1D_100_genes.dat'],...
                        'tsne_inv', [path, '/', 'TSNE_1D_inv_corrs_100_genes/TSNE_1D_inv_corrs_100_genes.dat']);
end



data = struct();
fn = fieldnames(datasets_path);
for i = 1:numel(fn)
    data.(fn{i}) = importdata(datasets_path.(fn{i}), ',');
end

M = cell(size(fn)+1);

else
    M = cell(3);
    data = struct('a', a, 'b', b);
    fn = fieldnames(data);
end

for i = 1:numel(fn)+1
    for j = 1:numel(fn)+1
        if i == 1 && j == 1
        elseif i == 1
            M{i,j} = fn{j-1};
        elseif j == 1
            M{i,j} = fn{i-1};
        else
            M{i,j} = 0;
            for k = 1:size(data.(fn{i-1}),2)
                M{i,j} = M{i,j} + max(min(...
                        abs(...
                            bsxfun(@minus, data.(fn{i-1})(:,k),data.(fn{j-1}))...
                        ) < 1e-3...
                    ));
            end
        end
    end
end

end