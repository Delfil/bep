ges = std(Gene_Expression);
select = ges >1.2982;
geneAct = Gene_Expression(:,select);
label = CancerTypeIndex == 3;
absCorr = abs(corr(geneAct));

numRuns = 100;
numGenes = 100;
numPatients = 16;
numGroupPatients = 101;

h_array = zeros(1,numRuns);
p_array = zeros(1,numRuns);
total_mean_1 = zeros(1,numRuns);
total_mean_avg = zeros(1,numRuns);
total_acc_1 = zeros(1,numRuns*numGenes);
total_acc_avg = zeros(1,numRuns*numGenes);
counter_acc = 1;

for t = 1:numRuns
counter = 1;
std_1 = zeros(1,numGenes);
std_avg = zeros(1,numGenes);
genes = randperm(size(absCorr,1),numGenes);

for i = 1:numGenes
    point = genes(i);

    column = absCorr(:,point);
    [sortedValues, sortedIndex] = sort(column,'descend');

    fourpoints = sortedIndex(1:9);
    avg_act = mean(geneAct(:,fourpoints),2);
    g1_act = geneAct(:,sortedIndex(1));
    temp = [g1_act,avg_act,label];
    group_acc_1 = zeros(1,numGroupPatients);
    group_acc_avg = zeros(1,numGroupPatients);
    for j = 1:numGroupPatients
        indeces = randperm(size(temp,1),numPatients);
        activations = temp(indeces,:);
        acc_1 = threshold(activations(:,1),activations(:,3));
        acc_avg = threshold(activations(:,2),activations(:,3));
      
        group_acc_1(j) = acc_1;
        group_acc_avg(j) = acc_avg;
        counter = counter + 1;
    end 
    
    std_1(i) = std(group_acc_1/numPatients);
    std_avg(i) = std(group_acc_avg/numPatients);
    total_acc_1(counter_acc) = mean(group_acc_1);
    total_acc_avg(counter_acc) = mean(group_acc_avg);
    counter_acc = counter_acc + 1;
    
end
[h,p] = ttest(std_1,std_avg);
h_array(t) = h;
p_array(t) = p;
total_mean_1(t) = mean(std_1);
total_mean_avg(t) = mean(std_avg);

end
%subplot(2,1,1);
%histogram(std_1)
%subplot(2,1,2);
%histogram(std_avg)

t_test_correct = sum(h_array)/100
TOTAL_MEAN_ONE_GENE = mean(total_mean_1)
TOTAL_MEAN_AVG_GENE = mean(total_mean_avg)
TOTAL_ACC_ONE_GENE = mean(total_acc_1)
TOTAL_ACC_AVG_GENE = mean(total_acc_avg)