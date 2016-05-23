ges = std(Gene_Expression);
select = ges >1.2982;
geneAct = Gene_Expression(:,select);
%geneAct = Gene_Expression;
label = CancerTypeIndex == 3;
absCorr = abs(corr(geneAct));
%absCorr = corr(geneAct);
numGenes = 100;
numPatients = 16;
numGroupPatients = 101;

h_array = zeros(1,100);
p_array = zeros(1,100);
total_mean_1 = zeros(1,100);
total_mean_avg = zeros(1,100);

for t = 1:100
counter = 1;
total_acc_1 = zeros(1,numGroupPatients*numGenes);
total_acc_avg = zeros(1,numGroupPatients*numGenes);
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
        
        total_acc_1(1,counter) = acc_1;
        total_acc_avg(1,counter) = acc_avg;
        group_acc_1(j) = acc_1;
        group_acc_avg(j) = acc_avg;
        counter = counter + 1;
    end 
    
    std_1(i) = std(group_acc_1/100);
    std_avg(i) = std(group_acc_avg/100);
    
end

std_total_1 = std(total_acc_1/100);
std_total_avg = std(total_acc_avg/100);
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
TOTAL_MEAN_AVG_GENE = mean(total_mean_avg)




