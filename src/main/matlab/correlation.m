geneAct = Gene_Expression;
label = CancerTypeIndex == 2;
absCorr = abs(corr(geneAct));
%absCorr = corr(geneAct);
numGenes = 5000;
numPatients = 25;
numGroupPatients = 50;

counter = 1;
total_acc_1 = zeros(1,numGroupPatients*numGenes);
total_acc_avg = zeros(1,numGroupPatients*numGenes);
std_1 = zeros(1,numGenes);
std_avg = zeros(1,numGenes);


for i = 1:numGenes
    point = ceil(size(absCorr,1)*rand());

    column = absCorr(:,point);
    [sortedValues, sortedIndex] = sort(column,'descend');

    fourpoints = sortedIndex(1:4);
    avg_act = mean(geneAct(:,fourpoints),2);
    g1_act = geneAct(:,sortedIndex(1));
    temp = horzcat(g1_act,avg_act);
    temp = horzcat(temp,label);
    group_acc_1 = zeros(1,numGroupPatients);
    group_acc_avg = zeros(1,numGroupPatients);
    for j = 1:numGroupPatients
        indexes = randperm(size(temp,1),numPatients);
        activations = temp(indexes,:);
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
[h,p] = ttest(std_1,std_avg)
subplot(2,1,1);
histogram(std_1)
subplot(2,1,2);
histogram(std_avg)




