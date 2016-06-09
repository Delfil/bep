%Algorithm running 100 times two t-test on both accuracy and std based on
%100 genes each with 101 groups of 16 patients.

%Here we select our 100 genes
ges = std(Gene_Expression);
select = ges > 1;
geneAct = Gene_Expression(:,select);
%This is our label array with 1 being label 3 and 0 being not 3.
label1 = CancerTypeIndex == 1;
label2 = CancerTypeIndex == 2;
label3 = CancerTypeIndex == 3;
label4 = CancerTypeIndex == 4;
label5 = CancerTypeIndex == 5;
label = [label1,label2,label3,label4,label5];

%Calculate the absolute correlations
absCorr = corr(geneAct);

%Number of times we would like to run this algorithm
numRuns = 100;
%The number of genes each run
numGenes = 334;
%Size of the group of patients
numPatients = 100;
%Number of groups each gene.
numGroupPatients = 100;

%Arrays for keeping track of h and p values of t-test
h_array = zeros(1,numRuns);
hacc_array = zeros(1,numRuns);
p_array = zeros(1,numRuns);
pacc_array = zeros(1,numRuns);
%Arrays for keeping track of the stds
total_mean_1 = zeros(1,numRuns);
total_mean_avg = zeros(1,numRuns);
%Arrays for keeping track of the overall accuracies.
total_acc_1 = zeros(1,numRuns);
total_acc_avg = zeros(1,numRuns);

for t = 1:numRuns
    counter = 1;
    %Arrays for the mean standard deviation of each of the genes.
    std_1 = zeros(1,numGenes);
    std_avg = zeros(1,numGenes);
    acc_1 = zeros(1,numGenes);
    acc_avg = zeros(1,numGenes);
    %Random order of the genes.
    genes = randperm(size(absCorr,1),numGenes);
    runLabel = label(:,randperm(size(label,2),1));
    
    for i = 1:numGenes    
        point = genes(i);
        
        %Sort the points to find the three most correlating points
        column = absCorr(:,point);
        [sortedValues, sortedIndex] = sort(column,'descend');
        
        %Select four correlating points
        fourpoints = sortedIndex(1:4);
        four_randompoints = randperm(size(column,1),4);
        avg_act = mean(geneAct(:,fourpoints),2);
        g1_act = mean(geneAct(:,four_randompoints),2);
        %Array to pass on to easily split in groups
        temp = [g1_act,avg_act,label(:,randperm(size(label,2),1))];
        %Arrays containing all the maximum accuracies for each group
        group_acc_1 = zeros(1,numGroupPatients);
        group_acc_avg = zeros(1,numGroupPatients);
        for j = 1:numGroupPatients
            %Select 16 random patients
            indeces = randperm(size(temp,1),numPatients);
            activations = temp(indeces,:);
            %Get maximum accuracies.
            temp_acc_1 = threshold(activations(:,1),activations(:,3));
            temp_acc_avg = threshold(activations(:,2),activations(:,3));
            %Add to array
            group_acc_1(j) = temp_acc_1;
            group_acc_avg(j) = temp_acc_avg;
            counter = counter + 1;
        end
        %Add the standard deviation of all the maximum accuracies to the
        %arrays.
        std_1(i) = std(group_acc_1/numPatients);
        std_avg(i) = std(group_acc_avg/numPatients);
        %Add the average accuracies to the arrays
        acc_1(i) = mean(group_acc_1/numPatients);
        acc_avg(i) = mean(group_acc_avg/numPatients);
        
        
    end
    %Run left tail t-test on the accuracy as we hope to improve accuracy
    [hacc,pacc] = ttest(acc_1,acc_avg, 'Tail', 'left');
    %Run right tail t-test on the std as we hope to minimize std.
    [h,p] = ttest(std_1,std_avg, 'Tail', 'right');
    %Add to correct array
    h_array(t) = h;
    p_array(t) = p;
    hacc_array(t) = hacc;
    pacc_array(t) = pacc;
    %Keeping track of the average std for each run
    total_mean_1(t) = mean(std_1);
    total_mean_avg(t) = mean(std_avg);
    total_acc_1(t) = mean(acc_1);
    total_acc_avg(t) = mean(acc_avg);
    
end

%Easy output
t_test_correct = sum(h_array)/100
TOTAL_MEAN_ONE_GENE = mean(total_mean_1)
TOTAL_MEAN_AVG_GENE = mean(total_mean_avg)
TOTAL_ACC_ONE_GENE = mean(total_acc_1)
TOTAL_ACC_AVG_GENE = mean(total_acc_avg)