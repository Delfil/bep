%Here we select our 100 genes
ges = std(Gene_Expression);
select = ges >1.2982;
geneAct = Gene_Expression(:,select);
%This is our label array with 1 being label 3 and 0 being not 3.
label = CancerTypeIndex == 3;
%Calculate the absolute correlations
absCorr = abs(corr(geneAct));

%Number of times we would like to run this algorithm
numRuns = 100;
%The number of genes each run
numGenes = 100;
%Size of the group of patients
numPatients = 16;
%Number of groups each gene.
numGroupPatients = 101;

%Arrays for keeping track of h and p values of t-test
h_array = zeros(1,numRuns);
p_array = zeros(1,numRuns);
%Arrays for keeping track of the stds
total_mean_1 = zeros(1,numRuns);
total_mean_avg = zeros(1,numRuns);
%Arrays for keeping track of the overall accuracies.
total_acc_1 = zeros(1,numRuns*numGenes);
total_acc_avg = zeros(1,numRuns*numGenes);
counter_acc = 1;

for t = 1:numRuns
    counter = 1;
    %Arrays for the mean standard deviation of each of the genes.
    std_1 = zeros(1,numGenes);
    std_avg = zeros(1,numGenes);
    %Random order of the genes.
    genes = randperm(size(absCorr,1),numGenes);
    
    for i = 1:numGenes
        point = genes(i);
        
        %Sort the points to find the three most correlating points
        column = absCorr(:,point);
        [sortedValues, sortedIndex] = sort(column,'descend');
        
        %Select four correlating points
        ninepoints = sortedIndex(1:4);
        avg_act = mean(geneAct(:,ninepoints),2);
        g1_act = geneAct(:,sortedIndex(1));
        %Array to pass on to easily split in groups
        temp = [g1_act,avg_act,label];
        %Arrays containing all the maximum accuracies for each group
        group_acc_1 = zeros(1,numGroupPatients);
        group_acc_avg = zeros(1,numGroupPatients);
        for j = 1:numGroupPatients
            %Select 16 random patients
            indeces = randperm(size(temp,1),numPatients);
            activations = temp(indeces,:);
            %Get maximum accuracies.
            acc_1 = threshold(activations(:,1),activations(:,3));
            acc_avg = threshold(activations(:,2),activations(:,3));
            %Add to array
            group_acc_1(j) = acc_1;
            group_acc_avg(j) = acc_avg;
            counter = counter + 1;
        end
        %Add the standard deviation of all the maximum accuracies to the
        %arrays.
        std_1(i) = std(group_acc_1/numPatients);
        std_avg(i) = std(group_acc_avg/numPatients);
        %Add the average accuracies to the arrays
        total_acc_1(counter_acc) = mean(group_acc_1);
        total_acc_avg(counter_acc) = mean(group_acc_avg);
        counter_acc = counter_acc + 1;
        
    end
    %Run t-test
    [h,p] = ttest(std_1,std_avg);
    h_array(t) = h;
    p_array(t) = p;
    %Keeping track of the average std for each run
    total_mean_1(t) = mean(std_1);
    total_mean_avg(t) = mean(std_avg);
    
end

%Easy output
t_test_correct = sum(h_array)/100
TOTAL_MEAN_ONE_GENE = mean(total_mean_1)
TOTAL_MEAN_AVG_GENE = mean(total_mean_avg)
TOTAL_ACC_ONE_GENE = mean(total_acc_1)
TOTAL_ACC_AVG_GENE = mean(total_acc_avg)