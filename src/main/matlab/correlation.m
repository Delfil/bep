test = [0 1 5 1; 1 3 1 1; 3 1 0 5; 0 1 0 1];
label = [1;0;1;1];
absCorr = abs(corr(test));

for i = 1:1
    point = ceil(size(absCorr,1)*rand());

    column = absCorr(:,point);
    [sortedValues, sortedIndex] = sort(column,'descend');

    fourpoints = sortedIndex(1:4);
    avg_act = mean(test(:,fourpoints),2);
    g1_act = test(:,sortedIndex(1));
    temp = horzcat(g1_act,avg_act);
    temp = horzcat(temp,label);
    for j = 1:5
        indexes = randperm(size(temp,1),2);
        activations = temp(indexes,:);
        acc_1 = threshold(activations(:,1),activations(:,3))
        acc_avg = threshold(activations(:,2),activations(:,3))
    end 
    
    
end




