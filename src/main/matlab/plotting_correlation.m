%In this file we plot four figures; two boxplots of the p-values of the two 
%t-tests and two histograms. One containing std, the other accuracy.

%Here we plot the p-values of both t-test we ran 100 times.
figure;
boxplot(-log(p_array));
title('-log(p) of the total runs');
figure;
boxplot(-log(pacc_array));
title('-log(p) of the total runs');

%Plot of the standard deviation for both distributions as histogram. 
figure;
histogram(total_mean_1, 24);
hold on; histogram(total_mean_avg, 24);
title('Means standard deviation of the 100 runs');
legend('Single gene', '4 correlating genes');

%Plot of the accuracy for both distributions as histogram.
figure;
histogram(total_acc_1, 10);
hold on; histogram(total_acc_avg, 10);
title('Means accuracy of the 100 runs');
legend('Single gene', '4 correlating genes');