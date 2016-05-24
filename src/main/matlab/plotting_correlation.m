
h(1) = subplot(2,1,1);
title(h(1),'Distribution of standard deviation of accurasy of one gene');
histogram(total_mean_1);
h(2) = subplot(2,1,2);
title(h(2),'Distribution of standard deviation of accurasy of the average of four correlating genes');
histogram(total_mean_avg);
linkaxes(h);
xlim([0.084 0.091]);
figure;
g(1) = subplot(2,1,1);
title(g(1),'Distribution of standard deviation of accurasy of one gene');
histogram(total_acc_1, 40);
g(2) = subplot(2,1,2);
title(g(2),'Distribution of standard deviation of accurasy of the average of four correlating genes');
histogram(total_acc_avg, 40);
linkaxes(g, 'x');
%xlim([0.084 0.091]);