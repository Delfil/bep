function t = threshold(act, label)
stepsize = abs(max(act) - min(act))/100;
thresh = min(act);

currentAcc = 0;
for i = 1:100
    pred = act >= thresh;
    res = pred == label;
    resAcc = sum(res);
    if resAcc > currentAcc
        currentAcc = resAcc;
    end
    thresh = thresh + stepsize;
end
t = currentAcc;
end