function t = threshold(act, label)
stepsize = abs(max(act) - min(act))/100;
thresh = min(act);

currentAcc = 0;
currentThresh = min(act);
for i = 1:100
    pred = act >= thresh;
    res = pred == label;
    resAcc = sum(res);
    if resAcc > currentAcc
        currentAcc = resAcc;
        currentThresh = thresh;
    end
    thresh = thresh + stepsize;
end
t = currentThresh;
end