%Function which calculates the maximum accuracy based on a simple threshold
function t = threshold(act, label)
stepsize = abs(max(act) - min(act))/100;
thresh = min(act);

currentAcc = 0;
for i = 1:100
    %Make a prediction about the label
    pred = act >= thresh;
    %Check if correct.
    res = pred == label;
    %Total accuracy
    resAcc = sum(res);
    %Check if it's a new maximum.
    if resAcc > currentAcc
        currentAcc = resAcc;
    end
    %New threshhold by increasing with the stepsize.
    thresh = thresh + stepsize;
end
t = currentAcc;
end