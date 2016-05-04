%actual normalisation
%should we do this?
GE = bsxfun(@rdivide,bsxfun(@minus,Gene_Expression,mean(Gene_Expression)),std(Gene_Expression));

%cut-off at 3 * standard deviation
Bool_Too_Large = Gene_Expression > 3;
Bool_Too_Small = Gene_Expression < -3;
GE(Bool_Too_Large) = 3;
GE(Bool_Too_Small) = -3;
GE = bsxfun(@rdivide, GE+3,6);