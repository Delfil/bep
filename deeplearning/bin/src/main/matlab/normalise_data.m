Bool_Too_Large = Gene_Expression > 1.5;
Bool_Too_Small = Gene_Expression < -1.5;
Gene_Expression1 = Gene_Expression;
Gene_Expression1(~Bool_Too_Large & ~Bool_Too_Small) = (Gene_Expression1(~Bool_Too_Large & ~Bool_Too_Small) + 1.5)/3;