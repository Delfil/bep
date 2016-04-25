Bool_Too_Large = Gene_Expression > 1.5;
Bool_Too_Small = Gene_Expression < -1.5;
Gene_Expression1 = Gene_Expression;
Gene_Expression1(Bool_Too_Large) = 255;
Gene_Expression1(Bool_Too_Small) = 0;
Gene_Expression1(~Bool_Too_Large & ~Bool_Too_Small) = (Gene_Expression1(~Bool_Too_Large & ~Bool_Too_Small) + 1.5)*255/3;
Gene_Expression1 = round(Gene_Expression1);
uint8(Gene_Expression1);