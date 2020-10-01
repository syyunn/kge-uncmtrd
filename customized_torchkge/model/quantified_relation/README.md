# Quantified Relation
I tweaked the vanilla TransE model so that the model can get `quantity` input for the relation. 

* Quantity means Trade Volume in the UNComtrade data

The relation embedding vector is keep normalized as L2-norm 1, then scaled with min-max normalized trade volume before translate head (h) to estimated point of tail (t) 
- h +r ~ t 


