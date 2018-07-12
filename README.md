
PrimateAI:  deep residual neural network for classifying the pathogenicity of missense mutations.

PrimateAI is trained on a dataset of ~380,000 common missense variants from humans and six non-human primate species, using a semi-supervised benign vs unlabeled training regimen.  The input to the network is the amino acid sequence flanking the variant of interest and the orthologous sequence alignments in other species, without any additional human-engineered features, and the output is the pathogenicity score from 0 (less pathogenic) to 1 (more pathogenic).  To incorporate information about protein structure, PrimateAI learns to predict secondary structure and solvent accessibility from amino acid sequence, and includes these as sub-networks in the full model.  The total size of the network, with protein structure included, is 36 layers of convolutions, consisting of roughly 400,000 trainable parameters. 

The method is described in the publication:
Sundaram, L et al, Predicting the clinical impact of human mutation with deep neural networks.  Nature Genetics 2018.

Labeled training data, and genome-wide predictions are available on BaseSpace:
https://basespace.illumina.com/s/cPgCSmecvhb4

