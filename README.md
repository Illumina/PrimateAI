
PrimateAI:  deep residual neural network for classifying the pathogenicity of missense mutations.    

PrimateAI is trained on a dataset of ~380,000 common missense variants from humans and six non-human primate species, using a semi-supervised benign vs unlabeled training regimen.  The input to the network is the amino acid sequence flanking the variant of interest and the orthologous sequence alignments in other species, without any additional human-engineered features, and the output is the pathogenicity score from 0 (less pathogenic) to 1 (more pathogenic).  To incorporate information about protein structure, PrimateAI learns to predict secondary structure and solvent accessibility from amino acid sequence, and includes these as sub-networks in the full model.  The total size of the network, with protein structure included, is 36 layers of convolutions, consisting of roughly 400,000 trainable parameters. 

The method is described in the publication:    
Sundaram, L et al, Predicting the clinical impact of human mutation with deep neural networks.  Nature Genetics 2018.

    
ARCHITECTURE of DEEP LEARNING NETWORK    
The pathogenicity prediction network takes as input the 51-length amino acid sequence centered at the variant of interest, and the outputs of the secondary structure and solvent accessibility networks for each variant.  To represent the variant, the network receives both the 51-length reference amino acid sequence ome and the alternative 51-length amino acid sequence with the missense variant substituted in at the central position.  Three 51-length position frequency matrices (PFMs) are generated from multiple sequence alignments of 99 vertebrates, including one for 11 primates, one for 50 mammals excluding primates, and one for 38 vertebrates excluding primates and mammals.      
The secondary structure deep learning network predicts 3-state secondary structure at each amino acid position: alpha helix (H), beta sheet (B), and coils (C).  The solvent accessibility network predicts 3-state solvent accessibility at each amino acid position:  buried (B), intermediate (I), and exposed (E).  Both networks only take the flanking amino acid sequence as their inputs, and were trained using labels from known non-redundant crystal structures in the Protein DataBank.  For the input to the pre-trained 3-state secondary structure and 3-state solvent accessibility networks, we used a single PFM matrix generated from the multiple sequence alignments for all 99 vertebrates, also with length 51 and depth 20.  After pre-training the networks on known crystal structures from the Protein DataBank, the final two layers for the secondary structure and solvent models were removed and the output of the network was directly connected to the input of the pathogenicity model.  The best testing accuracy achieved for the 3-state secondary structure prediction model is 79.86 %. There was no substantial difference when comparing the predictions of the neural network when using DSSP-annotated72,73 structure labels for the approximately ~4000 human proteins that had crystal structures, versus using predicted structure labels only.    
Both our deep learning network for pathogenicity prediction (PrimateAI) and deep learning networks for predicting secondary structure and solvent accessibility adopted the architecture of residual blocks.


LICENSE    
Copyright (c) 2018 Illumina, Inc. All rights reserved.

This software is provided under the terms and conditions of the GNU GENERAL PUBLIC LICENSE Version 3

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 along with this program. If not, see https://github.com/illumina/licenses/.


INSTRUCTIONS    
The Python script provided includes the deep residual neural network for variant pathogencity estimation, as well as two deep residual neural models for predicting secondary structure and solvent accessibility of amino acids.

To run this script, users need to pre-install Python packages numpy, tensorflow, and keras. In their Python script, they can import this script to adopt PrimateAI models. 


DATA for DOWNLOADING    
Users can download labeled training data on BaseSpace:
https://basespace.illumina.com/s/cPgCSmecvhb4
Or they can prepare their own training, validation, and testing datasets.

Users can also download exome-wide predictions of pathogenicity scores from BaseSpace:
https://basespace.illumina.com/s/cPgCSmecvhb4


