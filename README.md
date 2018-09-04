
# PrimateAI:  deep residual neural network for classifying the pathogenicity of missense mutations.    

PrimateAI is trained on a dataset of ~380,000 common missense variants from humans and six non-human primate species, using a semi-supervised benign vs unlabeled training regimen.  The input to the network is the amino acid sequence flanking the variant of interest and the orthologous sequence alignments in other species, without any additional human-engineered features, and the output is the pathogenicity score from 0 (less pathogenic) to 1 (more pathogenic).  To incorporate information about protein structure, PrimateAI learns to predict secondary structure and solvent accessibility from amino acid sequence, and includes these as sub-networks in the full model.  The total size of the network, with protein structure included, is 36 layers of convolutions, consisting of roughly 400,000 trainable parameters. 

The method is described in the publication:    
Sundaram, L et al, Predicting the clinical impact of human mutation with deep neural networks.  Nature Genetics 2018. 
https://www.nature.com/articles/s41588-018-0167-z
    

## ARCHITECTURE of DEEP LEARNING NETWORK for PATHOGENICITY PREDICTION    
The pathogenicity prediction network takes as input the 51-length amino acid sequence centered at the variant of interest, and the outputs of the secondary structure and solvent accessibility networks for each variant.  To represent the variant, the network receives both the 51-length reference amino acid sequence ome and the alternative 51-length amino acid sequence with the missense variant substituted in at the central position.  Three 51-length position frequency matrices (PFMs) are generated from multiple sequence alignments of 99 vertebrates, including one for 11 primates, one for 50 mammals excluding primates, and one for 38 vertebrates excluding primates and mammals.  
The five direct input channels are passed through an upsampling convolution layer of 40 kernels with linear activations. The human reference amino acid sequence (1a) is merged with the PFMs from primate, mammal, and vertebrate multiple sequence alignments (Merge 1a).   Similarly, the human alternative amino acid sequence (1b), is merged with the PFMs from primate, mammal, and vertebrate multiple sequence alignments (Merge 1b).  This creates two parallel tracks, one for the reference sequence, and one with the alternate sequence with the variant substituted in.      
The merged feature map of both reference channel and the alternative channel (Merge 1a and Merge 1b) are passed through a series of six residual blocks (Layers 2a to 7a, Merge 2a and Layers 2b to 7b, Merge 2b). The output of the residual blocks (Merge 2a and Merge 2b) are concatenated together to form a feature map of size (51,80) (Merge 3a, Merge 3b) which fully mixes the data from the reference and alternative channels.  Next, the data has two paths for passing through the network in parallel, either through a series of six residual blocks containing two convolutional layers each, as defined in section 2.1 (Merge 3 to 9, Layers 9 to 46 excluding layer 21,34), or via skip connections, which connect the output of every two residual blocks after passing through a 1D convolution (Layer 21, Layer 37, Layer 47).  Finally, the merged activations (Merge 10) are fed to another residual block (layers 48 to 53, Merge 11).  The activations from Merge 11 are given to a 1D convolution with filter size 1 and sigmoid activation (Layer 54, then passed through a global max pooling layer that picks a single value representing the network’s prediction for variant pathogenicity. The detailed architecture is shown in the figure below.   

![Illustration of PrimateAI network](doc/FigureS4.pdf)    

## ARCHITECTURE of DEEP LEARNING NETWORK for SECONDARY STRUCTURE & SOLVENT ACCESSIBILITY
The secondary structure deep learning network predicts 3-state secondary structure at each amino acid position: alpha helix (H), beta sheet (B), and coils (C).  The solvent accessibility network predicts 3-state solvent accessibility at each amino acid position:  buried (B), intermediate (I), and exposed (E).  Both networks only take the flanking amino acid sequence as their inputs, and were trained using labels from known non-redundant crystal structures in the Protein DataBank.  For the input to the pre-trained 3-state secondary structure and 3-state solvent accessibility networks, we used a single PFM matrix generated from the multiple sequence alignments for all 99 vertebrates, also with length 51 and depth 20.  After pre-training the networks on known crystal structures from the Protein DataBank, the final two layers for the secondary structure and solvent models were removed and the output of the network was directly connected to the input of the pathogenicity model.  The best testing accuracy achieved for the 3-state secondary structure prediction model is 79.86 %.        
Both our deep learning network for pathogenicity prediction (PrimateAI) and deep learning networks for predicting secondary structure and solvent accessibility adopted the architecture of residual blocks. The detailed architecture is shown in the figure below.        
   
![Illustration of secondary structure and solvent accessibility networks](doc/FigureS5.pdf)

## LICENSE    
Copyright (c) 2018 Illumina, Inc. All rights reserved.

This software is provided under the terms and conditions of the GNU GENERAL PUBLIC LICENSE Version 3.

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 along with this program. If not, see https://github.com/illumina/licenses/.



## DEMO DATA for DOWNLOADING    
Demo dataset (demodata.zip) can be downloaded from Illumina basespace at https://basespace.illumina.com/s/cPgCSmecvhb4 .

The demo dataset contains 9 files:
1. full_set_snp_info.csv: contains information of all the 70M possible missense SNPs. Generally users can reuse this file even if they prepare their own training, validation, and testing datasets.

   Each row has the following columns:
   
    id: SNP ID  
    chr: chromosome  
    pos: position on hg19  
    ref_nuc: reference nucleotide on hg19  
    ref_codon: reference codon  
    ref_aa: reference amino acid  
    alt_nuc: alternative nucleotide on hg19  
    alt_codon: alternative codon  
    alt_aa: alternative amino acid  
    strand: 1 = positive strand, 0 = negative strand  
    gene_name: UCSC ID for the gene containing this missense variant  
    change_position_1based: the position of amino acid in the protein (gene) where this variant occurrs. Note the position is 1-based.  
    total_length: the length of the protein (gene) in terms of amino acid   
    trinucleotide_bases: the trinucleotid background around this variant  
    label: prior labeling of pathogenicity. Benign and Unknown.  
    species: the species that this variant is observed  
    mirrored_column: used to sample unknown variants to match benign variants.  
    mean_coverage: averaged depth of ExAC data at this variant position  
    mean_coverage_bins: binning the mean coverage  
    
Example input is:  
```
id,chr,pos,ref_nuc,ref_codon,ref_aa,alt_nuc,alt_codon,alt_aa,strand,gene_name,change_position_1based,total_length,trinucleotide_bases,label,species,mirrored_column,mean_coverage,mean_coverage_bins
snp0,chr10,1046704,C,CGT,R,T,TGT,C,1,uc001ift.3,248,635,CCG,Benign,human,0.279792,45.49,50.0
snp1,chr10,1046704,C,CGT,R,G,GGT,G,1,uc001ift.3,248,635,CCG,Unknown,unknown,0.011504,45.49,50.0
snp2,chr10,1046704,C,CGT,R,A,AGT,S,1,uc001ift.3,248,635,CCG,Unknown,unknown,0.013494,45.49,50.0
snp3,chr10,1046705,G,CGT,R,A,CAT,H,1,uc001ift.3,248,635,CGT,Unknown,unknown,0.33432,45.67,50.0

............
```

2. conservation_profile.npy: contains the gene sequence and 99 vertebrate conservation profile for each canonical gene.

   Each row has the following columns:  
     gene_name: UCSC ID for this gene     
     sequence: amino acid sequence of this gene  
     primate: 51-length position frequecy matrix for 11 primates. The dimension of this matrix is 20x51.  
     mammal: 51-length position frequecy matrix for 50 mammals excluding primates. The dimension of this matrix is 20x51.  
     vertebrate: 51-length position frequecy matrix for 38 vertebrates excluding primates and mammals. The dimension of this matrix is 20x51.  
     
     
3. benign_train_snps.txt: contains the list of benign SNP IDs (50K) that are used for training.
4. benign_validation_snps.txt: contains the list of benign SNP IDs (10K) that are used for validation.
5. benign_test_snps.txt: contains the list of benign SNP IDs (10K) that are used for testing.
6. unknown_validation_snps.txt: contains the list of unknown SNP IDs (10K) that are used for validation.
7. unknown_test_snps.txt: contains the list of unknown SNP IDs (10K) that are used for testing.
8. secondary_structure_seqtoseq.hdf5: contains the trained weights for secondary structure prediction model.
9. solvent_accessibility_seqtoseq.hdf5: contains the trained weights for solvent accessibility prediction model.


Users can download extra data on BaseSpace:
https://basespace.illumina.com/s/cPgCSmecvhb4
Or they can prepare their own training, validation, and testing datasets according to the formats of demo datasets.

Users can also download exome-wide predictions of pathogenicity scores (PrimateAI_scores_v0.2.tsv.gz) from BaseSpace:
https://basespace.illumina.com/s/cPgCSmecvhb4



## RUN INSTRUCTIONS    
The Python scripts provided include the deep residual neural network for variant pathogencity estimation, as well as two deep residual neural models for predicting secondary structure and solvent accessibility of amino acids. These models are written using Python Keras package.

To run this script on the demo dataset, users need to set up a deep learning environment on a GPU-server. Pre-install Python packages numpy, scipy, tensorflow, keras, pandas, glob, and multiprocessing. Then download the demo dataset and unzip it. Download the source folder, which contains four Python scripts.

The command to run this PrimateAI script is :
```
python  /path/to/source/PrimateAI_v0.2.py \
    /path/to/demodata/full_set_snp_info.csv \
    /path/to/demodata/conservation_profile.npy
    /path/to/demodata/benign_train_snps.txt
    /path/to/demodata/benign_validation_snps.txt
    /path/to/demodata/unknown_validation_snps.txt
    /path/to/demodata/benign_test_snps.txt
    /path/to/demodata/unknown_test_snps.txt
    /path/to/demodata/secondary_structure_seqtoseq.hdf5
    /path/to/demodata/solvent_accessibility_seqtoseq.hdf5
    /path/to/output/folder/
```
The script trains eight separate neural net models and ensemble them. Thus it paralleles the jobs on 8 GPUs. Users can modify the script to suit their needs.

### OUTPUT FILES
PrimateAI v0.2 will generate two files benign_test.csv and unknown_test.csv to output the prediction results for benign_test_snps.txt and unknown_test_snps.txt. The last column is the ensembled predicted pathogenicity for each variants.

It will also make a directory "current_weights" to store the trained model weights from eight different neural net models.

### EVALUATION OUTPUTS
The script evaluation.py will evaluate the output files benign_test.csv and unknown_test.csv and output the accuracy generated from the test dataset.

```
python  /path/to/source/evaluation.py \
    /path/to/output/folder/
```      

## RELEASE NOTE
Current version of PrimateAI for downloading is v0.2.
