# Copyright 2018 Illumina Inc, San Diego, CA
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see https://www.gnu.org/licenses/gpl-3.0.txt


import pandas as pd 
import numpy as np 
import numpy
import sys
import multiprocessing
from multiprocessing import Process,Manager
import time
import subprocess
import os 
import glob
import math
from scipy.stats import ranksums
import re
from model import primateAI_model
from preprocess_data import data_preprocessing,get_benign_counts,get_mirrored
seed_val=128310


def evaluation(arr,X_train_conservation_primates,X_train_conservation_mammals,X_train_conservation_otherspecies,X_train_orig_1,X_train_snp_1,private_args,conn,weight_ss_file, weight_sa_file):
  import os
  os.environ["CUDA_VISIBLE_DEVICES"]=private_args
  import keras
  model=primateAI_model(weight_ss_file, weight_sa_file)
  model.load_weights(arr)
  predict=model.predict({'orig_seq':X_train_orig_1 ,'snp_seq':X_train_snp_1, 'conservation_primates' : X_train_conservation_primates , 'conservation_mammals' : X_train_conservation_mammals, 'conservation_otherspecies' : X_train_conservation_otherspecies },verbose=0,batch_size=5000)
  conn.send(predict.squeeze())
  conn.close()


def iteration_processing(benign_train_dataset,required_data,seed_val,iteration_num,private_args,process_name,weight_ss_file, weight_sa_file, out_dir):
    seed_val=seed_val+np.random.randint(1,1998)
    start_time = time.time()
    print 'this ' + process_name + ' has started'
    #benign_train_dataset.to_csv('./evaluation_files/benign_train_dataset_'+process_name+'_'+str(j)+'.csv',index=False)
    rest_pathogenic=get_mirrored(required_data,benign_train_counts_human,benign_train_counts_otherspecies,seed_val,'mirroring')
    benign_train_dataset=benign_train_dataset.append(rest_pathogenic)
    print benign_train_dataset.shape
    m=iteration_num
    requried_data=''
    benign_train_dataset=benign_train_dataset.sample(frac=1)
    while True:
      try:
        data=data_preprocessing(benign_train_dataset,conservation_sequence_data.copy(),process_name)
        break
      except Exception as e:
        print str(e)
    benign_train_dataset=''
    stopping_criteria=0
    old_pvalue=100
    old_training_statistics=0
    y_train=data[5]
    X_train_orig_1=data[0]
    X_train_snp_1=data[1]
    X_train_conservation_primates=data[2]
    X_train_conservation_mammals=data[3]
    X_train_conservation_otherspecies=data[4]
    print ' training shapes orig : ' + str(X_train_orig_1.shape) + ' snp ' + str(X_train_snp_1.shape) + ' labels ' + str(y_train.shape) + ' conservation primates' + str(X_train_conservation_primates.shape) + ' conservation mammals' + str(X_train_conservation_mammals.shape) + ' conservation otherspecies' + str(X_train_conservation_otherspecies.shape)
    print process_name+' '+str((start_time-time.time())/60.0)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=private_args
    import keras
    model=primateAI_model(weight_ss_file, weight_sa_file)
    while True:  
      model.fit({'orig_seq':X_train_orig_1 ,'snp_seq':X_train_snp_1, 'conservation_primates' : X_train_conservation_primates , 'conservation_mammals' : X_train_conservation_mammals, 'conservation_otherspecies' : X_train_conservation_otherspecies } ,{'output':y_train},verbose=0,batch_size=2500,epochs=1)
      unknown_pred=model.predict({'orig_seq':X_valid_orig_1 ,'snp_seq':X_valid_snp_1, 'conservation_primates' : conservation_valid_primates , 'conservation_mammals' : conservation_valid_mammals, 'conservation_otherspecies' : conservation_valid_otherspecies },verbose=0,batch_size=5000 )
      benign_pred=model.predict({'orig_seq':X_valid_orig_1_benign ,'snp_seq':X_valid_snp_1_benign, 'conservation_primates' : conservation_valid_primates_benign , 'conservation_mammals' : conservation_valid_mammals_benign, 'conservation_otherspecies' : conservation_valid_otherspecies_benign },verbose=0,batch_size=5000 )
      training_pvalue=ranksums(unknown_pred,benign_pred)
      if training_pvalue[1]!=0.0:
        pvalue=training_pvalue[1]
        old_training_statistics=training_pvalue[0]
        if pvalue<old_pvalue:
          old_pvalue=pvalue
          print pvalue
          model.save_weights(out_dir + "./current_weights/weights"+str(m)+".hdf5")
          print 'saving model to current_weights/weights'+str(m)+".hdf5"
          stopping_criteria=0
        else:
          stopping_criteria+=1
      else:
        new_training_statistics=training_pvalue[0]
        if new_training_statistics>old_training_statistics:
          old_training_statistics=new_training_statistics
          model.save_weights(out_dir + "./current_weights/weights"+str(m)+".hdf5")
          print old_training_statistics
          print 'saving model current_weights/weights'+str(m)+".hdf5"
          stopping_criteria=0
        else:
          stopping_criteria+=1
      if stopping_criteria==5:
        break
    print process_name + ' the model run time is ' + str((start_time-time.time())/60.0)
    print ' the process ' + process_name + ' is done'



def evaluation_data(evaluation_variants,outfile_name, out_dir):
  name=out_dir + "./current_weights/weights*.hdf5"
  data=data_preprocessing(evaluation_variants,conservation_sequence_data,'evaluation_data')
  X_train_orig_1=data[0]
  X_train_snp_1=data[1]
  X_train_conservation_primates=data[2]
  X_train_conservation_mammals=data[3]
  X_train_conservation_otherspecies=data[4]
  evaluation_labels=evaluation_variants['id'].as_matrix()
  models=glob.glob(name)
  models.sort()
  p1_conn,c1_conn = multiprocessing.Pipe()
  p2_conn,c2_conn = multiprocessing.Pipe()
  p3_conn,c3_conn = multiprocessing.Pipe()
  p4_conn,c4_conn = multiprocessing.Pipe()
  p5_conn,c5_conn = multiprocessing.Pipe()
  p6_conn,c6_conn = multiprocessing.Pipe()
  p7_conn,c7_conn = multiprocessing.Pipe()
  p8_conn,c8_conn = multiprocessing.Pipe()
  p1 = Process(target=evaluation, args=(models[0],X_train_conservation_primates,X_train_conservation_mammals,X_train_conservation_otherspecies,X_train_orig_1,X_train_snp_1,'0',c1_conn,weight_ss_file, weight_sa_file))
  p2 = Process(target=evaluation, args=(models[1],X_train_conservation_primates,X_train_conservation_mammals,X_train_conservation_otherspecies,X_train_orig_1,X_train_snp_1,'1',c2_conn,weight_ss_file, weight_sa_file))
  p3 = Process(target=evaluation, args=(models[2],X_train_conservation_primates,X_train_conservation_mammals,X_train_conservation_otherspecies,X_train_orig_1,X_train_snp_1,'2',c3_conn,weight_ss_file, weight_sa_file))
  p4 = Process(target=evaluation, args=(models[3],X_train_conservation_primates,X_train_conservation_mammals,X_train_conservation_otherspecies,X_train_orig_1,X_train_snp_1,'3',c4_conn,weight_ss_file, weight_sa_file))
  p5 = Process(target=evaluation, args=(models[4],X_train_conservation_primates,X_train_conservation_mammals,X_train_conservation_otherspecies,X_train_orig_1,X_train_snp_1,'4',c5_conn,weight_ss_file, weight_sa_file))
  p6 = Process(target=evaluation, args=(models[5],X_train_conservation_primates,X_train_conservation_mammals,X_train_conservation_otherspecies,X_train_orig_1,X_train_snp_1,'5',c6_conn,weight_ss_file, weight_sa_file))
  p7 = Process(target=evaluation, args=(models[6],X_train_conservation_primates,X_train_conservation_mammals,X_train_conservation_otherspecies,X_train_orig_1,X_train_snp_1,'6',c7_conn,weight_ss_file, weight_sa_file))
  p8 = Process(target=evaluation, args=(models[7],X_train_conservation_primates,X_train_conservation_mammals,X_train_conservation_otherspecies,X_train_orig_1,X_train_snp_1,'7',c8_conn,weight_ss_file, weight_sa_file))
  p1.start()
  p2.start()
  p3.start()
  p4.start()
  p5.start()
  p6.start()
  p7.start()
  p8.start()
  data1=p1_conn.recv()
  data2=p2_conn.recv()
  data3=p3_conn.recv()
  data4=p4_conn.recv()
  data5=p5_conn.recv()
  data6=p6_conn.recv()
  data7=p7_conn.recv()
  data8=p8_conn.recv()
  p1.join()
  p2.join()
  p3.join()
  p4.join()
  p5.join()
  p6.join()
  p7.join()
  p8.join()
  cycle_labels=pd.concat([evaluation_variants.reset_index().drop('index',axis=1),pd.Series(data1.squeeze()),pd.Series(data2.squeeze()),pd.Series(data3.squeeze()),pd.Series(data4.squeeze()),pd.Series(data5.squeeze()),pd.Series(data6.squeeze()),pd.Series(data7.squeeze()),pd.Series(data8.squeeze())],axis=1)
  cycle_labels['mean']=np.mean(cycle_labels.iloc[:,-8:],axis=1)
  cycle_labels.to_csv(out_dir + outfile_name,index=False)






input_fulldata_file=sys.argv[1] #/illumina/scratch/DeepLearning/lsundaram/new_datasources/full_data_coverage_species.csv
conservation_file=sys.argv[2]  #'/illumina/scratch/DeepLearning/lsundaram/data_sources/conservation_without_msa_full.npy'
benign_train_file=sys.argv[3]  #/illumina/scratch/DeepLearning/hgao/SNPDL/PrimateAI/inputdata/benign_train_snps.csv
benign_validation_file=sys.argv[4]  #/illumina/scratch/DeepLearning/hgao/SNPDL/PrimateAI/inputdata/benign_validation_freq_0.001.csv
unknown_validation_file=sys.argv[5] #'/illumina/scratch/DeepLearning/hgao/SNPDL/PrimateAI/inputdata/unknown_validation_freq_0.001.csv'
benign_test_file=sys.argv[6]  #'/illumina/scratch/DeepLearning/hgao/SNPDL/PrimateAI/inputdata/benign_test_freq_0.001.csv'
unknown_test_file=sys.argv[7]  #'/illumina/scratch/DeepLearning/hgao/SNPDL/PrimateAI/inputdata/unknown_test_freq_0.001.csv'
weight_ss_file=sys.argv[8]   #'/illumina/scratch/DeepLearning/hgao/SNPDL/PrimateAI/inputdata/secondary_structure_seqtoseq.hdf5'
weight_sa_file=sys.argv[9]  #'/illumina/scratch/DeepLearning/hgao/SNPDL/PrimateAI/inputdata/solvent_accessibility_seqtoseq.hdf5'
out_dir=sys.argv[10]  #'/illumina/scratch/DeepLearning/hgao/SNPDL/PrimateAI/output/'


if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(out_dir+'data_sources/'):
    os.makedirs(out_dir+'data_sources/')
    os.makedirs(out_dir+'current_weights/')



###
required_data=pd.read_csv(input_fulldata_file)
required_data=required_data[required_data['mean_coverage_bins']!=0.0]
#
conservation_sequence_data=pd.DataFrame(np.load(conservation_file))
conservation_sequence_data.columns=['gene_name','sequence','primate','mammal','vertebrate']
#
benign_train_variants_name=pd.read_csv(benign_train_file,header=None)[0]
benign_validation_variants_name=pd.read_csv(benign_validation_file,header=None)[0]
benign_test_variants_name=pd.read_csv(benign_test_file,header=None)[0]
unknown_validation_variants_name=pd.read_csv(unknown_validation_file,header=None)[0]
unknown_test_variants_name=pd.read_csv(unknown_test_file,header=None)[0]

benign_validation_variants=required_data[required_data['id'].isin(benign_validation_variants_name)]
benign_test_variants=required_data[required_data['id'].isin(benign_test_variants_name)]
unknown_validation_variants=required_data[required_data['id'].isin(unknown_validation_variants_name)]
unknown_test_variants=required_data[required_data['id'].isin(unknown_test_variants_name)]
unknown_used_variants=unknown_validation_variants_name.append(unknown_test_variants_name)
#
benign_train_dataset=required_data[required_data['id'].isin(benign_train_variants_name)]
#benign_train_dataset=required_data[(required_data['label']=='Benign') & (~required_data['id'].isin(benign_test_variants_name)) & (~required_data['id'].isin(benign_validation_variants_name))]
benign_train_counts_human,benign_train_counts_otherspecies=get_benign_counts(benign_train_dataset)
#
unknown_validation_data=data_preprocessing(unknown_validation_variants,conservation_sequence_data,'unknown_validation')
conservation_valid_primates=unknown_validation_data[2]
conservation_valid_mammals=unknown_validation_data[3]
conservation_valid_otherspecies=unknown_validation_data[4] 
X_valid_orig_1=unknown_validation_data[0] 
X_valid_snp_1=unknown_validation_data[1]
y_valid=unknown_validation_data[5]
print ' validation shapes orig : ' + str(X_valid_orig_1.shape) + ' snp ' + str(X_valid_snp_1.shape) + ' labels ' + str(y_valid.shape) + ' conservation primates' + str(conservation_valid_primates.shape) + ' conservation mammals' + str(conservation_valid_mammals.shape) + ' conservation otherspecies' + str(conservation_valid_otherspecies.shape)

#
benign_validation_data=data_preprocessing(benign_validation_variants,conservation_sequence_data,'benign_validation')
conservation_valid_primates_benign=benign_validation_data[2]
conservation_valid_mammals_benign=benign_validation_data[3]
conservation_valid_otherspecies_benign=benign_validation_data[4] 
X_valid_orig_1_benign=benign_validation_data[0] 
X_valid_snp_1_benign=benign_validation_data[1]
y_valid_benign=benign_validation_data[5]
print ' validation shapes orig : ' + str(X_valid_orig_1_benign.shape) + ' snp ' + str(X_valid_snp_1_benign.shape) + ' labels ' + str(y_valid_benign.shape) + ' conservation primates' + str(conservation_valid_primates_benign.shape) + ' conservation mammals' + str(conservation_valid_mammals_benign.shape) + ' conservation otherspecies' + str(conservation_valid_otherspecies_benign.shape)


if __name__ == '__main__':
    print "Starting the iterations."
    start_time = time.time()
    p = Process(target=iteration_processing, args=(benign_train_dataset,required_data,seed_val+np.random.randint(1,105),0,'0','process_name1',weight_ss_file, weight_sa_file, out_dir))
    q = Process(target=iteration_processing, args=(benign_train_dataset,required_data,seed_val+np.random.randint(1,141),1,'1','process_name2',weight_ss_file, weight_sa_file, out_dir))
    p1 = Process(target=iteration_processing, args=(benign_train_dataset,required_data,seed_val+np.random.randint(1,876),2,'2','process_name3',weight_ss_file, weight_sa_file, out_dir))
    q1 = Process(target=iteration_processing, args=(benign_train_dataset,required_data,seed_val+np.random.randint(1,532),3,'3','process_name4',weight_ss_file, weight_sa_file, out_dir))
    p2 = Process(target=iteration_processing, args=(benign_train_dataset,required_data,seed_val+np.random.randint(1,241),4,'4','process_name5',weight_ss_file, weight_sa_file, out_dir))
    q2 = Process(target=iteration_processing, args=(benign_train_dataset,required_data,seed_val+np.random.randint(1,124),5,'5','process_name6',weight_ss_file, weight_sa_file, out_dir))
    p3 = Process(target=iteration_processing, args=(benign_train_dataset,required_data,seed_val+np.random.randint(1,423),6,'6','process_name7',weight_ss_file, weight_sa_file, out_dir))
    q3 = Process(target=iteration_processing, args=(benign_train_dataset,required_data,seed_val+np.random.randint(1,532),7,'7','process_name8',weight_ss_file, weight_sa_file, out_dir))
    p.start()
    q.start()
    p1.start()
    q1.start()
    q2.start()
    p3.start()
    q3.start()
    p2.start()
    p2.join()
    q2.join()
    p3.join()
    q3.join()
    p.join()
    q.join()
    p1.join()
    q1.join()
    evaluation_data(unknown_test_variants,'unknown_test.csv',out_dir)
    evaluation_data(benign_test_variants,'benign_test.csv',out_dir)

  

