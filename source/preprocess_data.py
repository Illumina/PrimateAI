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
import time
import os 
import glob
import math



def data_preprocessing(dataframe,conservation_data,process_name):
  start_time=time.time()
  print 'start time'+ str(start_time)
  seq_length=51
  flank=(seq_length-1)/2
  #
  conservation_data[['primatet','mammalt','vertebratet']]=conservation_data[['primate','mammal','vertebrate']].applymap(lambda x :\
    np.concatenate((np.zeros((flank,20)),x,np.zeros((flank,20))),axis=0).flatten())
  conservation_data['sequencet']=conservation_data.apply(lambda x : 'Z'*flank+x['sequence']+'Z'*flank,axis=1)
  #
  dataframe=dataframe.reset_index()
  del dataframe['index']
  dataframe['index_sort']=dataframe.index
  dataframe=dataframe.merge(conservation_data,on='gene_name').sort_values('index_sort')
  #sequence
  dataframe['extractseq']=dataframe.apply(lambda x : x['sequencet']\
    [x['change_position_1based']-flank+(flank-1):x['change_position_1based']+(flank-1)+(flank+1)],axis=1)
  dataframe['orig_sequence']=dataframe.apply(lambda x : x['extractseq'][:flank] +x['ref_aa']+x['extractseq'][(flank+1):],axis=1)
  dataframe['snp_sequence']=dataframe.apply(lambda x : x['extractseq'][:flank] +x['alt_aa']+x['extractseq'][(flank+1):],axis=1)
  dataframe[['ref_seq','alt_seq']]=dataframe[['orig_sequence','snp_sequence']].applymap(lambda x : x.replace('-','Z').replace('*','Z').\
  replace('Z','00000000000000000000').replace('Y','00000000000000000001').replace('W','00000000000000000010').\
  replace('V','00000000000000000100').replace('T','00000000000000001000').replace('S','00000000000000010000').\
  replace('R','00000000000000100000').replace('Q','00000000000001000000').replace('P','00000000000010000000').\
  replace('N','00000000000100000000').replace('M','00000000001000000000').replace('L','00000000010000000000').\
  replace('K','00000000100000000000').replace('I','00000001000000000000').replace('H','00000010000000000000').\
  replace('G','00000100000000000000').replace('F','00001000000000000000').replace('E','00010000000000000000').\
  replace('D','00100000000000000000').replace('C','01000000000000000000').replace('A','10000000000000000000'))
  X_test_orig_1=dataframe['ref_seq'].as_matrix().astype(str).view('S1').reshape(len(dataframe),-1,20)   #change as needed
  print X_test_orig_1.shape
  X_test_snp_1=dataframe['alt_seq'].as_matrix().astype(str).view('S1').reshape(len(dataframe),-1,20) #change as needed
  dataframe=dataframe.drop(['extractseq','orig_sequence','snp_sequence','ref_seq','alt_seq'], axis=1)
  #
  temp_index_changepositon=dataframe.columns.get_loc('change_position_1based')+1
  #
  temp_index_primatet=dataframe.columns.get_loc('primatet')+1
  X_train_conservation_onlyprimates=numpy.empty((len(dataframe),seq_length*20))
  for i, el in enumerate((x[temp_index_primatet][(x[temp_index_changepositon]+(flank-1)-flank)*20:(x[temp_index_changepositon]+(flank-1)+(flank+1))*20] for x in dataframe.itertuples())):
    X_train_conservation_onlyprimates[i] = el
  X_train_conservation_onlyprimates=X_train_conservation_onlyprimates.reshape(-1,seq_length,20)
  #
  temp_index_mammalt=dataframe.columns.get_loc('mammalt')+1
  X_train_conservation_mammals=numpy.empty((len(dataframe),seq_length*20))
  for i, el in enumerate((x[temp_index_mammalt][(x[temp_index_changepositon]+(flank-1)-flank)*20:(x[temp_index_changepositon]+(flank-1)+(flank+1))*20] for x in dataframe.itertuples())):
    X_train_conservation_mammals[i]=el
  X_train_conservation_mammals=X_train_conservation_mammals.reshape(-1,seq_length,20)
  #
  temp_index_vertebratet=dataframe.columns.get_loc('vertebratet')+1
  X_train_conservation_vertebrates=np.empty((len(dataframe),seq_length*20))
  for i, el in enumerate((x[temp_index_vertebratet][(x[temp_index_changepositon]+(flank-1)-flank)*20:(x[temp_index_changepositon]+(flank-1)+(flank+1))*20] for x in dataframe.itertuples())):
    X_train_conservation_vertebrates[i]=el
  X_train_conservation_vertebrates=X_train_conservation_vertebrates.reshape(-1,seq_length,20)
  #
  dataframe['label']=dataframe['label'].apply(lambda x : x.replace('Unknown','1').replace('Benign','0').replace('likely benign','1')).astype(int)
  y_train=dataframe['label'].as_matrix()
  print ' the time for the data preprocessing is ' + str(time.time()-start_time)
  return (X_test_orig_1,X_test_snp_1,X_train_conservation_onlyprimates,X_train_conservation_mammals,X_train_conservation_vertebrates,y_train)



def get_benign_counts(benign_dataframe):
  benign_counts_otherspecies=benign_dataframe[benign_dataframe['species']!='human'].groupby(['species','mirrored_column'])
  benign_counts_otherspecies=pd.concat([ benign_counts_otherspecies['id'].count(),benign_counts_otherspecies['mean_coverage_bins'].apply(set)],axis=1).reset_index()
  benign_counts_human=benign_dataframe[benign_dataframe['species']=='human'].groupby(['mirrored_column'])['id'].count().reset_index()
  return benign_counts_human,benign_counts_otherspecies


def get_mirrored(required_data,required_counts_human,required_counts_species,seed_val,type,mul_factor=1):
  seed_val=seed_val+np.random.randint(1,1998)
  np.random.seed(seed_val)
  start_time=time.time()
  try:
    unknown_data=required_data[(required_data['label']=='Unknown') & (~required_data['id'].isin(unknown_used_variants))]
    print 'done'
  except:
    unknown_data=required_data[(required_data['label']=='Unknown')]
  ids=[]
  bin_dict={10.0:0.008220888,20.0:0.008506095,30.0:0.008743091,40.0:0.008981522,50.0:0.009229482,60.0:0.00947869,70.0:0.009734821,80.0:0.009999581,90.0:0.01027276,100.0:0.010513217}
  for i in set(required_counts_species['species']):
    print i
    if i in unknown_data.columns:
      temp=unknown_data.loc[(~unknown_data['id'].isin(ids)) &(unknown_data[i])].groupby(['mirrored_column','mean_coverage_bins'])['id'].apply(list).to_dict()
      for species_counts in required_counts_species.iloc[np.where(required_counts_species['species'].values==i)].itertuples():
        pvalues=map(lambda x : bin_dict[x] if x in species_counts[4] else 0,[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        counts_unk=np.round((np.array(pvalues)/sum(pvalues))*species_counts[3]*mul_factor).astype(int)
        counts_unk_dict={prob:cnts for prob,cnts in zip([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],counts_unk)}
        ids=np.concatenate([ids,np.concatenate(map(lambda x : np.array(temp[(species_counts[2],x)])[np.random.permutation(len(temp[(species_counts[2],x)]))[:counts_unk_dict[x]]] ,[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0] ))])
  #
  if not required_counts_human.empty:
    human_temp=unknown_data.loc[(~unknown_data['id'].isin(ids))].groupby(['mirrored_column'])['id'].apply(list).reset_index()
    ids=np.concatenate([ids,np.concatenate(required_counts_human.merge(human_temp, on='mirrored_column').apply(lambda x: [np.array(x[2])[np.random.permutation(len(x[2]))[:int(np.round(x[1]*mul_factor))]]],axis=1).ravel(),axis=1).squeeze()])
  res=unknown_data.loc[(unknown_data['id'].isin(ids))]
  print 'the total time for mirroring is ' + str(time.time()-start_time)
  return res
