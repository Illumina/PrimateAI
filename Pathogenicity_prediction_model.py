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
#
import numpy as np

def get_sa_model():
  model_type='sa'
  import keras
  from keras.models import Model
  from keras.layers.convolutional import Conv1D
  from keras.layers import Input, Dropout
  from keras.layers import Lambda, Activation, merge
  from keras.layers.normalization import BatchNormalization
  import keras.backend as K
  L=40 #number of filters
  N = np.asarray([2,2,2]) #Depth of the model
  W = np.asarray([5,5,5]) #filter length 
  AR = np.asarray([1,1,1])
  def residual_unit(nb_fil,f_len,ar,indx,residual='yes'):
      def f(input_node):
          bn1 = BatchNormalization(name=model_type+'_BatchNormalization_'+indx+'_1')(input_node)
          act1 = Activation('relu',name=model_type+'_relu_'+indx+'_1')(bn1)
          conv1= Conv1D(nb_fil, f_len,dilation_rate=ar,padding='same',name=model_type+'_conv1d_'+indx+'_1')(act1)
          bn2 = BatchNormalization(name=model_type+'_BatchNormalization_'+indx+'_2')(conv1)
          act2 = Activation('relu',name=model_type+'_relu_'+indx+'_2')(bn2)
          conv2= Conv1D(nb_fil, f_len,dilation_rate=ar,padding='same',name=model_type+'_conv1d_'+indx+'_2')(act2)
          if residual=='yes':
            output_node = keras.layers.add([conv2, input_node])
          else:
            output_node = conv2
          return output_node
      return f  
  input0 = Input(shape=(None, 20),name=model_type+'_sequence')
  conv = Conv1D(L, 1,name=model_type+'_1d_conv_sequence')(input0)
  skip = Conv1D(L, 1,name=model_type+'_skip_1d_conv_sequence')(input0)
  conv = Conv1D(L, 1,name=model_type+'_1d_conv_down')(conv)
  skip = Conv1D(L, 1,name=model_type+'_1d_skip_down')(skip)
  for i in range(len(N)):
      for j in range(N[i]):
          conv = residual_unit(L, W[i], AR[i],model_type+'_residual_'+str(i)+'_'+str(j),residual='yes')(conv)
      dense = Conv1D(L, 1,name=model_type+'_denseforskip'+str(i))(conv)    
      skip = keras.layers.add([skip, dense],name=model_type+'_skip'+str(i)) 
  conv = residual_unit(L, 1, 1 ,model_type+'_residual_final')(skip)
  model=Model(input=[input0],output=[conv])
  model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
  model.load_weights('trained_sa_model_wrights.hdf5',by_name=True)
  return model




def get_ss_model():
  model_type='ss'
  import keras
  from keras.models import Model
  from keras.layers.convolutional import Conv1D
  from keras.layers import Input, Dropout
  from keras.layers import Lambda, Activation, merge
  from keras.layers.normalization import BatchNormalization
  import keras.backend as K
  L=40 #number of filters
  N = np.asarray([2,2,2]) #Depth of the model
  W = np.asarray([5,5,5]) #filter length 
  AR = np.asarray([1,1,1])
  def residual_unit(nb_fil,f_len,ar,indx,residual='yes'):
      def f(input_node):
          bn1 = BatchNormalization(name=model_type+'_BatchNormalization_'+indx+'_1')(input_node)
          act1 = Activation('relu',name=model_type+'_relu_'+indx+'_1')(bn1)
          conv1= Conv1D(nb_fil, f_len,dilation_rate=ar,padding='same',name=model_type+'_conv1d_'+indx+'_1')(act1)
          bn2 = BatchNormalization(name=model_type+'_BatchNormalization_'+indx+'_2')(conv1)
          act2 = Activation('relu',name=model_type+'_relu_'+indx+'_2')(bn2)
          conv2= Conv1D(nb_fil, f_len,dilation_rate=ar,padding='same',name=model_type+'_conv1d_'+indx+'_2')(act2)
          if residual=='yes':
            output_node = keras.layers.add([conv2, input_node])
          else:
            output_node = conv2
          return output_node
      return f  
  input0 = Input(shape=(None, 20),name=model_type+'_sequence')
  conv = Conv1D(L, 1,name=model_type+'_1d_conv_sequence')(input0)
  skip = Conv1D(L, 1,name=model_type+'_skip_1d_conv_sequence')(input0)
  conv = Conv1D(L, 1,name=model_type+'_1d_conv_down')(conv)
  skip = Conv1D(L, 1,name=model_type+'_1d_skip_down')(skip)
  for i in range(len(N)):
      for j in range(N[i]):
          conv = residual_unit(L, W[i], AR[i],model_type+'_residual_'+str(i)+'_'+str(j),residual='yes')(conv)
      dense = Conv1D(L, 1,name=model_type+'_denseforskip'+str(i))(conv)    
      skip = keras.layers.add([skip, dense],name=model_type+'_skip'+str(i)) 
  conv = residual_unit(L, 1, 1 ,model_type+'_residual_final')(skip)
  model=Model(input=[input0],output=[conv])
  model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
  model.load_weights('trained_ss_model_wrights.hdf5',by_name=True)
  return model






def get_model():
  import keras
  from keras.layers.pooling import GlobalMaxPooling1D
  from keras.models import Sequential,Model
  from keras.layers import Dense,Activation,Dropout,Flatten
  from keras.layers.convolutional import Conv1D
  from keras.layers import Input, Embedding, LSTM, Dense
  from keras.layers.normalization import BatchNormalization
  from keras.regularizers import l1,l2
  from keras.optimizers import SGD,RMSprop,Adam
  def residual_unit(nb_fil,f_len,ar,indx,residual='yes'):
    def f(input_node):
        bn1 = BatchNormalization(name='BatchNormalization_'+indx+'_1')(input_node)
        act1 = Activation('relu',name='relu_'+indx+'_1')(bn1)
        conv1= Conv1D(nb_fil, f_len,dilation_rate=ar,padding='same',name='conv1d_'+indx+'_1')(act1)
        bn2 = BatchNormalization(name='BatchNormalization_'+indx+'_2')(conv1)
        act2 = Activation('relu',name='relu_'+indx+'_2')(bn2)
        conv2= Conv1D(nb_fil, f_len,dilation_rate=ar,padding='same',name='conv1d_'+indx+'_2')(act2)
        if residual=='yes':
          output_node = keras.layers.add([conv2, input_node])
        else:
          output_node = conv2
        return output_node
    return f   
  seq_length=51
  L=40 #number of filters
  N=np.asarray([2,2,2])  #Depth of the model
  W=np.asarray([5,5,5])  #filter length 
  AR=np.asarray([1,1,1])   #atrous rate
  input0 = Input(shape=(seq_length,20 ), name='orig_seq')
  input1 = Input(shape=(seq_length,20),name='snp_seq')
  input2 = Input (shape=(seq_length,20),name='conservation_full')
  input3 = Input (shape=(seq_length,20),name='conservation_primates')
  input4 = Input (shape=(seq_length,20),name='conservation_otherspecies')
  input5 = keras.layers.add([input2,input3,input4],name='cons_counts')
  ss_model=get_ss_model()
  struc=ss_model([input5])
  sa_model=get_sa_model()
  solv=sa_model([input5])
  conv_orig = Conv1D(L, 1, name='1d_conv_orig')(input0)
  conv_snp = Conv1D(L, 1, name='1d_conv_snp')(input1)
  conv_consrv_mammals=Conv1D(L, 1, name='1d_conv_mammals')(input2)
  conv_consrv_primate=Conv1D(L, 1, name='1d_conv_primates')(input3)
  conv_consrv_vertebrates=Conv1D(L, 1, name='1d_conv_vertebrates')(input4)
  conv_orig=keras.layers.add([conv_orig,conv_consrv_mammals,conv_consrv_primate,conv_consrv_vertebrates,struc,solv],name='merge_orig_conserv')
  conv_snp=keras.layers.add([conv_snp,conv_consrv_mammals,conv_consrv_primate,conv_consrv_vertebrates,struc,solv],name='merge_snp_conserv')
  conv_orig1=residual_unit(L,W[0],1,'orig_residual',residual='no')(conv_orig)
  conv_snp1=residual_unit(L,W[0],1,'snp_residual',residual='no')(conv_snp)
  conv=keras.layers.concatenate([conv_orig1,conv_snp1],name='snp_orig_conv_merge')
  skip=keras.layers.concatenate([conv_orig1,conv_snp1],name='skip_snp_orig_conv_merge')
  conv=Conv1D(L, W[0], name='1d_conv_reduce',padding='same')(conv) # see if 1 better or W
  skip=Conv1D(L, W[0], name='1d_skip_reduce',padding='same')(skip) # see if 1 better or W
  for i in range(len(N)):
    for j in range(N[i]):
      conv = residual_unit(L, W[i], AR[i],'residual_'+str(i)+'_'+str(j),residual='yes')(conv)
    dense = Conv1D(L, 1,name='denseforskip'+str(i))(conv)    
    skip = keras.layers.add([skip, dense],name='skip'+str(i))
  conv = residual_unit(L, 1, 1,'residual_final',residual='yes')(skip)
  conv= Conv1D(1, 1,padding='same', activation='sigmoid',name='conv_final')(conv)
  output0=GlobalMaxPooling1D(name='output')(conv)
  model=Model(input=[input0,input1,input2,input3,input4],output=[output0])
  model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
  return model


