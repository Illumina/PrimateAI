import pandas as pd
import numpy as np
import sys


dir1=sys.argv[1]

unknown_test=pd.read_csv(dir1+'unknown_test.csv')
benign_test=pd.read_csv(dir1+'benign_test.csv')
#
TP= benign_test['mean'][benign_test['mean']<unknown_test['mean'].median()].shape[0]
TPR=TP*1.0 / benign_test.shape[0]
print "Accuracy of test dataset is "+ str(TPR)



