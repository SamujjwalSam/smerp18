# coding=utf-8
# !/usr/bin/python
"""
INFO:
DESC:
script options
--------------
--param : parameter list

Created by Samujjwal_Ghosh on 11-Apr-17.

__author__ : Samujjwal Ghosh
__version__ = ": 1 $"
__date__ = "$"
__copyright__ = "Copyright (c) 2017 Samujjwal Ghosh"
__license__ = "Python"

Supervised approaches:
    SVM,

Features:
    # 1. Unigrams, bigrams
    # 2. count of words like (lakh,lakhs,millions,thousands)
    # 3. count of units present (litre,kg,gram)
    # 4. k similar tweets class votes
    # 5. k closest same class distance avg
    # 6. count of frequent words of that class (unique to that class)
    # 7. Length related features.
"""
import os,sys,json,math
import numpy as np
from collections import OrderedDict
import platform
if platform.system() == 'Windows':
    sys.path.append('D:\GDrive\Dropbox\IITH\\0 Research')
else:
    sys.path.append('/home/cs16resch01001/codes')
# print(platform.system(),"os detected.")
import my_modules as mm
date_time_tag = mm.get_date_time_tag(current_file_name=os.path.basename(__file__))

np.set_printoptions(threshold=np.inf,precision=4,suppress=True)

# change here START------------------------------
n_classes       = 7     # number of classes
result_file     = "fire16_"
# change here END--------------------------------

dataset_file    = result_file+'labeled_' # Dataset file name
grid_flag       = False # Sets the flag to use grid search
n_grams         = 2 # bigrams
min_df          = 1 # min count for each term to be considered
class_names     = ['RESOURCES AVAILABLE',
                   'RESOURCES REQUIRED',
                  ]

dataset_path = mm.get_dataset_path()


def main(result_all):
    print(dataset_file)
    train,validation,test = mm.read_labeled(dataset_file)
    train = mm.merge_dicts(train,validation)
    print("Training data:",mm.count_class([val["classes"] for id,val in train.items()],n_classes))
    print("Testing data:",mm.count_class([val["classes"] for id,val in test.items()],n_classes))

    vec,train_tfidf_matrix_1 = mm.vectorizer([vals["parsed_tweet"] for twt_id,vals in train.items()],n_grams,min_df)
    test_tfidf_matrix_1 = vec.transform([vals["parsed_tweet"] for twt_id,vals in test.items()])

    # test_names = ["alphabeta","alpha","mul","add","iw"]
    test_names = ["mul","add","iw"]
    alphas = [0.0001,0.001,0.01,0.1,0.3,0.5,0.7,0.9,1]
    betas  = [0.0001,0.001,0.01,0.1,0.3,0.5,0.7,0.9,1]
    ks     = [0.0001,0.001,0.01,0.1,1,2,5,10]
    for test_name in test_names:
        matrices = OrderedDict()
        if test_name == "alphabeta":
            for beta in betas:
                for alpha in alphas:
                    run_name = result_file+" "+test_name+" "+str(alpha)+" "+str(beta)
                    print("-------------------------------------------------------------------")
                    print("TEST:",run_name)
                    print("-------------------------------------------------------------------")
                    matrices = mm.class_tfidf_CNE(train,vec,train_tfidf_matrix_1,n_classes,alpha,beta,test_name)
                    assert(train_tfidf_matrix_1.shape == matrices[0].shape)
                    for cls in range(n_classes):
                        result_all[cls],predictions,probabilities = mm.supervised_bin(train,test,matrices[cls],test_tfidf_matrix_1.todense(),2,class_id=cls,metric=True,grid=grid_flag)
                    print(run_name,json.dumps(result_all,indent=4))
                    mm.save_json(result_all,run_name,tag=False)
        if test_name == "alpha":
            for alpha in alphas:
                run_name = result_file+" "+test_name+" "+str(alpha)
                print("-------------------------------------------------------------------")
                print("TEST:",run_name)
                print("-------------------------------------------------------------------")
                matrices = mm.class_tfidf_CNE(train,vec,train_tfidf_matrix_1,n_classes,alpha,test_name)
                assert(train_tfidf_matrix_1.shape == matrices[0].shape)
                for cls in range(n_classes):
                    result_all,predictions,probabilities = mm.supervised_bin(train,test,matrices[cls],test_tfidf_matrix_1.todense(),2,class_id=cls,metric=True,grid=grid_flag)
                    print(run_name,json.dumps(result_all,indent=4))
                    mm.save_json(result_all,run_name,tag=False)
        else:
            for k in ks:
                run_name = result_file+" "+test_name+" "+str(k)
                print("-------------------------------------------------------------------")
                print("TEST:",run_name)
                print("-------------------------------------------------------------------")
                matrices = mm.class_tfidf_CNE(train,vec,train_tfidf_matrix_1,n_classes,k,test_name)
                assert(train_tfidf_matrix_1.shape == matrices[0].shape)
                for cls in range(n_classes):
                    result_all,predictions,probabilities = mm.supervised_bin(train,test,matrices[cls],test_tfidf_matrix_1.todense(),2,class_id=cls,metric=True,grid=grid_flag)
                    print(run_name,json.dumps(result_all,indent=4))
                    mm.save_json(result_all,run_name,tag=False)
    return


if __name__ == "__main__":
    result = OrderedDict()
    main(result)
    print("MAIN: ",json.dumps(result,indent=4))