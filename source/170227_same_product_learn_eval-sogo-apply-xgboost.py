
# coding: utf-8

# In[2]:

from datalabs.ryan.workflow.search import price_comparison_group_engine as pcge
from sklearn.externals import joblib
reload(pcge)


# In[3]:

from collections import Counter


# In[4]:

# ================================================================================================================


# In[5]:

# 네이버 데이터 학습 
# 네이버 데이터 평가


# In[6]:

nmslib_top_k=30

index_train_begin = 200
index_train_end = 700
# index_train_end = 100000
index_test_begin = 0
index_test_end = 100

FRAMEWORK_NAME = '170215'


# In[ ]:




# In[7]:

print "캐시 파일 로드"
cache = pcge.load_cache_from_file(FRAMEWORK_NAME)


# In[ ]:




# In[8]:

reload(pcge)
print "엔진을 초기화 합니다"
engine = pcge.NaverPCGEngine(cache)

print "학습데이터 생성"
# generate train data 
id_list, x_list, y_list, text_x_list, vectorizer1, vectorizer2, vec_text_x_list =     engine.get_train_data(index_train_begin, index_train_end, nmslib_top_k=nmslib_top_k)



# In[9]:

print len(x_list[0])
print len(text_x_list[0])
print vec_text_x_list.get_shape()


# In[ ]:




# In[10]:

vec_text_x_list.get_shape()


# # 여기서부터 파라메터 튜닝 XGBOOST

# In[177]:

# 170223 sogo: xgboost
import xgboost as xgb
# https://github.com/dmlc/xgboost/blob/master/doc/parameter.md

xgb_params = {
    'data': 'sparse_matrix',
    'booster': 'gbtree', # gblinear > lambda/alpha/lambda_bias
    'silent': 0,
    'objective': 'binary:logistic',
}

# xgb_params = {
#     'data': 'sparse_matrix',
#     'booster': 'gbtree', # gblinear > lambda/alpha/lambda_bias
#     'silent': 1,
#     'eta': 0.005,
#     'max_depth': 8,
#     'min_child_weight': 0, # minimum sum of instance weight (hessian) needed in a child.
#     'lambda': 0.04, # L2 regularization default is 0. increase this value will make model more conservative.
#     'alpha': 0, # L1 regularization default is 0.  increase this value will make model more conservative.
#     'objective': 'binary:logistic',
#     'eval_metric': 'error'
# }

# boosting rounds
num_round = 100

# xgb는 0 or 1로 bin classification을 수행합니다
for i in range(len(y_list)):
    if y_list[i] == -1:
        y_list[i] = 0

dtrain = xgb.DMatrix(vec_text_x_list, y_list)

xgboost_model = xgb.train(xgb_params, dtrain, num_round)

print 'Learning Done.'


# In[ ]:




# In[ ]:




# In[178]:

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
import numpy as np
from IPython.core.display import display,Image
import math


# In[179]:

param = {'C':np.logspace(-4,1,20)}


# In[180]:

# gs_svc = GridSearchCV(LogisticRegression(),param, cv=4,n_jobs=4)
# gs_svc.fit(vec_text_x_list,y_list)
# print gs_svc.best_score_


# In[ ]:




# In[181]:

# print gs_svc.best_params_
# print math.log10(gs_svc.best_params_['C'])


# In[182]:

#### test
id_test_list, x_test_list, y_test_list, text_x_test_list, vectorizer1_test, vectorizer2_test, vec_text_x_test_list = engine.get_train_data(index_test_begin, index_test_end, vectorizer1, vectorizer2, nmslib_top_k=nmslib_top_k)


# In[183]:

# train
print vec_text_x_list.get_shape()
# test
print vec_text_x_test_list.get_shape()


# In[184]:

import sys
# sys.stdout.write("\r" + msg)


# In[185]:

import pandas as pd
total_ct = 0
system_ct = 0
gold_ct = 0
correct_ct = 0
true_negative_ct = 0
index_for_estim = 0

print len(y_test_list)

pred_prob = []

temp_df = pd.DataFrame()
for vec, truth in zip(vec_text_x_test_list, y_test_list):

    # [Linear Regression]
#     pred = gs_svc.predict(vec)[0]
    # [XGBoost Classifier]
    # xgboost: sparse to dense
    vec = vec.toarray()[0]
    for i in range(len(vec)):
        temp_df['f'+str(i)] = [vec[i]]
    
    X_test = xgb.DMatrix(temp_df)
    pred = xgboost_model.predict(X_test)[0]
    pred_prob.append(pred)
    
    msg = "%.3f%% percent processing.. %f" % (round(float(index_for_estim*100)/1012,3), pred)
    sys.stdout.write("\r" + msg)
    index_for_estim += 1
    
    # binary classification x > 0.5 == 1 && x <= 0.5 == 0
    if pred > 0.5:
        pred = 1
    else:
        pred = 0

    total_ct += 1
    if truth == 1:
        gold_ct+=1
#             print convert_eng_digit(inter_str)
        if pred==1:
            correct_ct+=1
    if pred>0:
        system_ct+=1

    if truth == -1 and pred == 0:
        true_negative_ct += 1

print '\nDone.'


# In[186]:

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


pred_prob.sort()
prob_mean = np.mean(pred_prob)
prob_std = np.std(pred_prob)
pdf = stats.norm.pdf(pred_prob, prob_mean, prob_std)
# plt.plot(pred_prob, pdf, '-')
# plt.show()
print prob_mean, '\t', prob_std

plt.plot(pred_prob, pdf, '-', color='red')
plt.hist(pred_prob, 
         normed=True, 
         color='green', 
         edgecolor='black', 
         linewidth='2',
         alpha=0.8)
plt.show()


# In[187]:

from pylab import rcParams
rcParams['figure.figsize'] = 15, 15

# plt.show(xgb.plot_importance(xgboost_model))


# In[ ]:




# In[188]:

import cStringIO
import collections
# def show_model_eval_result(correct_ct, system_ct, gold_ct, true_negative_ct, total_ct, y_list=None, param=None,
#                        gs_svc=None):
"""
@param int correct_ct: # true positive
@param int system_ct: # predicted condition positive
@param int gold_ct: # condition positive
@param int true_negative_ct: # true negative
@param list y_list: (optional) list of target variable of train
@return None
"""
output = cStringIO.StringIO()

if system_ct != 0:
    precision = correct_ct / float(system_ct)
else:
    precision = None

if gold_ct != 0:
    recall = correct_ct / float(gold_ct)
else:
    recall = None

if recall is not None         and precision is not None         and precision + recall != 0:
    f1 = (2 * precision * recall) / (precision + recall)
else:
    f1 = None

accuracy = float(correct_ct + true_negative_ct) / total_ct

if y_list is not None:
    print >> output, 'Train data distribution'
    for each in collections.Counter(y_list).most_common(20):
        print >> output, '\t', each

if param is not None:
    print >> output, 'parameter candidates', param

# if gs_svc is not None:
#     print >> output, 'selected parameter', gs_svc.best_params_
#     print >> output, '[Training score   ]', gs_svc.best_score_


print 'correct_ct %d, true_negative_ct %d, total_ct %d' % (correct_ct, true_negative_ct, total_ct)
print >> output, '[Test Accuracy    ]', accuracy
print >> output, '[Test Precision   ]', precision, "(", correct_ct, "/", system_ct, ")"
print >> output, '[Test Recall      ]', recall, "(", correct_ct, "/", gold_ct, ")"
print >> output, '[Test F1 Score    ]', f1

print output.getvalue()


# In[189]:

print accuracy,'\t', precision, '\t', recall,'\t', f1


# In[139]:

reload(pcge)

# eval_result = pcge.show_model_eval_result(correct_ct, system_ct, gold_ct, true_negative_ct, total_ct, y_list, param, gs_svc)

# print eval_result


# # Booster: dart & params
# 
# xgb_params = {
#     'data': 'sparse_matrix',
#     'booster': 'dart',
#     'silent': 1,
#     'eta': 0.01,
#     'max_depth': 10,
#     'objective': 'reg:logistic',
# }
# 
# [Test Accuracy    ] 0.46837944664
# [Test Precision   ] 0.580882352941 ( 474 / 816 )
# [Test Recall      ] 0.97131147541 ( 474 / 488 )
# [Test F1 Score    ] 0.726993865031
# 
# # gbtree: logistic + params
# 
# xgb_params = {
#     'data': 'sparse_matrix',
#     'booster': 'gbtree',
#     'silent': 0,
#     'eta': 0.01,
#     'max_depth': 50,
#     'lambda': 0.001,
#     'alpha': 0.0001,
#     'objective': 'binary:logistic',
#     'eval_metric': 'error'
# }
# 
# [Test Accuracy    ] 0.466403162055
# [Test Precision   ] 0.60824742268 ( 472 / 776 )
# [Test Recall      ] 0.967213114754 ( 472 / 488 )
# [Test F1 Score    ] 0.746835443038
# 
# # Default
# 
# xgb_params = {
#     'data': 'sparse_matrix',
#     'booster': 'gbtree', # gblinear > lambda/alpha/lambda_bias
#     'silent': 0,
#     'objective': 'binary:logistic',
# }
# 
# [Test Accuracy    ] 0.120553359684
# [Test Precision   ] 0.797385620915 ( 122 / 153 )
# [Test Recall      ] 0.25 ( 122 / 488 )
# [Test F1 Score    ] 0.380655226209
# 
# # Default + param tune
# 
# xgb_params = {
#     'data': 'sparse_matrix',
#     'booster': 'gbtree', # gblinear > lambda/alpha/lambda_bias
#     'silent': 0,
#     'eta': 0.01,
#     'max_depth': 100,
#     'lambda': 0.1, # L2 regularization default is 1. increase this value will make model more conservative.
#     'alpha': 0.0001, # L1 regularization default is 0.  increase this value will make model more conservative.
#     'objective': 'binary:logistic',
#     'eval_metric': 'error'
# }
# 
# [Test Accuracy    ] 0.474308300395
# [Test Precision   ] 0.567375886525 ( 480 / 846 )
# [Test Recall      ] 0.983606557377 ( 480 / 488 )
# [Test F1 Score    ] 0.71964017991
# 
# # Default + L1/L2 = 0
# 
# xgb_params = {
#     'data': 'sparse_matrix',
#     'booster': 'gbtree', # gblinear > lambda/alpha/lambda_bias
#     'silent': 0,
#     'eta': 0.01,
#     'max_depth': 100,
#     'lambda': 0, # L2 regularization default is 1. increase this value will make model more conservative.
#     'alpha': 0, # L1 regularization default is 0.  increase this value will make model more conservative.
#     'objective': 'reg:logistic',
#     'eval_metric': 'error'
# }
# 
# [Test Accuracy    ] 0.47233201581
# [Test Precision   ] 0.575210589651 ( 478 / 831 )
# [Test Recall      ] 0.979508196721 ( 478 / 488 )
# [Test F1 Score    ] 0.724791508719
# 
# # n_rounds=150 / tree depth=2000
# 
# [Test Accuracy    ] 0.416007905138
# [Test Precision   ] 0.820662768031 ( 421 / 513 )
# [Test Recall      ] 0.862704918033 ( 421 / 488 )
# [Test F1 Score    ] 0.841158841159
# 
# # n_rounds=2500 / tree depth = 15
# 
# overfitting?
# 
# xgb_params = {
#     'data': 'sparse_matrix',
#     'booster': 'gblinear', # gblinear > lambda/alpha/lambda_bias
#     'silent': 0,
#     'eta': 0.0001,
#     'max_depth': 15,
#     'min_child_weight': 0, # minimum sum of instance weight (hessian) needed in a child.
#     'lambda': 0,
#     'alpha': 0,
#     'objective': 'binary:logistic',
#     'eval_metric': 'error'
# }
# 
# [Test Accuracy    ] 0.433794466403
# [Test Precision   ] 0.817504655493 ( 439 / 537 )
# [Test Recall      ] 0.899590163934 ( 439 / 488 )
# [Test F1 Score    ] 0.856585365854
# 
# # n_rounds=2500 / tree depth = 6 / lr=0.03
# 
# [Test Accuracy    ] 0.424901185771
# [Test Precision   ] 0.784671532847 ( 430 / 548 )
# [Test Recall      ] 0.881147540984 ( 430 / 488 )
# [Test F1 Score    ] 0.830115830116
# 
# # n_rounds=500 / tree depth = 6 / lr =0.03
# 
# overfit?
# 
# xgb_params = {
#     'data': 'sparse_matrix',
#     'booster': 'gblinear', # gblinear > lambda/alpha/lambda_bias
#     'silent': 0,
#     'eta': 0.03,
#     'max_depth': 6,
#     'min_child_weight': 0, # minimum sum of instance weight (hessian) needed in a child.
#     'lambda': 0,
#     'alpha': 0,
#     'objective': 'binary:logistic',
#     'eval_metric': 'error'
# }
# num_round = 500
# 
# [Test Accuracy    ] 0.431818181818
# [Test Precision   ] 0.804788213628 ( 437 / 543 )
# [Test Recall      ] 0.895491803279 ( 437 / 488 )
# [Test F1 Score    ] 0.847720659554
# 
# 
# # lr= 0.003
# 
# [Test Accuracy    ] 0.416007905138
# [Test Precision   ] 0.838645418327 ( 421 / 502 )
# [Test Recall      ] 0.862704918033 ( 421 / 488 )
# [Test F1 Score    ] 0.850505050505
# 
# # n_ronds, lr, depth
# 
# xgb_params = {
#     'data': 'sparse_matrix',
#     'booster': 'gblinear',
#     'silent': 0,
#     'eta': 0.005,
#     'max_depth': 8,
#     'min_child_weight': 0,
#     'lambda': 0.04,
#     'alpha': 0,
#     'objective': 'binary:logistic',
#     'eval_metric': 'error'
# }
# num_round = 300
# 
# [Test Accuracy    ] 0.882411067194
# [Test Precision   ] 0.866799204771 ( 436 / 503 )
# [Test Recall      ] 0.893442622951 ( 436 / 488 )
# [Test F1 Score    ] 0.879919273461
# 
# # 1000 rounds
# 
# xgb_params = {
#     'data': 'sparse_matrix',
#     'booster': 'gblinear', # gblinear > lambda/alpha/lambda_bias
#     'silent': 0,
#     'eta': 0.005,
#     'max_depth': 8,
#     'min_child_weight': 0, # minimum sum of instance weight (hessian) needed in a child.
#     'lambda': 0.04, # L2 regularization default is 0. increase this value will make model more conservative.
#     'alpha': 0, # L1 regularization default is 0.  increase this value will make model more conservative.
#     'objective': 'binary:logistic',
#     'eval_metric': 'error'
# }
# 
# num_round = 1000
# 
# [Test Accuracy    ] 0.897233201581
# [Test Precision   ] 0.869230769231 ( 452 / 520 )
# [Test Recall      ] 0.926229508197 ( 452 / 488 )
# [Test F1 Score    ] 0.896825396825
# 
# # 2000 rounds
# 
# xgb_params = {
#     'data': 'sparse_matrix',
#     'booster': 'gblinear', # gblinear > lambda/alpha/lambda_bias
#     'silent': 0,
#     'eta': 0.005,
#     'max_depth': 8,
#     'min_child_weight': 0, # minimum sum of instance weight (hessian) needed in a child.
#     'lambda': 0.04, # L2 regularization default is 0. increase this value will make model more conservative.
#     'alpha': 0, # L1 regularization default is 0.  increase this value will make model more conservative.
#     'objective': 'binary:logistic',
#     'eval_metric': 'error'
# }
# 
# num_round = 2000
# 
# [Test Accuracy    ] 0.900197628458
# [Test Precision   ] 0.872832369942 ( 453 / 519 )
# [Test Recall      ] 0.928278688525 ( 453 / 488 )
# [Test F1 Score    ] 0.899702085402
# 
# 
# 

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



