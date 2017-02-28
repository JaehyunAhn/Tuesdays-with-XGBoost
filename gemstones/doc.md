# XGBoost Parameters
* referred from : https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
* official website : http://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters
* XGBoost 저자 직강 : http://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf
* XGBoost란? : http://xgboost.readthedocs.io/en/latest/model.html
* XGBoost의 선조격인 GBM(Gradient Boosting Machine)은 뭣? : https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
* 파라메터 정리: https://github.com/dmlc/xgboost/blob/master/doc/parameter.md << **여기가 좋다**
* XGBoost Official APIs: http://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.plotting

---

* 용어와 핵심 개념을 정리하자
    * GBM (Gradient Boosting Machine) : Boosted Tree [여기 링크를 보며 감을 잡자](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)
    * Gradient Boosting : 약한 앙상블 예측 문제(ex) decision tree)를 만들고, stage wise하게 boosting을 적용한다.
    * XGBoost는 supervised learning 문제를 해결하는데 사용한다. XGBoost는 Tree를 사용한 앙상블 부스팅 방법이다. CART라 불리는 트리 앙상블 모델을 구축하고 사용한다. (그림도 보고 싶다면 이곳으로 : http://xgboost.readthedocs.io/en/latest/model.html)
    * CART : Classification and Regression Trees
    * 다양한 기준의 Tree를 구축하고, leaf 데이터의 점수를 통해 모델의 적합성을 체크한다. 이후 Tree Boosting을 이용하여 Tree를 이 문제에 맞도록 최적화한다.

* 이미지도 있다 
    ~~사실 마크다운 연습하고 싶어서 해봤다~~

    ![alt text](https://raw.githubusercontent.com/dmlc/web-data/master/xgboost/model/cart.png)
    ![alt text](https://raw.githubusercontent.com/dmlc/web-data/master/xgboost/model/twocart.png)

---
## General Parameters
    Guide the overall functioning
###  booster [default=gbtree]
#### gbtree
tree-based models
#### gblinear
linear models
### silent
1, no running messages will be printed
### nthread
number of cores, automatically sets

---
## Booster Parameters
    Guide the individual booster at each step
### eta [default=0.3]
learning rate in GBM
* Typical final values to be used 0.01 - 0.2
### min_child_weight [default = 1]
minimum sum of weights of all observations required in a child
This used to control over-fitting. Higher values prevent a model from learning relations which might be highlt specifice to the particular sample selected for a tree.
Too high values can lead to uner-fitting.
### max_depth [default = 6]
maximum depth of a tree. Used to control over-fitting as higher depth,
* Typical values: 3 - 10
### max_leaf_nodes
maximum number of terminal nodes or leaves in a tree.
* 2^n leaves.
### gamma [ default = 0]
Gamma specifies the minimum loss reduction required to make a split.
Makes the algorithm conservative. This values can very depending on the loss function and should be tuned.
### max_delta_step [default = 0 ]
If the value set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative.
* Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced.
* **This generally not used but you can explore further if you wish**
### subsample [default = 0]
Fractions of observations to be randomly samples for each tree.
* Typical values: 0.5 -1
### lambda [default = 1]
* L2 regularization term on wieghts (analogus to Ridge Regression)
**This used to handle the regularization part of XGBoost. Through many data scientists do not use it often, it should be explored to reduce overfitting.**
### alpha [default = 0]
* L1 regularization term on weight (analogus to Lasso Regression)
Can be used in case of very high dimensionality so that the algorithm runs faster when implemented
### scale_pos_weight [default = 1]
A value grater than 0 should be used in case of high class imbalance as it helps in faster convergence.

---
## Learning Task Parameters
    Guide the optimization method.
### objective [default=reg:linear]
Loss function to be minimized. Moustly used values are:
    * binary:logistic - logistic regression for binary classificiation, **returns predicted probability** (not class)

    * multi:softmax - multiclass classification using softmax objective, **returns predicted class** (not prbabilites)

        * You also need to set an additional **num_class** parameter defining the number of unique classes

    * multi:softprob - same as softmax, but returns predicted probability of each data point belonging to each class.

### eval_metric [default according to objective]
The matric to be used for validation data.
The default values are **rmse** for regression and error for classification.
Typical values are:
    * rmse - root mean squre error
    * mae - mean absolute error
    * logloss - negative log-likelihood
    * error - Binary classification error rate (0.5 threshold)
    * merror - Multiclass classification error rate
    * mloglos - Multiclass logloss
    * Auc - Area under the curve
### seed [ default = 0]
Random number sedd.
Can be used for generating reproducible results and also for parameter tuning.

# control Overfitting

The first way is to directly control model complexity
* This include max_depth, min_child_weight and gamma

The second way is to add randomness to make training robust to noise
* This include subsample, colsample_bytree
* You can also reduce stepsize eta, but needs to remember to increase num_round when you do so.

# num rounds

* number of rounds to fit. 너무 높으면 오버피팅.
* 이곳에서 rounds로 검색하자: http://xgboost.readthedocs.io/en/latest/R-package/discoverYourData.html
* **라운드란?**: There will be two passes on the data, the second one will enhance the model by further reducing the difference between ground truth and prediction.
