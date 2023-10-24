---
layout: post
title: ML_guide/Evaluation
subtitle: "Evaluation"
categories: ML_guide
tags: [ai]
---


## 평가 지표

> ## 오차 행렬 (confusion matrix)

<img src="https://user-images.githubusercontent.com/52434993/89038156-c4d53780-d37a-11ea-836e-44aab9748ea0.png" width = "80%">   

* 학습된 분류 모델이 예측을 수행할 때의 상태를 나타낸다.   
* **정확도 = 예측결과와 실제 값이 동일한 건수 / 전체 데이터 수 = (TN + TP) / (TN + TP + FP + FN)**
<br>
<br>

> ## 정밀도, 재현율 (precision, recall)   

* **정밀도 = TP / (FP + TP)**    
* **재현율 = TP / (FN + TP)**   
* 즉, 정밀도는 예측을 positive로 한 대상 중에 예측과 실제 값이 일치하는 데이터의 비율이다.   
* 반면 재현율은 실제 값이 positive인 대상 중에 예측과 실제 값이 positive로 일치한 데이터의 비율을 말한다.   
* **두 지표 모두 높은 수치를 얻는 것이 바람직하다.**

* 임계값에 따라 두 지표를 조정할 수 있다.   

```
def precision_recall_curve_plot(y_test, pred_proba_c1):
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)
    
    plt.figure(figsize =(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], ls = '--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary],label='recall')
    
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start,end,0.1),2))
    
    plt.xlabel('Threshold value')
    plt.ylabel('Precision and Recall value')
    plt.legend()
    plt.grid()
    plt.show()    
```   



*단 정밀도와 재현율은 예측기준을 각각의 지표에 유리하게 한다면, 두 수치 모두 적절한 지점을 찾기가 어려워 진다.*   

<br>
<br>

> ## F1 스코어   

* F1 = 2*(precision * recall) / (precision + recall)   
* F1 스코어는 정밀도와 재현율을 결합한 지표이다.    
* 두 지표가 치우치지 않을 때, 높은 값을 가지게 된다.   
<br>
<br>


> ## ROC 곡선, AUC   

* FPR(False Positive Rate)이 변할 때 TPR(True Positive Rate)의 변화를 나타내는 곡선이다.   
* 임계값을 1부터 0까지 변경하면 곡선을 나타낼 수 있다.   

```
def roc_curve_plot(y_test, pred_proba_c1):
    fprs, tprs, thresholds = roc_curve(y_test, pred_proba_c1)
    plt.plot(fprs, tprs, label='ROC')
    plt.plot([0,1],[0,1],'k--', label = 'Random')
    
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start,end,0.1),2))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('FPR(1 -Sensitivity)')
    plt.ylabel('TPR(Recall)')
    plt.legend()
```   

<img src="https://user-images.githubusercontent.com/52434993/89048204-79c32080-d38a-11ea-9a22-36c65cfea515.jpg" width = "80%">  
 

-------------------------------------   
<br>
<br>
<br>

## 실제 데이터를 이용한 평가   

> ## Pima Indian Diabetes   

* 데이터를 불러오고 feature의 타입과 null 개수를 살펴본다.   
```
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Pregnancies               768 non-null    int64  
 1   Glucose                   768 non-null    int64  
 2   BloodPressure             768 non-null    int64  
 3   SkinThickness             768 non-null    int64  
 4   Insulin                   768 non-null    int64  
 5   BMI                       768 non-null    float64
 6   DiabetesPedigreeFunction  768 non-null    float64
 7   Age                       768 non-null    int64  
 8   Outcome                   768 non-null    int64 
```   

* null 값이 없고 모두 숫자형이므로 따로 인코딩이 필요 없어 보인다.   
* 위에서 공부한 평가지표를 계산하는 함수를 이용하여 지표를 출력한다.   

```
오차 행렬
[[88 12]
 [23 31]]
정확도 0.7727, 정밀도 0.7209, 재현율 : 0.5741, F1 : 0.6392, AUC : 0.7919
```

<img src="https://user-images.githubusercontent.com/52434993/89049685-952f2b00-d38c-11ea-86e4-2967148e3c9f.jpg" width = "80%"> 

* 재현율과 정밀도가 균형을 맞추는 지점에서 두 지표가 각각 0.7도 안되는 것을 볼 수 있다.   
* **이는 데이터의 값을 다시 조정해야 할 필요가 있음을 말한다.**      

```
diabetes_data.describe()
```   

<img src="https://user-images.githubusercontent.com/52434993/89050013-1686bd80-d38d-11ea-8b31-676c172cbf5c.jpg" width = "80%">   

* min값이 0이 되면 안되는 feature들이 보인다.   
* 총 데이터의 개수가 적으므로 지우기보다는 평균값으로 이들을 대체한다.   

```
mean_zero_features = diabetes_data[zero_features].mean()
diabetes_data[zero_features] = diabetes_data[zero_features].replace(0,mean_zero_features)
```   

* 다시 학습을 시킨다. **(로지스틱 회귀의 경우 숫자 데이터에 스케일링을 적용하는 것이 좋다.)**   

```
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 156, stratify=y)

lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:,1]

get_clf_eval(y_test, pred, pred_proba)
```   

*get_clf_eval 함수는 5가지 지표들을 얻는 함수*   
<br>

```
임계값 :  0.3
오차 행렬
[[68 32]
 [10 44]]
정확도 0.7273, 정밀도 0.5789, 재현율 : 0.8148, F1 : 0.6769, AUC : 0.8404

임계값 :  0.33
오차 행렬
[[71 29]
 [11 43]]
정확도 0.7403, 정밀도 0.5972, 재현율 : 0.7963, F1 : 0.6825, AUC : 0.8404

임계값 :  0.36
오차 행렬
[[74 26]
 [14 40]]
정확도 0.7403, 정밀도 0.6061, 재현율 : 0.7407, F1 : 0.6667, AUC : 0.8404

임계값 :  0.39
오차 행렬
[[77 23]
 [14 40]]
정확도 0.7597, 정밀도 0.6349, 재현율 : 0.7407, F1 : 0.6838, AUC : 0.8404

임계값 :  0.42
오차 행렬
[[80 20]
 [16 38]]
정확도 0.7662, 정밀도 0.6552, 재현율 : 0.7037, F1 : 0.6786, AUC : 0.8404

임계값 :  0.45
오차 행렬
[[81 19]
 [17 37]]
정확도 0.7662, 정밀도 0.6607, 재현율 : 0.6852, F1 : 0.6727, AUC : 0.8404

임계값 :  0.48
오차 행렬
[[88 12]
 [19 35]]
정확도 0.7987, 정밀도 0.7447, 재현율 : 0.6481, F1 : 0.6931, AUC : 0.8404

임계값 :  0.5
오차 행렬
[[89 11]
 [20 34]]
정확도 0.7987, 정밀도 0.7556, 재현율 : 0.6296, F1 : 0.6869, AUC : 0.8404
```    

* 가장 균형 있는 지표를 가진 임계값은 0.48로 보인다.   
* 임계값을 0.48로 고정한 뒤에 에측을 실행한다.   

```
binarizer = Binarizer(threshold=0.48)

pred_th_048 = binarizer.fit_transform(pred_proba[:,1].reshape(-1,1))

get_clf_eval(y_test, pred_th_048, pred_proba[:,1])
```   

```
오차 행렬
[[88 12]
 [19 35]]
정확도 0.7987, 정밀도 0.7447, 재현율 : 0.6481, F1 : 0.6931, AUC : 0.8404
```   

* 데이터 조작을 통해서 조금 더 안정적인 정밀도와 재현율을 얻은 것을 확인할 수 있다.   

#### 많은 평가 지표들을 이용해서 데이터를 잘 다듬고, 더 좋은 평가 지표를 얻는 것을 목표로 해야함을 알 수 있다.



