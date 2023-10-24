---
layout: post
title: ML_guide/creditcardfraud
subtitle: "오버샘플링 -이상 분포 변화"
categories: ML_guide
tags: [ai]
---


## Ensemble - 신용카드 사기 검출(Kaggle)   

> ## Undersampling , Oversampling   

* 레이블 값이 불균형한 분포를 가질 때, 제대로 된 학습이 되자 않는다.   
* 따라서 적절한 학습 데이터를 얻기 위해서, **언더 샘플링(Undersampling)과 오버 샘플링(Oversampling)을 사용한다.**      

<img src = "https://user-images.githubusercontent.com/52434993/89797054-67549e00-db65-11ea-9f97-bb1d8f7a3a10.jpg">   

* 언더 샘플링(Undersampling)의 경우 정상 레이블 데이터를 너무 많이 감소시킨다는 단점 때문에, 오히려 학습에 방해가 된다.   
* 따라서 오버 샘플링(Oversampling)을 이용하며, 대표적으로 SMOTE(Synthetic Minority Over-sampling Technique)을 사용한다.   
<br>

---------------------------------   

<br>

> ## 실제 데이터를 이용한 분류   

* 데이터를 불러와서 피처를 파악한다.   

**[코드]**   
```
card_df = pd.read_csv('./train_creditcard.csv')
card_df.head()
```   

**[결과]**   

<img src="https://user-images.githubusercontent.com/52434993/89798007-94ee1700-db66-11ea-9eef-eb94c4654d33.jpg">   
<img src="https://user-images.githubusercontent.com/52434993/89797946-87389180-db66-11ea-8cfc-982a35c32350.jpg">   

* Amount 피처는 신용카드 트랜잭션 금액을 의미한다.   
* Class는 0의 경우 정상, 1의 경우 사기 트랜잭션이다.   
* Time 피처는 작업용 속성으로서 의미가 없다. 따라서 **삭제**한다.   
* 학습 데이터와 테스트 데이터로 분리한다.   

<br>

**[코드]**   
```
def get_preprocessed_df(df=None):
    df_copy = df.copy()
    df_copy.drop('Time', axis=1, inplace = True)
    return df_copy

def get_train_test_dataset(df=None):
    df_copy = get_preprocessed_df(df)
    X_features = df_copy.iloc[:,:-1]
    y_target = df_copy.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state=0, stratify=y_target)
    return X_train, X_test, y_train, y_test
```   

**[결과]**   
```
학습 데이터의 레이블 값 비율
0    99.827451
1     0.172549

테스트 데이터의 레이블 값 비율
0    99.826785
1     0.173215
```   

* **학습 데이터와 테스트 데이터의 레이블의 비율이 차이없이 잘 분할된 것을 확인할 수 있다.**   
<br>

* 로지스틱 회귀와 lightGBM을 이용하여 데이터를 학습시키고 예측 성능을 비교한다.   
<br>

**[코드]**   
```
def get_model_train_eval(model, ftr_train=None, ftr_test=None, tgt_train=None, tgt_test=None):
    model.fit(ftr_train, tgt_train)
    pred = model.predict(ftr_test)
    pred_proba = model.predict_proba(ftr_test)[:,1]
    get_clf_eval(tgt_test, pred, pred_proba)
    
    
lr_clf = LogisticRegression()
get_model_train_eval(lr_clf, ftr_train=X_train, ftr_test = X_test, tgt_train = y_train, tgt_test = y_test)

lgbm_clf = LGBMClassifier(n_estimators = 1000, num_leaves=64, n_jobs=1, boost_from_average=False)
get_model_train_eval(lgbm_clf, ftr_train=X_train, ftr_test = X_test, tgt_train = y_train, tgt_test = y_test)
```   

**[결과]**   
```
[로지스틱 회귀]
오차 행렬
[[85282    13]
 [   57    91]]
정확도 0.9992, 정밀도 0.8750, 재현율 : 0.6149, F1 : 0.7222, AUC : 0.9570

[lightGBM]
오차 행렬
[[85289     6]
 [   36   112]]
정확도 0.9995, 정밀도 0.9492, 재현율 : 0.7568, F1 : 0.8421, AUC : 0.9797
```   

* **lightGBM의 정밀도 재현율, AUC의 값이 로지스틱의 것보다 높은 것을 확인할 수 있다.**   
<br>

* **재현율과 AUC의 더 높이기 위해 데이터의 분포를 변화시킨다.**   

<img src= "https://user-images.githubusercontent.com/52434993/89800866-4cd0f380-db6a-11ea-99d6-dc29c58de222.jpg">

* 데이터의 분포를 보면 꼬리가 긴 형태의 분포 곡선을 가진 것을 볼 수 있다.(1000이하의 데이터가 대부분이다.)   
* 정규 분포 형태로 바꾼다.   

**[코드]**
```
def get_preprocessed_df(df=None):
    df_copy = df.copy()
    scaler = StandardScaler()
    amount_n = scaler.fit_transform(df_copy['Amount'].values.reshape(-1,1))
    df_copy.insert(0,'Amount_Scaled', amount_n)
    df_copy.drop(['Time','Amount'], axis=1, inplace=True)
    return df_copy

```   

* 다시 분류후 학습/예측/평가를 한다.   

```
[로지스틱 회귀]
오차 행렬
[[85281    14]
 [   58    90]]
정확도 0.9992, 정밀도 0.8654, 재현율 : 0.6081, F1 : 0.7143, AUC : 0.9702

[lightGBM]
오차 행렬
[[85289     6]
 [   36   112]]
정확도 0.9995, 정밀도 0.9492, 재현율 : 0.7568, F1 : 0.8421, AUC : 0.9773

```   

* 정규 분포 형태로 바꾸기 전과 큰 차이가 없음을 알 수 있다.   
<br>

* 로그 변환 기법을 이용해본다. (데이터의 분포도가 심하게 왜곡되었을 때, 자주 적용하는 기법)   

**[코드]**   
```
def get_preprocessed_df(df=None):
    df_copy = df.copy()
    amount_n = np.log1p(df_copy['Amount'])
    df_copy.insert(0,'Amount_Scaled', amount_n)
    df_copy.drop(['Time','Amount'], axis=1, inplace=True)
    return df_copy
```   

**[결과]**   
```
[로지스틱 회귀]
오차 행렬
[[85283    12]
 [   59    89]]
정확도 0.9992, 정밀도 0.8812, 재현율 : 0.6014, F1 : 0.7149, AUC : 0.9727

[lightGBM]
오차 행렬
[[85290     5]
 [   35   113]]
정확도 0.9995, 정밀도 0.9576, 재현율 : 0.7635, F1 : 0.8496, AUC : 0.9786
```   

* **두 모델 모두 재현율, AUC가 소폭 상승한 것을 볼 수 있다.**      

<br>

* **이상치 데이터**를 탐색하고 해당 데이터를 삭제한다.   
* IQR 방식으로 사분위 값을 이용한다.   

**[코드]**   
```
def get_outlier(df=None, column=None, weight = 1.5):
    fraud = df[df['Class']== 1][column]
    quantile_25 = np.percentile(fraud.values,25)
    quantile_75 = np.percentile(fraud.values,75)
    iqr = quantile_75 - quantile_25
    iqr_weight = iqr*weight
    lowest_val = quantile_25 - iqr_weight
    highest_val = quantile_75 + iqr_weight
    outlier_index = fraud[(fraud < lowest_val) | (fraud > highest_val)].index
    
    return outlier_index
```   

* 해당하는 행을 삭제하고 다시 분류/학습/예측을 수행한다.   

```
[로지스틱 회귀]
오차 행렬
[[85281    14]
 [   48    98]]
정확도 0.9993, 정밀도 0.8750, 재현율 : 0.6712, F1 : 0.7597, AUC : 0.9743

[lightGBM]
오차 행렬
[[85291     4]
 [   25   121]]
정확도 0.9997, 정밀도 0.9680, 재현율 : 0.8288, F1 : 0.8930, AUC : 0.9831
```   

* **이전의 재현율 60.14%, 76.35%에 비해서 67.12%, 82.88%로 크게 상승한 것을 볼 수 있다.**   
* 현재까지 이상치 제거하는 방법이 가장 높은 상승률을 보여준다.   

<br>

* 다음으로 SMOTE 오버샘플링을 적용하여 학습/예측/평가를 진행한다(**해당 데이터세트는 이상치를 제거한 데이터 세트이다.**)   

**[코드]**    
```
smote = SMOTE(random_state=0)
X_train_over, y_train_over = smote.fit_sample(X_train, y_train)
```   

**[결과]**   
```
SMOTE 적용전 학습용 피처/레이블 데이터 세트 :  (199364, 29) (199364,)
SMOTE 적용 후 학습용 피처/레이블 데이터 세트 :  (398040, 29) (398040,)
SMOTE 적용 후 레이블 값 분포 : 
0    199020
1       344
```   

* 위의 표에서 알 수 있듯이, **데이터 세트의 개수는 늘었지만, 레이블의 분포는 동일한 것을 볼 수 있다.**   
* 해당 데이터 세트에 학습/예측/평가를 진행한다.(로지스틱 회귀)

```
오차 행렬
오차 행렬
[[82937  2358]
 [   11   135]]
정확도 0.9723, 정밀도 0.0542, 재현율 : 0.9247, F1 : 0.1023, AUC : 0.9737
```

* 재현율이 92.47% 큰 상승을 보이는 반면 정밀도는 5.42%로 폭락한 것을 알 수 있다.   
* **이러한 현상은 오버 샘플링으로 인해 Class=1의 데이터가 지나치게 많아지면서 정밀도가 떨어진것으로 보인다.**   
* 임계값에 따라서 정밀도와 재현율이 어떠한 변화를 보이는지 시각화해서 확인한다.   

<img src= "https://user-images.githubusercontent.com/52434993/89802532-97536f80-db6c-11ea-840b-1a7a7b85ced5.jpg">   

* **임계값 0.99에서 상당히 민감도가 심한 것을 볼 수 있다. (0.99를 기준으로 정밀도, 재현율 값이 크게 변한다.)**   
* **따라서 로지스틱 회귀는 SMOTE로 오버샘플링된 데이터 세트에 적절하지 못하다고 할 수 있다.**      
* lightGBM 모델로 학습/예측/평가를 수행한다. 


```
오차 행렬
[[85286     9]
 [   22   124]]
정확도 0.9996, 정밀도 0.9323, 재현율 : 0.8493, F1 : 0.8889, AUC : 0.9789
```   

* 재현율은 82.88%에서 84.93%로 상승했다. 하지만 정밀도는 93.23%로 하락한 것을 볼 수 있다.   
* **SMOTE 오버 샘플림을 적용하면 일반적으로 재현율은 상승하고 정밀도가 떨어진다.**   














