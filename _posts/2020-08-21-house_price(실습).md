---
layout: post
title: ML_guide/house_price
subtitle: "ridge lasso 실습"
categories: ML_guide
tags: [ai]
---



## House_Price - Regression   

> ## 데이터 전처리(preprocessing)

<br>

```
데이터 세트의 shape :  (1460, 81)

전체 피처의 type 
object     43
int64      35
float64     3
dtype: int64

NULL 칼럼과 그 건수 : 
PoolQC          1453
MiscFeature     1406
Alley           1369
Fence           1179
FireplaceQu      690
LotFrontage      259
GarageYrBlt       81
GarageType        81
GarageFinish      81
GarageQual        81
GarageCond        81
BsmtFinType2      38
BsmtExposure      38
BsmtFinType1      37
BsmtCond          37
BsmtQual          37
MasVnrArea         8
MasVnrType         8
Electrical         1
dtype: int64
```   
* 총 81개의 피처가 있고, 그중 object형이 43개, 숫자 자료형이 38개이다.   
* 또 null 칼럼의 개수도 파악한다.   
* null 값이 너무 많은 'PoolQC', 'MiscFeature', 'Alley', 'Fence' 는 드롭하고,
나머지 null 값은 평균값으로 대체한다.   
* 데이터 값(타겟 값 = 'SalePrice')의 분포가 정규분포를 따르는지 확인한다.   

<br>
<img src = "https://user-images.githubusercontent.com/52434993/90861865-72aa9380-e3c7-11ea-80ea-eb0b069748c2.jpg">   

* 왼쪽으로 편향된 것을 알 수 있다. 따라서 로그 변환을 이용해 분포도를 변화 시킨다.   

<br>
<img src ="https://user-images.githubusercontent.com/52434993/90862223-13994e80-e3c8-11ea-817c-6457d4d5831b.jpg">    

* 정규 분포 형태로 결괏값이 분포하는 것을 확인할 수 있다.   
* null값이 있는 숫자형 피처는 평균값으로 대체하고,
문자형 피처는 원-핫-인코딩으로 변환을 한다.   

**[코드]**   
*pandas의 get_dummies()를 이용하여 null 값을 None 칼럼으로 대체한다.*   
*또한 이로 인해 칼럼의 개수가 늘어나므로, 얼마나 늘었는지 확인한다.*   
```
house_df_ohe = pd.get_dummies(house_df)

null_column_count = house_df_ohe.isnull().sum()[house_df_ohe.isnull().sum()>0]
```

**[결과]**   
```
### Null 피처의 type : 
Series([], dtype: object)
 
get_dummies() 수행 전 데이터 shape :  (1460, 75)
get_dummies() 수행 후 데이터 shape :  (1460, 271)

```
* null 값이 모두 사라지고, 데이터 칼럼의 개수가 늘어난 것을 볼 수 있다.   

<br>
<br>


> ## 회귀 모델 학습/예측/평가   

* 로그 변환을 이용하였으므로, RMSLE를 측정하는 함수를 만든다.   

**[코드]**
*LinearRegression, Ridge, Lasso를 사용한다.*

```
def get_rmse(model):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    print(model.__class__.__name__,'로그 변환된 RMSE : ', np.round(rmse,3))
    return rmse


def get_rmses(models):
    rmses = []
    for model in models:
        rmse = get_rmse(model)
        rmses.append(rmse)
    return rmses
```   

**[결과]**
```
LinearRegression 로그 변환된 RMSE :  0.132
Ridge 로그 변환된 RMSE :  0.128
Lasso 로그 변환된 RMSE :  0.176
```   

* **Lasso 회귀의 경우 다른 두 경우에 비해 성능이 좋지 않다.**      
* 따라서 각 회귀 모델별 피처의 회귀 계수를 시각화해서 구성을 확인한다.   

<br>

*상위 10개, 하위 10개*   
<img src ="https://user-images.githubusercontent.com/52434993/90864447-a4255e00-e3cb-11ea-87f8-66fa700dcbea.jpg" width="80%">   

* **Lasso 회귀 계수의 값은 극단적으로 크거나 작다.**      
* 데이터 분할 시에 문제가 있거나, 데이터 자체에 문제가 있다고 생각할 수 있다.   
* **따라서 데이터 세트를 분할하지 않고, 전체 데이터를 교차 검증으로 분할해 RMSE를 측정한다.**      

**[코드]**   
```
def get_avg_rmse_cv(models):
    for model in models:
        rmse_list = np.sqrt(-cross_val_score(model, X_features, y_target, scoring='neg_mean_squared_error', cv=5))
        
        rmse_avg = np.mean(rmse_list)
        print('\n {} CV RMSE 값 리스트 : {}'.format(model.__class__.__name__, np.round(rmse_list,3)))
        print('{} CV 평균 RMSE값 : {}'.format(model.__class__.__name__, np.round(rmse_avg,3)))

```   

**[결과]**
```
LinearRegression CV RMSE 값 리스트 : [0.135 0.165 0.168 0.111 0.198]
LinearRegression CV 평균 RMSE값 : 0.155

Ridge CV RMSE 값 리스트 : [0.117 0.154 0.142 0.117 0.189]
Ridge CV 평균 RMSE값 : 0.144

Lasso CV RMSE 값 리스트 : [0.161 0.204 0.177 0.181 0.265]
Lasso CV 평균 RMSE값 : 0.198
```   

* **원 데이터 세트로 교차 검증을 해봐도 Lasso의 RMSE값이 유독 높은 것을 확인할 수 있다.**   
* Ridge 모델과 Lasso 모델의 최적 하이퍼 파라미터를 찾아내어 RMSE값을 확인한다.   

**[코드]**
```
def print_best_params(model, params):
    grid_model = GridSearchCV(model, param_grid=params, scoring='neg_mean_squared_error',cv=5)
    grid_model.fit(X_features, y_target)
    rmse = np.sqrt(-1*grid_model.best_score_)
    print('{} 5 CV 시 최적 평균 RMSE 값 : {}, 최적 alpha : {}'.format(model.__class__.__name__, np.round(rmse,4), grid_model.best_params_))
    
```


**[결과]**
```
Ridge 5 CV 시 최적 평균 RMSE 값 : 0.1418, 최적 alpha : {'alpha': 12}
Lasso 5 CV 시 최적 평균 RMSE 값 : 0.142, 최적 alpha : {'alpha': 0.001}
```   

* **최적 하이퍼 파라미터로 인해 RMSE 값이 낮아진 것을 확인할 수 있다.**      
* 다시 데이터를 분할하고 하이퍼 파라미터를 적용해 모델의 성능을 확인한다.   

```
LinearRegression 로그 변환된 RMSE :  0.132
Ridge 로그 변환된 RMSE :  0.124
Lasso 로그 변환된 RMSE :  0.12
```   

<img src="https://user-images.githubusercontent.com/52434993/90865182-f74be080-e3cc-11ea-9054-5ff8aabcf3cf.jpg" width="80%">

* **성능이 좋아진 모습을 확인할 수 있습니다. 하지만, 아직 피처간의 회귀계수 값이
많이 차이가 나므로, 추가적인 데이터 가공이 필요합니다.**      

<br>
<br>


> ## 데이터 전처리(피처데이터)   

* 피처 데이터의 데이터 분포도를 확인하고, 얼마나 왜곡되었는지 확인한다.   

**[코드]**
*scipy 모듈의 skew()함수를 이용하여 왜곡의 정도를 추출한다.*       
*skew의 값이 1이상이면 왜곡의 정도가 심하다고 생각할 수 있다.*    
*원-핫-인코딩으로 카테고리 숫자형 피처는 왜곡될 가능성이 높으므로 제외한다.*   
```
features_index = house_df.dtypes[house_df.dtypes != 'object'].index
skew_features = house_df[features_index].apply(lambda x : skew(x))
skew_features_top = skew_features[skew_features > 1]
```   

**[결과]**
```
MiscVal          24.451640
PoolArea         14.813135
LotArea          12.195142
3SsnPorch        10.293752
LowQualFinSF      9.002080
KitchenAbvGr      4.483784
BsmtFinSF2        4.250888
ScreenPorch       4.117977
BsmtHalfBath      4.099186
EnclosedPorch     3.086696
MasVnrArea        2.673661
LotFrontage       2.382499
OpenPorchSF       2.361912
BsmtFinSF1        1.683771
WoodDeckSF        1.539792
TotalBsmtSF       1.522688
MSSubClass        1.406210
1stFlrSF          1.375342
GrLivArea         1.365156
```   

* 위의 피처들을 로그변환 후, 다시 Ridge, Lasso 모델의 하이퍼 파라미터와 RMsE를 확인한다.   

```
Ridge 5 CV 시 최적 평균 RMSE 값 : 0.1275, 최적 alpha : {'alpha': 10}
Lasso 5 CV 시 최적 평균 RMSE 값 : 0.1252, 최적 alpha : {'alpha': 0.001}
```   

<img src="https://user-images.githubusercontent.com/52434993/90867247-0b451180-e3d0-11ea-917d-72c452a21f77.jpg" width="80%">

* 다음으로 이상치를 제거한다.    
* 현재 세 모델에서 가장 큰 회귀계수 값을 가진 'GrLivArea'와 타깃 값의 관계를 시각화한다.   

<br>
<img src ="https://user-images.githubusercontent.com/52434993/90867644-adfd9000-e3d0-11ea-86ac-e51ab0cea461.jpg">   

* 양의 상관관계가 높은 것을 알 수 있다.   
* **하지만 일부 데이터가 관계에서 어긋난 것을 알 수 있고, 이 데이터를 삭제한다.**       
* 다시 세모델의 RMSLE값을 확인한다.   

```
Ridge 5 CV 시 최적 평균 RMSE 값 : 0.1125, 최적 alpha : {'alpha': 8}
Lasso 5 CV 시 최적 평균 RMSE 값 : 0.1122, 최적 alpha : {'alpha': 0.001}


LinearRegression 로그 변환된 RMSE :  0.129
Ridge 로그 변환된 RMSE :  0.103
Lasso 로그 변환된 RMSE :  0.1
```   

<br>

<img src="https://user-images.githubusercontent.com/52434993/90868221-83600700-e3d1-11ea-9281-9be6c579d8f9.jpg" width="80%">   


* 최적 하이퍼 파라미터의 변화가 있었고, 이를 적용하여 성능을 확인한다.   
* 세 모델 모두 이상치를 제거해서 상당한 성능의 향상을 확인할 수 있다.   
* **'GrLivArea'가 모델에서 차지하는 영향도가 크기 때문에, 
이상치를 제거하여 성능이 크게 올라간 것으로 보인다.**   


<br>
<br>


> ## 예측결과 혼합   

* 개별 회귀 모델의 예측 결과값을 혼합하여 최종 회귀 값을 예측한다.   
* 앞의 Ridge, Lasso로 혼합을 한다.    

**[코드]**   
```
def get_rmse_pred(preds):
    for key in preds.keys():
        pred_value = preds[key]
        mse = mean_squared_error(y_test, pred_value)
        rmse = np.sqrt(mse)
        print('{} 모델의 RMSE : {}'.format(key, rmse))
        
        
ridge_pred = ridge_reg.predict(X_test)
lasso_pred = lasso_reg.predict(X_test)

pred = 0.4 * ridge_pred + 0.6 * lasso_pred        
```

**[결과]**
```
최종혼합 모델의 RMSE : 0.10007930884470519
Ridge 모델의 RMSE : 0.10345177546603272
Lasso 모델의 RMSE : 0.10024170460890039
```

* 둘 중 성능이 더 좋은 쪽에 가중치를 더 둔다.
* **개별 모델보다 혼합 모델의 성능이 약간 개선된 것을 알 수 있다.**
















