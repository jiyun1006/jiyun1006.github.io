---
layout: post
title: ML_guide/Regression(2)
subtitle: "회귀 분석(2)"
categories: ML_guide
tags: [ai]
---


## Regression(2)   

> ## 규제선형모델 (Regularized_linear_model)      

* RSS를 최소화하는 모델만 고려하다 보면, **테스트 데이터에 예측 성능이 저하되기 쉽다.**      
* 따라서 **회귀 계수의 크기 제어를 수행하는 역할(alpha)이 필요하다.**      

<br>

**[Ridge]**   
```
Min(RSS(W) + alpha * ||W||^2
```   

**[Lasso]**
```
Min(RSS(W) + alpha * ||W||
```   

**[ElasticNet]**
```
Min(RSS(W) + alpha1 * ||W|| + alpha2 * ||W||^2
```   
<br>
<br>

> ## Ridge    

* **회귀 계수(W)의 제곱에 규제를 가하는 방법이 Ridge 회귀이다. (L2규제)**        
* 앞선 보스턴 주택 가격 데이터를 활용하여 Ridge 회귀를 구현한다.   

**[코드]**
```
ridge = Ridge(alpha = 10)
neg_mse_scores = cross_val_score(ridge, X_data, y_target, scoring = "neg_mean_squared_error", cv = 5)
rmse_scores = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)
```   

**[결과]**
```
5 folds의 개별 Negative MSE scores :  [-11.422 -24.294 -28.144 -74.599 -28.517]
5 folds의 개별 RMSE scores :  [3.38  4.929 5.305 8.637 5.34 ]
5 folds의 평균 RMSE : 5.518
```

* **규제없는 linearregression보다 값이 작으므로, 더 뛰어난 성능을 보여준다고 할 수 있다.**   
* alpha값을 변화 시키면서 RMSE의 변화를 살펴본다.   

```
alpha가 0 일 때 5 folds의 평균 RMSE : 5.829
alpha가 0.1 일 때 5 folds의 평균 RMSE : 5.788
alpha가 1 일 때 5 folds의 평균 RMSE : 5.653
alpha가 10 일 때 5 folds의 평균 RMSE : 5.518
alpha가 100 일 때 5 folds의 평균 RMSE : 5.330
```   

* alpha가 100일 때, RMSE가 5.330으로 가장 높은 성능을 보여준다.   
* 시각화를 이용해서 alpha값의 변화에 따른 회귀계수 값의 변화를 살펴본다.   

<img src="https://user-images.githubusercontent.com/52434993/90307338-eeac6380-df0f-11ea-97a9-fd58509fe1ba.jpg" width = "100%">   

<img src ="https://user-images.githubusercontent.com/52434993/90307411-c96c2500-df10-11ea-8f32-3927f0c5bb68.jpg" width = "50%">


* 회귀 계수의 절댓값이 너무 컸던 'NOX'의 회귀 계수가 작아지는 것을 확인할 수 있다.   
* **Ridge 회귀의 경우에는 회귀 계수를 0으로 만들지 않는다.**   

<br>
<br>

> ## Lasso   

* **회귀계수(W)의 절댓값에 규제를 가하는 것이 Lasso 회귀이다. (L1규제)**     
* alpha값을 변화 시키면서 RMSE의 변화를 살펴본다.      

```
alpha 0.07일 때 5 폴드 세트의 평균 RMSE : 5.612
alpha 0.1일 때 5 폴드 세트의 평균 RMSE : 5.615
alpha 0.5일 때 5 폴드 세트의 평균 RMSE : 5.669
alpha 1일 때 5 폴드 세트의 평균 RMSE : 5.776
alpha 3일 때 5 폴드 세트의 평균 RMSE : 6.189
```   

* alpha가 0.07일 떄, RMSE가 5.612로 가장 높은 성능을 보여준다.    
* 시각화를 이용해서 alpha값의 변화에 따른 회귀계수 값의 변화를 살펴본다.   

<img src ="https://user-images.githubusercontent.com/52434993/90307438-06d0b280-df11-11ea-9132-3e0f75c474a6.jpg" width ="100%">

<img src ="https://user-images.githubusercontent.com/52434993/90307451-29fb6200-df11-11ea-9496-9e7a3cb779c8.jpg" width = "50%">    

* **불필요한 회귀 계수를 0으로 만드는 것을 확인할 수 있다.**   

<br>
<br>

> ## ElasticNet   

* **L2 규제와 L1 규제를 결합한 회귀이다.**   
* Lasso 회귀와 같이 중요 피처를 제외하고 회귀계수를 0으로 만든다면,   
alpha값에 따라 회귀계수가 급변할 수 있다.   
* 따라서 이를 완화하기 위해 L2 규제를 추가한다.   

*ElasticNet의 규제 : a * L1 + b * L2*   
*l1_ratio =  a/(a+b)*   
```
alpha 0.07일 때 5 폴드 세트의 평균 RMSE : 5.542
alpha 0.1일 때 5 폴드 세트의 평균 RMSE : 5.526
alpha 0.5일 때 5 폴드 세트의 평균 RMSE : 5.467
alpha 1일 때 5 폴드 세트의 평균 RMSE : 5.597
alpha 3일 때 5 폴드 세트의 평균 RMSE : 6.068
```   

* alpha가 0.5일 때 RMSE가 5.467로 가장 높은 성능을 보여준다.   
* 시각화를 이용해서 alpha값의 변화에 따른 회귀계수 값의 변화를 살펴본다.      

<img src = "https://user-images.githubusercontent.com/52434993/90307545-1c92a780-df12-11ea-9f8e-c12b4a74b0f4.jpg" width ="100%">

<img src = "https://user-images.githubusercontent.com/52434993/90307554-33d19500-df12-11ea-9b64-23e90becc8fb.jpg" width ="50%">   

* **Lasso와 비교해서 0이된 피처의 개수가 적은 것을 알 수 있다.**    

<br><br>

> ## 선형회귀를 위한 데이터 변환   

* 왜곡된 형태의 분포도일 경우 예측 성능에 부정적인 영향을 끼친다.   
* 따라서 모델을 적용하기 전에 **데이터에 대한 스케일링/정규화 작업**을 하는 것이 일반적이다.   
* 표준정규분포 변환, 최댓값/최솟값 정규화, 로그 변환 세 가지 방법이 있다.   
* 경우에 따라서 다항식 특성을 위해 다할식 차수도 입력한다.   


```
## 변환 유형 : None, Polynomial Degree :None
alpha 0.1일 때 5 폴드 세트의 평균 RMSE : 5.788
alpha 1일 때 5 폴드 세트의 평균 RMSE : 5.653
alpha 10일 때 5 폴드 세트의 평균 RMSE : 5.518
alpha 100일 때 5 폴드 세트의 평균 RMSE : 5.330

## 변환 유형 : Standard, Polynomial Degree :None
alpha 0.1일 때 5 폴드 세트의 평균 RMSE : 5.826
alpha 1일 때 5 폴드 세트의 평균 RMSE : 5.803
alpha 10일 때 5 폴드 세트의 평균 RMSE : 5.637
alpha 100일 때 5 폴드 세트의 평균 RMSE : 5.421

## 변환 유형 : Standard, Polynomial Degree :2
alpha 0.1일 때 5 폴드 세트의 평균 RMSE : 8.827
alpha 1일 때 5 폴드 세트의 평균 RMSE : 6.871
alpha 10일 때 5 폴드 세트의 평균 RMSE : 5.485
alpha 100일 때 5 폴드 세트의 평균 RMSE : 4.634 *

## 변환 유형 : MinMax, Polynomial Degree :None
alpha 0.1일 때 5 폴드 세트의 평균 RMSE : 5.764
alpha 1일 때 5 폴드 세트의 평균 RMSE : 5.465
alpha 10일 때 5 폴드 세트의 평균 RMSE : 5.754
alpha 100일 때 5 폴드 세트의 평균 RMSE : 7.635

## 변환 유형 : MinMax, Polynomial Degree :2
alpha 0.1일 때 5 폴드 세트의 평균 RMSE : 5.298
alpha 1일 때 5 폴드 세트의 평균 RMSE : 4.323 *
alpha 10일 때 5 폴드 세트의 평균 RMSE : 5.185
alpha 100일 때 5 폴드 세트의 평균 RMSE : 6.538

## 변환 유형 : Log, Polynomial Degree :None
alpha 0.1일 때 5 폴드 세트의 평균 RMSE : 4.770 *
alpha 1일 때 5 폴드 세트의 평균 RMSE : 4.676  *
alpha 10일 때 5 폴드 세트의 평균 RMSE : 4.836 *
alpha 100일 때 5 폴드 세트의 평균 RMSE : 6.241 
```

* 일반적으로 다항식 변환을 했을 때, 성능이 개선됨을 볼 수 있다.   
* 하지만 **다항식 개선은 피처의 개수가 많거나 데이터 크다면 시간이 많이 걸리는 단점이 있다.**        
* 그에 비해 **log 변환은 alpha 값의 변화에도 대체로 좋은 성능 향상을 보인다.**   

<br>
<br>


> ## Logistic Regression   

* 선형회귀 방식을 분류에 적용한 알고리즘이다.   
* 일반적인 선형 함수의 회귀 최적선을 찾는 것이 아니라, 시그모이드 함수 최적선을 찾는다.   

<img src ="https://user-images.githubusercontent.com/52434993/90308072-4a2e1f80-df17-11ea-93d6-1d212eea2e87.jpg" width = "80%">   

* **일반적인 연속형 값을 구하는 것 뿐만 아니라, 이산형 값도 시그모이드 함수를 이용하여 정확한 분류가 가능한다.**   


