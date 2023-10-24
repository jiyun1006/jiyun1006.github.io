---
layout: post
title: ML_guide/Regression(1)
subtitle: "회귀 분석(1)"
categories: ML_guide
tags: [ai]
---



## Regression(1)

> ## Residual sum of squares(RSS)   

* 실제 값과 회귀 모델의 차이의 제곱이다.   
* **최적의 회귀 모델을 위해서는 잔차의 합이 최소가 되어야 한다.**      

<br>

<img src ="https://user-images.githubusercontent.com/52434993/90238041-4390a100-de60-11ea-974b-2e2509b21edb.jpg" width="50%">   

<br>

> ## 경사 하강법(Gradient Descent)   

* RSS를 최소화해주는 방법.   
* **점진적인 반복 계산을 통해 회귀 계수의 값을 업데이트하며 오류를 최소화 한다.**      

<br>

<img src="https://user-images.githubusercontent.com/52434993/90238543-1b557200-de61-11ea-85cc-e46245648228.jpg" width = "60%">

<br>

* RSS를 각각의 회귀 계수에 대해서 편미분하고, 결과값을 반복적으로 보정한다.   
<br>
<br>

**[코드]**   

*np.dot을 이용한 내적으로 식을 완성한다.*

```
def get_weight_updates(w1, w0, x, y, learning_rate=0.01):
    N = len(y)
    w1_update = np.zeros_like(w1)
    w0_update = np.zeros_like(w0)
    y_pred = np.dot(x, w1.T) + w0
    diff = y-y_pred
    
    w0_factors = np.ones((N,1))
    w1_update = -(2/N)*learning_rate*(np.dot(x.T, diff))
    w0_update = -(2/N)*learning_rate*(np.dot(w0_factors.T, diff))
    
    return w1_update, w0_update

```   

```
def gradient_descent_steps(x,y,iters=10000):
    w0 = np.zeros((1,1))
    w1 = np.zeros((1,1))
    
    for ind in range(iters):
        w1_update, w0_update = get_weight_updates(w1, w0, x, y, learning_rate=0.01)
        w1 = w1 - w1_update
        w0 = w0 - w0_update
        
    return w1, w0
```   

**[결과]**
```
w1 : 4.022 w0 : 6.162
Gradient Descent Total Cost : 0.9935
```

**단, 경사 하강법은 수행 시간이 매우 오래 걸린다.**   
**따라서 일부 데이터만을 이용하는 확률적 경사 하강법(Stochastic Gradient Descent)를 이용한다.**    

<br>
<br>

**[코드]**   

*x,y 대신 따로 뽑은 sample_x, sample_y를 이용한다.*

```
def stochastic_gradient_descent_steps(x,y,batch_size=10, iters=1000):
    w0 = np.zeros((1,1))
    w1 = np.zeros((1,1))
    prev_cost  = 100000
    iter_index = 0
    
    for ind in range(iters):
        np.random.seed(ind)
        stochastic_random_index = np.random.permutation(x.shape[0])
        sample_x = x[stochastic_random_index[0:batch_size]]
        sample_y = y[stochastic_random_index[0:batch_size]]
        w1_update, w0_update = get_weight_updates(w1,w0,sample_x, sample_y, learning_rate = 0.01)
        w1 = w1 - w1_update
        w0 = w0 - w0_update
    return w1 , w0
```

**[결과]**   

```
w1 : 4.028 w0 : 6.156
Stochastic Gradient Descent Total Cost : 0.9937
```

**일반 하강 경사법의 회귀 계수값과 예측 오류비용이 큰 차이가 없다.**   
**따라서 큰 데이터를 처리할 경우 수행속도를 고려하여 확률적 하강 경사법을 사용한다.**   

<br>
<br>

> ## LinearRegression   

* LinearRegression 클래스는 RSS를 최소하하는 OLS(Ordinary Least Squares) 추정 방식으로 구현했다.    
* 회귀 모델은 피처의 독립성에 많은 영향을 받는다.   


* **회귀 모델을 평가하는 평가지표.**   

```   
MAE(Mean Absolute Error) : 실제 값과 예측값의 차이를 절대값으로 변환해 평균한 것.

MSE(Mean Squared Error) : 실제 값과 예측값의 차이를 제곱해 평균한 것.

RMSE(Root Mean Squared Error) : MSE에 루트를 씌운 값.

R^2 : 분산 기반으로 예측 성능 평가하는 지표. 1에 가까울수록 예측 정확도가 높다.
```    

<br>

**사이킷런 내장 데이터를 이용 선형회귀 모델 (보스턴 주택 가격 데이터)**          

* 데이터를 먼저 로드하고 DataFrame으로 변경한다.   

```
boston = load_boston()

bostonDF = pd.DataFrame(boston.data, columns = boston.feature_names)
```

<img src="https://user-images.githubusercontent.com/52434993/90241661-789ff200-de66-11ea-807e-073edef45a8e.jpg" width="80%">   

<br>

* 각각의 피처들이 'PRICE'에 미치는 영향을 알아본다.  

<img src="https://user-images.githubusercontent.com/52434993/90241862-d59ba800-de66-11ea-8f82-f21eb18690d6.jpg" width ="80%">

* 'RM'과 'LSTAT'의 영향이 가장 큰 것을 볼 수 있다.   
* 'RM'(방 개수)는 양 방향 선형성을 보여주고, 'LSTAT'(하위 계층의 비율)은 음 방향 선형성을 보여준다.    
* LinearRegression 클래스를 이용해서 회귀 모델을 만든다.   

<br>
<br>

**[코드]**   

```
y_target = bostonDF['PRICE']
X_data = bostonDF.drop(['PRICE'], axis=1, inplace=False)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size = 0.3, random_state=156)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_preds = lr.predict(X_test)
mse = mean_squared_error(y_test, y_preds)
rmse = np.sqrt(mse)
```

**[결과]**   

```
MSE : 17.297, RSME : 4.159
Variance score : 0.757


#절편 값과 각 회귀 계수의 값

절편 값 :  40.995595172164336
회귀 계수 값 :  [ -0.1   0.1   0.    3.  -19.8   3.4   0.   -1.7   0.4  -0.   -0.9   0. -0.6]
```

* 회귀 계수의 값을 순서대로 피처를 나열    

```
RM          3.4
CHAS        3.0
RAD         0.4
ZN          0.1
B           0.0
TAX        -0.0
AGE         0.0
INDUS       0.0
CRIM       -0.1
LSTAT      -0.6
PTRATIO    -0.9
DIS        -1.7
NOX       -19.8
```   


<br><br>

> ## 다항 회귀(Polynomial Regression)   

* 독립변수가 단항식이 아닌, 2,3차 방정식과 같은 다항식으로 표현 되는 것.   
* 단순 선형 회귀보다 예측 성능이 높다.   
* 하지만 차수가 너무 높아지게 되면 학습 데이터에만 맞춘 학습이 이루어진다.(과적합)   

<br><br>

**과적합과 과소적합의 예시**   

* 원래 데이터 세트는 피처 X 와 target y가 약간의 잡음이 포함된 다항식의 코사인 그래프 관계이다.   
* 이를 다항 회귀의 차수를 변화시키며 예측곡선의 모양과 정확도를 비교한다.   
* 먼저 조건에 맞는 학습 데이터를 생성한다.

<br>
 

**[코드]**   

*X는 0부터 1까지 30개의 임의의 값을 순서대로 샘플링.*   
*y는 코사인 기반의 true_fun()에서 약간의 노이즈를 포함한 값.   
(약간의 노이즈 --> np.random.randn(n_samples) * 0.1)*   

```
def true_fun(X):
    return np.cos(1.5 * np.pi * X)

np.random.seed(0)
n_samples = 30
X = np.sort(np.random.rand(n_samples))

y = true_fun(X) + np.random.randn(n_samples) * 0.1
```

* 차수 1, 4, 15 로 변화시키면서 다항회귀 비교   

<br>

**[코드]**   

*개별 차수별로 polynomial로 변환한다.(Pipeline 이용)*      
*교차 검증으로 다항회귀를 평가*   
*테스트 데이터를 0 부터 1까지 만든다.(linspace이용)*   
*테스트 데이터에 회귀 예측을 수행하고 예측 곡선과 실제곡선을 비교한다.*   

```
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())
    
    polynomial_features = PolynomialFeatures(degree = degrees[i], include_bias = False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([('poly', polynomial_features), ('linear', linear_regression)])
    pipeline.fit(X.reshape(-1,1), y)
    
    
    scores = cross_val_score(pipeline, X.reshape(-1,1), y, scoring='neg_mean_squared_error', cv= 10)
    
    coefficients = pipeline.named_steps['linear'].coef_
    print('\n Degree {} 회귀 계수는 {} 입니다.'.format(degrees[i], np.round(coefficients, 2)))
    print('Degree {} MSE는 {} 입니다.'.format(degrees[i], -1*np.mean(scores)))
    
    
    X_test = np.linspace(0,1,100)
    plt.plot(X_test, pipeline.predict(X_test[:,np.newaxis]), label = "Model")
    plt.plot(X_test, true_fun(X_test), '--', label='True function')
    plt.scatter(X,y, edgecolor = 'b', s=20, label = "Samples")
```   

**[결과]**   
```
Degree 1 회귀 계수는 [-1.61] 입니다.
Degree 1 MSE는 0.40772896250986834 입니다.

Degree 4 회귀 계수는 [  0.47 -17.79  23.59  -7.26] 입니다.
Degree 4 MSE는 0.04320874987231747 입니다.

Degree 15 회귀 계수는 [-2.98295000e+03  1.03899930e+05 -1.87417069e+06  2.03717225e+07
 -1.44873988e+08  7.09318780e+08 -2.47066977e+09  6.24564048e+09
 -1.15677067e+10  1.56895696e+10 -1.54006776e+10  1.06457788e+10
 -4.91379977e+09  1.35920330e+09 -1.70381654e+08] 입니다.
Degree 15 MSE는 182815433.47648773 입니다
```

<img src = "https://user-images.githubusercontent.com/52434993/90248083-eeaa5600-de72-11ea-93eb-5c1d841ef328.jpg" width = "80%">   

<br>

* **차수가 1일 때는 단순 선형 회귀랑 같기 때문에, 예측 곡선이 학습 데이터의 패턴을 반영하지 못했다.**   
* **따라서 과소적합 모델이다.**   

<br>

* **차수가 4일 때는 실제 데이터 세트와 예측곡선이 유사한 모습을 보인다. 사소한 잡음을 제대로 예측하지는 못했지만, 잘 예측한 곡선이다.**   

<br>

* **차수가 15일 때는 MSE의 값과 예측 곡선 둘 다 말도 안되는 모습을 보인다.**   
* **예측 곡선은 잡음을 지나치게 반영하여, 학습 데이터에만 치중된 모습을 보인다.**   
* **따라서 과적합 모델이다.**   

<br><br>

> ## 편향 - 분산 트레이드 오프 (Bias-Variance Trade Off)   

* 앞선 데이터에서 차수가 1일때의 모델은 고편향(high bias)성을 가진다.(**과소적합**)
* 또 차수가 15일때의 모델은 고분산(high variance)성을 가진다.(**과적합**)
* **따라서 편향과 분산이 서로 트레이드 오프를 이루면서 오류 비용이 최대로 낮아지는 모델을 구축하는 것이 효율적이다.**

<br>

<img src ="https://user-images.githubusercontent.com/52434993/90248576-edc5f400-de73-11ea-92c2-b2c59bd7dd12.jpg" width ="70%">
















