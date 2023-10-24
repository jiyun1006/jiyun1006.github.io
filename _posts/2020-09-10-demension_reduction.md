---
layout: post
title: ML_guide/Demension Reduction
subtitle: "차원 축소 기법 (PCA, LDA, SGD...)"
categories: ML_guide
tags: [ai]
---

## Dimension_Reduction   

> ## PCA(Principal Component Analysis)   

* 가장 대표적인 차원 축소 기범.   
* **여러 변수 간에 상관관계를 이용해 주성분을 추출하여 차원을 축소한다.**     


<br>

<img src ="https://user-images.githubusercontent.com/52434993/91048814-2c6b6380-e657-11ea-9788-49743b6f5dd7.jpg" width = "60%">   

<br>

**붗꽃 데이터로 PCA변환 실습.**   

<br>

<img src="https://user-images.githubusercontent.com/52434993/91049097-a4d22480-e657-11ea-9f22-8933ddb5071d.jpg" width = "80%">

* 'sepal length'와 'sepal width'를 기준으로 데이터 분포를 살펴본다.   

<br>

**[코드]**
```
markers=['^','s','o']

for i, marker in enumerate(markers):
    x_axis_data = irisDF[irisDF['target'] ==i]['sepal_length']
    y_axis_data = irisDF[irisDF['target'] ==i]['sepal_width']
    plt.scatter(x_axis_data, y_axis_data, marker = marker, label=iris.target_names[i])
```

**[결과]**   

<img src ="https://user-images.githubusercontent.com/52434993/91049228-d2b76900-e657-11ea-8328-941fee74c012.jpg"> 

<br>

* 'setosa' 품종의 경우 일정하게 분포되어 있고, 분류가 잘 되어있다.   
* 하지만 나머지 두 품종은 분류가 어렵다.   
* 다음으로 PCA로 4개의 속성을 2개로 압축한 뒤 비교한다.   

<br>

**[코드]**   
*여러 속성의 값을 연산하므로 속성값을 동일하게 스케일링 해야 한다.*   
*그런 다음에 PCA를 적용해 2차원 데이터로 변환한다.*   
```  
iris_scaled = StandardScaler().fit_transform(irisDF.iloc[:,:-1])



pca = PCA(n_components=2)

pca.fit(iris_scaled)
iris_pca = pca.transform(iris_scaled)
print(iris_pca.shape)

pca_columns = ['pca_component_1','pca_component_2']
irisDF_pca = pd.DataFrame(iris_pca, columns = pca_columns)
irisDF_pca['target'] = iris.target
```   


**[결과]**
<br>

<img src ="https://user-images.githubusercontent.com/52434993/91049554-51aca180-e658-11ea-837b-052923c57eea.jpg">

<br>

* PCA 변환된 데이터를 가지고 시각화를 한다.   

<br>
<img src ="https://user-images.githubusercontent.com/52434993/91049656-799c0500-e658-11ea-9ffc-e051c5839475.jpg" width="60%">
<br>

* 변환 전보다 비교적 'versicolor' 과 'virginica'의 분류가 잘 구분된 것을 볼 수 있다.   

<br>

**신용카드 고객 데이터로 PCA 실습(uci 데이터)**   

* 총 23개의 피처를 가지고 있고, 30000개의 데이터를 가진다.   
* 각각의 속성끼리의 상관도를 파악하기 위해 시각화를 한다.   

<br>
<img src = "https://user-images.githubusercontent.com/52434993/91049965-f62ee380-e658-11ea-8620-ab3759776760.jpg" width="80%">
<br>

* 피처들끼리의 가장 높은 상관도를 가지는 곳은 BILL_AMT1~6 피처이다.   
* 해당 피처들의 PCA변환뒤의 변동성을 확인한다.   

<br>
 
**[코드]**   
*explained_variance_ratio 속성을 이용하여 변동성을 확인한다.*   
```
cols_bill = ['BILL_AMT'+str(i) for i in range(1,7)]

scaler = StandardScaler()
df_cols_scaled = scaler.fit_transform(X_features[cols_bill])
pca = PCA(n_components=2)
pca.fit(df_cols_scaled)
print('PCA Component별 변동성 : ', pca.explained_variance_ratio_)
```   

**[결과]**   
```
대상 속성명 :  ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
PCA Component별 변동성 :  [0.90555253 0.0509867 ]
```   

* **첫 번째 PCA축으로 90%의 변동성을 수용할 수 있다.**      
* 따라서 6개의 피처가 상관도가 매우 높음을 알 수 있다.   
* 마지막으로 PCA 변환 전과, 변환 후의 정확도를 비교한다.   

<br>

*변환 전 정확도*
```
CV=3인 경우의 개별 Fold세트별 정확도 :  [0.8083 0.8196 0.8232]
평균 정확도 : 0.8170
```

*변환 후 정확도*
```
PCA 변환 데이터 교차 검증 개별 정확도 : [0.7892 0.799  0.8026]
PCA 변환 데이터 평균 정확도: 0.7969333333333334
```   

* **비록 1%정도의 성능저하가 있지만, 전체 피처의 1/4으로도
괜찮은 예측 성능을 보여준것으로 생각한다.**       

<br>
<br>

> ## LDA(Linear Discriminant Analysis)   

* PCA와 유사하게 데이터 세트의 차원을 축소한다.   
* 데이터의 결정 값 클래스를 최대한으로 분리할 수 있는 축을 찾는다.   
* **클래스 간의 분산은 최대로, 클래스 내부의 분산은 최저로 한다.**   

<br>

<img src ="https://user-images.githubusercontent.com/52434993/91052639-b36f0a80-e65c-11ea-951f-6f80d0f93085.jpg" width = "80%">
                                                                                                                       
<br>

* 앞의 붓꽃 데이터 예제로 LDA 변환하고 시각화를 해본다.   

<br>

**[코드]**   
*PCA와는 다르게 fit 메서드를 호출할 때 결정값을 입력한다.*    
```
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(iris_scaled, iris.target)
iris_lda = lda.transform(iris_scaled)
```


**[결과]**   

<img src ="https://user-images.githubusercontent.com/52434993/91052876-0779ef00-e65d-11ea-9ddb-8e836548eebf.jpg" width ="80%">

<br>
<br>

> ## SVD(Singular Value Decomposition)   

* PCA와 유사한 행렬 분해 기법이다.   

```
A=UΣVT
```

* **일반적으로 sigma의 비대각인 부분과 대각원소 중에 특이값이 0인 부분도 제거하고,
그에 대응되는 U, Vt의 원소도 함께 제거해 차원을 줄인형태로 SVD를 적용한다.**   

* **Truncated SVD는 대각원소 상위 몇개만 추출해서 더욱 차원을 줄인 형태로 분해한다.**    
* 간단한 행렬로 SVD 분해를 실험해본다.   

<br>

**[코드]**   
*4x4 랜덤 행렬을 생성한다.*   
*svd() 를 이용하여 U, Sigma, Vt에 해당하는 값을 반환한다.*   
*다시 내적을 하여 원본 행렬로 복원되는지 확인한다.*    
```
np.random.seed(121)
a = np.random.randn(4,4)


U, Sigma, Vt = svd(a)


Sigma_mat = np.diag(Sigma)
a_ = np.dot(np.dot(U, Sigma_mat), Vt)
```   

**[결과]**    
```
[[-0.212 -0.285 -0.574 -0.44 ]
 [-0.33   1.184  1.615  0.367]
 [-0.014  0.63   1.71  -1.327]
 [ 0.402 -0.191  1.404 -1.969]]
 
 
 U matrix : 
 [[-0.079 -0.318  0.867  0.376]
 [ 0.383  0.787  0.12   0.469]
 [ 0.656  0.022  0.357 -0.664]
 [ 0.645 -0.529 -0.328  0.444]]
Sigma Value : 
 [3.423 2.023 0.463 0.079]
V transpose matrix : 
 [[ 0.041  0.224  0.786 -0.574]
 [-0.2    0.562  0.37   0.712]
 [-0.778  0.395 -0.333 -0.357]
 [-0.593 -0.692  0.366  0.189]]
 
 
 [[-0.212 -0.285 -0.574 -0.44 ]
 [-0.33   1.184  1.615  0.367]
 [-0.014  0.63   1.71  -1.327]
 [ 0.402 -0.191  1.404 -1.969]]
```   

**붓꽃 데이터를 이용한 PCA, TruncatedSVD 비교**   

* 사이킷런의 TruncatedSVD 클래스를 활용하여, 데이터 세트를 분류한다.    

<br>

*TruncatedSVD*   

<img src ="https://user-images.githubusercontent.com/52434993/91057121-c7693b00-e661-11ea-9f86-6d2896f5dbe6.jpg" width ="50%">

<br>

*PCA*   

<img src ="https://user-images.githubusercontent.com/52434993/91049228-d2b76900-e657-11ea-8328-941fee74c012.jpg" width ="50%"> 

<br>

* PCA와 유사하게 품종별로 분류가 잘 되었음을 알 수 있다.   
* 데이터 세트를 스케일링하고 두 클래스 변환을 비교해 본다.   

<br>

<img src ="https://user-images.githubusercontent.com/52434993/91057428-23cc5a80-e662-11ea-8981-dc29754df316.jpg" width = "80%">
<br>

* **두 클래스 변환 모두 SVD 알고리즘으로 구현되었기 때문에, 동일한 변화를 보여준다.**     


-----------------------------------

**차원 축소로 데이터를 잘 설명하는 잠재적인 요소를 추출하는 것에 의미를 둔다.**
