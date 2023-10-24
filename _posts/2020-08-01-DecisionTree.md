---
layout: post
title: ML_guide/DecisionTree
subtitle: "DecisionTree"
categories: ML_guide
tags: [ai]
---
## 분류 - 결정 트리

> ## 결정 트리   

* 머신러닝 알고리즘 중 가장 직관적이고 이해하기 쉬운 알고리즘.   
* **'균일도'라는 룰을 기반으로 이루어진 알고리즘이다.**      
* Graphviz 패키지로 트리를 시각화할 수 있다.

<img src="https://user-images.githubusercontent.com/52434993/89186969-89837480-d5d7-11ea-8776-65a1399f3ed7.jpg" width = "80%">   

*특별한 파라미터 없이 시각화한 결정트리*   

*데이터 세트에서의 피처별로 중요도의 값 시각화 (iris 데이터 이용)*   

**[코드]**

```
print("Feature importances : \n{0}".format(np.round(dt_clf.feature_importances_,3)))

for name, value in zip(iris_data.feature_names, dt_clf.feature_importances_):
    print('{0} : {1:.3f}'.format(name,value))
    
sns.barplot(x=dt_clf.feature_importances_, y = iris_data.feature_names)
```   

**[결과]**   

```
sepal length (cm) : 0.025
sepal width (cm) : 0.000
petal length (cm) : 0.555
petal width (cm) : 0.420
```   
*petal length 가 중요도가 가장 높은 것을 알 수 있다.*      

<img src ="https://user-images.githubusercontent.com/52434993/89197941-6cef3880-d5e7-11ea-8714-8d46ad85bcb2.jpg" width = "80%">

<br><br>


> ## 결정 트리 과적합 (overfitting)   

* train데이터에 최적화를 시키다보면 train데이터의 특성과 조금만 다른 데이터 세트를 예측할 때 **정확도가 크게 떨어진다.**   
* scikit-learn의 make_classification() 함수로 임의의 데이터 세트를 만들고 시험해 본다.   

**[코드]**   
```
X_features, y_labels = make_classification(n_features=2, n_redundant = 0, n_informative=2, n_classes = 3, n_clusters_per_class = 1,
                                          random_state = 0)

plt.scatter(X_features[:,0], X_features[:,1], marker='o', c=y_labels, s = 25, edgecolor='k')
```    

**[결과]**   

<img src="https://user-images.githubusercontent.com/52434993/89187765-ac625880-d5d8-11ea-9872-dbd251a3a04c.jpg" width = "80%">


* 다른 하이퍼 파라미터의 조작없이 디폴트 값으로 결정트리를 학습하고 시각화 한다.   

**[코드]**   
```
dt_clf = DecisionTreeClassifier().fit(X_features,y_labels)
visualize_boundary(dt_clf, X_features, y_labels)
```   
*visualize_boundary() 함수는 시각화 하는 함수*   

**[결과]**   
<img src ="https://user-images.githubusercontent.com/52434993/89187995-006d3d00-d5d9-11ea-898d-2fda0823e8d2.jpg" width = "70%">   

* 일부의 이상치 데이터까지 분류하기 위해서 경계가 많아졌다.   
* 트리의 깊이가 깊고 복잡하여 정확도가 떨어지고, 다른 형태의 데이터 세트 예측에 적합하지 않다.   
* 따라서 리프노드의 최소 데이터 개수를 정하고 학습시킨다.   

**[코드]**   
```
dt_clf = DecisionTreeClassifier(min_samples_leaf=6).fit(X_features, y_labels)
visualize_boundary(dt_clf, X_features, y_labels)
```   

**[결과]**   

<img src = "https://user-images.githubusercontent.com/52434993/89188316-72de1d00-d5d9-11ea-9f9e-209b52243bdd.jpg" width ="70%">   

* 조금 더 일반화 된 규칙으로 분류가 되었음을 알 수 있다.   
**하이퍼 파라미터를 찾으며 최적의 정확도 수치를 찾아내는 작업으로 학습을 하면 테스트 데이터 세트 예측의 정확도가 좋아진다...**    





