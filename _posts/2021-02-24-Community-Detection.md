---
layout: post
title:  Recommendation/군집(community) 탐색
summary: "군집(community) 탐색"
author: jiyun
date: '2021-02-24 15:35:23 +0530'
category: AI
thumbnail: /assets/img/posts/recommendation.jpeg
keywords: pytorch, AI, recommendation, 군집(community)  
permalink: /blog/community/
usemathjax: true
---       


>## 군집 탐색   
>#### 그래프를 여러 군집으로 나누는 문제를 군집 탐색 문제라고 한다.   

<br>

- 배치모형 (configuration model)   

  `주어진 그래프에 대한 배치 모형은, 각 정점의 연결성을 보존한 상태에서 간선들을 무작위로 재배치하여 얻은 그래프이다.`   


- 군집성 (Modularity)   

  `군집 탐색의 성공여부를 판단하기 위해 척도.`
  
  `그래프와 군집들의 집합 S가 있다고 가정할 때, 각 군집들이 군집의 성질을 잘 만족하는지를 살펴야 한다.`   
  
  `배치 모형과 비교했을 때, 그래프에서 군집 내부 간선의 수가 월등히 많을수록 성공한 군집 탐색이다.`   
  
  $$
  -1\;\leq\;\frac{1}{2|E|}\sum_{s \in S}(그래프에서 군집\,s\,내부간선의 수 \; - \; 배치 모형에서 군집 \,s\, 내부 간선의 수의 기대값)\; \leq \; 1
  $$


<br><br>


>## Girvan-Newman 알고리즘   
>#### 대표적인 하향식 군집 탐색 알고리즘이다.
>#### 전체 그래프에서 탐색을 시작하며, 군집들이 서로 분리되도록 간선을 순차적으로 제거한다. (다른 군집을 연결하는 다리 역할의 간선을 제거한다.)   


<br>


- ### 매개 중심성(Betweenness Centrality)   

`해당 간선이 정점 간의 최단 경로에 놓이는 횟수`    

`모든 정점에서 각 정점의 최단 경로 중 특정 간선을 포함한 비율`   

`매개 중심성이 높은 간선일수록, 서로 다른 군집을 연결하는 다리 역할을 한다.`   

`간선을 제거하면서, 다시 매개 중심성을 계산한다.`   

<br>

$$
정점 i로 부터 j로의 최단 경로 수를 \sigma_{i,j}라고 하고   
그 중 간선 (x,y)를 포함한 것을 \sigma_{i,j}(x,y)라고 한다.   
이 때의, 간선 (x,y)의 매개 중심성은 아래의 수식과 같다.   
$$

$$
\sum_{i<j}\frac{\sigma_{i,j}(x,y)}{\sigma_{i,j}} 
$$   


<br>


