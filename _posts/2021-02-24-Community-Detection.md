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




