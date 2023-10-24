---
layout: post
title: ML_guide/Recommendation
subtitle: "Recommendation 협업 필터링"
categories: AI
tags: [ai]
---


>## 추천 시스템의 유형   

추천 시스템의 초창기에는 콘텐츠 기반이나 최근접 이웃 기반 필터링이 사용되었지만, 최근에는 잠재 요인 협업 필터링 방 

- 콘텐츠 기반 필터링   
  `특정한 아이템을 선홀할 때, 비슷한 아이템을 추천하는 방식`

- 협업 필터링   
  `새로운 영화가 나왔을 때, 친구들에게 물어보는 것과 같이 사용자 행동양식만을 기반으로 추천을 하는 것이 협업 필터링이다.`   
  
  `사용자-아이템 평점 행렬 데이터를 사용.`   
  
  
  - 최근접 이웃 협업 필터링   
  
  - 잠재 요인 협업 필터링   

<img src="https://user-images.githubusercontent.com/52434993/108805939-adde8700-75e3-11eb-92b2-1fca106bc1bc.jpeg" width="780px">   


*행렬분해를 위해 SVD방식을 이용하지만, NaN값이 있다면 적용할 수 없다. 따라서 SGD를 이용해서 SVD를 수행한다.*   


<br><br>


>## 확률적 경사 하강법(SGD)   
>#### P와 Q행렬로 계산된 예측 R 행렬 값이 실제 R 행렬 값과 가장 최소의 오류를 가질 수 있도록 하는 반복적인 비용함수 최적화를 통해 P와 Q를 유추한다.   

- P와 Q를 임의값을 가진 행렬로 설정   

- P와 Q.T값을 곱해 예측 R행렬을 계산 후, 실제 값과의 차이 계산   

- 해당 오류값을 최소화하도록 업데이트   


$$
min\sum(r_{(u,i)}\,-\,p_uq_{i}^{t})^2\;+\; \lambda(||q_i||^2\,+\,||p_u||^2)
$$


<br><br>

