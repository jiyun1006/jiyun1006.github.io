---
layout: post
title:  NLP/NLP Preprocessing
summary: "NLP Preprocessing"
author: jiyun
date: '2021-02-15 14:35:23 +0530'
category: NLP
thumbnail: /assets/img/posts/nlp.png
keywords: pytorch, AI, Bag-of_words    
permalink: /blog/NLP_preprocessing/
usemathjax: true
---


>## Bag-of-Words   

- 중복되지 않은(unique) 단어를 저장 한다.(categorical variable 범주형 자료)     

- 원-핫 인코딩을 이용해서 벡터를 나타낸다. (단어의 의미에 상관없이 벡터로 나타낸다.)   


<br><br>

>## Naive bayes classification   
>#### Bag-of-Words를 분류할 수 있는 기법.   

<br>

*특정 카테고리 c가 고정되었을 때, 문서 d가 나타날 확률*   


$$
P(d|c)P(c) = P(w_1, w_2, ... , w_n | c)P(c) \rightarrow  P(c)\prod_{w_i\in W}P(w_i|c) 
$$   


<br><br>

>## Word Embedding   
>#### 각 단어들을 특정한 차원으로 이루어진 공간상의 점, 벡터로 변환하는 것.      
>#### 비슷한 의미를 가진 단어를 비슷한 공간의 점에 매핑되도록 한다. (의미상 유사도를 고려)   

<br>

>### Word2Vec   
>#### 같은 문장에서의 인접한 단어들이 관련성이 높다 라는 개념 사용.   

<br>


<img src="https://user-images.githubusercontent.com/52434993/107907750-d9d08b80-6f97-11eb-9681-e024208c5c40.jpg">


*Word2Vec과정을 시각화해서 볼 수 있는 곳*   <a href="http://ronxin.github.io/wevi/">[클릭]</a>

<br><br>

>### GloVe   
>#### 입/출력 단어 쌍을 학습데이터가 한 윈도우내에서 몇 번 등장했는지 사전에 계산   

<br>

*중복되는 단어에 대해 더 잘 대응할 수 있다.*   


$$J(\theta) = \frac{1}{2}\sum_{i,j = 1}^{W}f(P_i_j) ({u_i}^{T}v_j - \log{P_i_j})^{2}$$   



<img src= "https://user-images.githubusercontent.com/52434993/107919289-68500780-6fae-11eb-84ce-54485433254b.gif">




