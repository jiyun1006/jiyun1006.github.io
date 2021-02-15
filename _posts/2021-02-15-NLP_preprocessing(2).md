---
layout: post
title:  NLP/NLP Preprocessing(2)
summary: "NLP Preprocessing - Word2Vec"
author: jiyun
date: '2021-02-15 14:35:23 +0530'
category: NLP
thumbnail: /assets/img/posts/nlp.png
keywords: pytorch, AI, Word2Vec, Glove   
permalink: /blog/NLP_preprocessing(2)/
usemathjax: true
---


>## Word Embedding   
>#### 각 단어들을 특정한 차원으로 이루어진 공간상의 점, 벡터로 변환하는 것.      
>#### 비슷한 의미를 가진 단어를 비슷한 공간의 점에 매핑되도록 한다. (의미상 유사도를 고려)   

<br>

>### Word2Vec   
>#### 같은 문장에서의 인접한 단어들이 관련성이 높다 라는 개념 사용.   

<br>


<img src="https://user-images.githubusercontent.com/52434993/107907750-d9d08b80-6f97-11eb-9681-e024208c5c40.jpg">


*Word2Vec과정을 시각화해서 볼 수 있는 곳*   <a href="http://ronxin.github.io/wevi/">[클릭]</a>   


- CBOW (Continuous Bag of Words)       


`학습시킬 모든 단어를 one-hot-encoding 방식으로 벡터화 한다.`   

`주변 단어를 이용해서 중심 단어를 예측 (window size 이용)`      


<img src = "https://user-images.githubusercontent.com/52434993/107971673-1d59e280-6ff6-11eb-93a0-b9bf0533b3e9.png" width = "700px">




<br>


- Skip-Gram   

`하나의 단어에서 여러 단어를 예측하는 방식.`   

`CBOW와 마찬가지로 one-hot-encoding을 이용해서 단어를 벡터화한다.`

<img src = "https://user-images.githubusercontent.com/52434993/107972231-e33d1080-6ff6-11eb-8e78-6bdc4fd755f8.png" width = "700px">


<br><br>

>### GloVe   
>#### 입/출력 단어 쌍을 학습데이터가 한 윈도우내에서 몇 번 등장했는지 사전에 계산   

<br>

*중복되는 단어에 대해 더 잘 대응할 수 있다.*   


$$ J(\theta) = \frac{1}{2}\sum_{i,j = 1}^{W}f(P_{ij})(u_{i}^{T}v_{j} - \log{P_{ij}})^{2} $$   


<br><br>
