---
layout: post
title: NLP/Beam Serach and BLEU
subtitle: "Beam Search and BLEU score"
categories: AI
tags: [ai]
---

>## Beam search   
>#### 바로 다음 단어만을 예측하는 task를 greedy decoding라고 부른다. (되돌아 갈 수가 없다...)      
>#### 이러한 greedy decoding의 단점을 해소(?)하는 방법   


<br>


`매 time step 마다 하나의 단어만을 고려하는 greedy decoding과 모든 조합을 고려하는 Exhaustive search알고리즘의 사이에 있는 알고리즘`   

$$
score(y_1,\,...\,,y_t) = logP_{LM}(y_1,\,...\,,y_t|x) = \sum_{i=1}^{t}logP_{LM}(y_i|y_1,\,...\,,y_{i-1},x)
$$   

<br>


`모든 경우의 수를 다 따지지는 않지만, 시간적 비용면에서 훨씬 효율적이다.`   


`서로 다른 경로가 존재하고 다른 시점에서 <END>토큰이 생성되기 때문에, 토큰이 생성될 때 해당 hypothesis를 중단한다.`   


`모든 hypothesis가 종료되면, 가장 높은 score값을 가진 hypothesis를 예측값으로 줄 수 있다.`   


`매 단어를 생성할 때 마다 확률값이 작아진다. (step마다 확률값이 음수이기 때문에 계속해서 감소한다.)`\
`따라서 이로 인한 오류를 없게하기 위해서 Normalize(by length) 적용한다.`   


$$
score(y_1,\,...\,,y_t) = \frac{1}{t}\sum_{i=1}^{t}logP_{LM}(y_i|y_1,\,...\,,y_{i-1},x)
$$   




<img src ="https://user-images.githubusercontent.com/52434993/108162615-cefd2e80-7130-11eb-8190-f6ab3f7f0397.jpg" width="780px">   



<br><br>


>## BLEU score   


`일반적인 평가 지표`   


$$
precision = \frac{(\#correct \; words)}{length\_of\_reference} 
$$   

$$
recall = \frac{(\#correct \; words)}{length\_of\_reference}
$$


$$
F\, - \,measure = \frac{precision\, \times \,recall}{\frac{1}{2}(precision\, + \,recall)} 
$$

<br>

`순서에 대한 검증은 하지 못하기 때문에 해당 평가 방법이 아닌 BLUE score를 이용한다.`    

`BLEU에서는 precision만 고려한다.`  

`N-gram에 대한 기하평균`   


$$
BLEU = min(1, \frac{length\_of\_prediction}{length\_of\_reference})(\prod_{i=1}^{4}precision_i)^{\frac{1}{4}}
$$




