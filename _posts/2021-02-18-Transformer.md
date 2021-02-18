---
layout: post
title:  NLP/Transformer
summary: "Transformer"
author: jiyun
date: '2021-02-18 15:35:23 +0530'
category: AI
thumbnail: /assets/img/posts/nlp.png
keywords: pytorch, AI, Transformer   
permalink: /blog/transformer/
usemathjax: true
---   

>## Transformer
>#### Attention모델이 기본적으로 깔려있다.    
>#### self attention에서 input vector가 자기자신과의 내적으로 인해 output vector 자기자신의 정보를 포함할 경우가 있다.    


- ### self Attention   

<br>

<img src = "https://user-images.githubusercontent.com/52434993/108291140-15a46480-71d5-11eb-8565-db169b437ef8.jpg">

$$
Queries : x_1 \times W^Q 
$$

$$
Keys : (x_1, x_2, x_3) \times W^K
$$

$$
values : (x_1, x_2, x_3) \times W^V
$$


`사진에서 볼 수 있듯이, "i" 에 대한 sequence전체를 고려한 encoding vector를 얻기 위해서, 각각`\
`queries, keys, values의 vector를 이용해서, 가중평균을 얻어낸다.`    

`따라서 모든 embedding vector를 이용해서 다른 정보도 활용을 하는 것이 가능하다.`    

<br>

```
Output : weighted sum of values   

Weight of each value : inner product (query, key) 
```

$$
A(q,K,V) = \sum_{i}\frac{\exp(q\,\cdot\,k_i)}{\sum_{j}\exp(q\,\cdot\,k_j)}v_i
$$

<br>

$$
\frac{\exp(q\,\cdot\,k_i)}{\sum_{j}\exp(q\,\cdot\,k_j)}
$$

$$
i번째 키의 유사도(소프트맥스)\, :\, \frac{i번째 key\,vector와 query의 곱}{각각의 key\,vector와 query의 곱의 합} 
$$


<br>

- ### Multi-Head Attention   

`동일한 sequence에서 다른 측면에서 여러 정보가 필요할 때 사용`

$$
MultiHead(Q,\,K,\,V) = Concat(head_1,\,...\,,\,head_h)W^O 
$$
$$
where \;\; head_i = Attention(QW_{i}^{Q},\,KW_{i}^{K},\,VW_{i}^{V}) 
$$


<br>


`rnn과 달리 순서에 대한 특징을 가져오지 못하기 때문에(?) 각각의 문자의 순서를 구별하게 되는 순서 벡터를 encoding 벡터에 적용한다.`\
`이러한 특징을 Positional Encoding 이라고 한다.`   










