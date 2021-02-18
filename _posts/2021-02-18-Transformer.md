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

*Output : weighted sum of values*   

*Weight of each value : inner product (query, key)*   


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


i번째 키값을 query  query와 key의 각각의 내적을 구한 후의 합  --> i번째 키값의 소프트맥스 후의 유사도 ---> 이것들의 합 * value




