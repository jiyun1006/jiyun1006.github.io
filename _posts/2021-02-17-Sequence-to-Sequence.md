---
layout: post
title:  NLP/Sequence to Sequence
summary: "Sequence to Sequence 모델"
author: jiyun
date: '2021-02-17 15:35:23 +0530'
category: AI
thumbnail: /assets/img/posts/nlp.png
keywords: pytorch, AI, Sequence-to-Sequence   
permalink: /blog/seq2seq/
usemathjax: true
---   

>## Sequence-to-Sequence   
>#### RNN의 구조 중 many to many의 구조. (입력도 sequence 출력도 sequence)   
>#### encoder와 decoder는 서로 파라미터를 공유하지 않는다.   

<br>

- encoder의 마지막 timestep의 hidden state vector는 decoder의 h_0의 역할을 한다.    

- decoder에서의 문장 생성시에, 첫 번째 단어로 `<SoS>` 토큰을 넣어준다. (Start of Sentence) \
  또한 마지막에는 `<EoS>` 토큰을 넣어준다. (End of Sentence)    
  
<br>
  
>### Seq2Seq model with Attention   
>#### Seq-to-Seq 모델에서 순차적으로 단어의 정보를 매 time step마다 축적해가면서 hidden state vector를 생성하는 과정   
>#### 많은 Sequence를 거치면서 정보가 손실될 수 있다. 때문에 각각의 hidden state vector를 decoder에 넘겨줘서 정보 손실을 최소화 한다.   

<br>

<img src ="https://user-images.githubusercontent.com/52434993/108142492-399a7400-7109-11eb-8dd9-22bc8f1022f7.jpg" width="780px">   

- encoder의 hidden state vector 와 decoder의 hidden state vector의 내적으로 attention output(context vector)을 구해낸다. \
  (파란색이 encoder, 빨간색이 decoder)     


*Teacher forcing : 잘못된 예측을 막는 것.*




