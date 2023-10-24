---
layout: post
title: Computer Vision
subtitle: "Computer Vision"
categories: AI
tags: [ai]
---


>## Computer vision(Image Classification)   

<br>

- ### Overview

**visual perception**은 흔히 오감이라고 불리는 인간의 시각 능력이다.   

이를 컴퓨터에 비교하게 된다면 input data를 시각적으로 본 사물로 볼 수 있다.   

하지만, 사람의 시각능력은 불완전하다. 기존의 사물을 생각하며, 사물을 바라보기 때문에 **편향성이 존재**하기 때문이다.   

이러한 특징으로 과거의 고전적인 머신러닝은 데이터를 받고 직접 데이터에 대한 특징을 검사했다. (feature extracting)   

하지만 딥러닝에서는 데이터를 받고, 이에 대한 데이터 검사와 분류 과정을 모두 신경망이 처리하도록 한다.(**end to end**)   


<br>


#### 어떻게 분류할 수 있을까?
모든 데이터가 다 있다면, 그에 기반해서 분류하면 된다.(k Nearest Neighbors)   

하지만 모든 데이터를 저장하는 것이 현실적으로 힘들다.   

<br>

#### CNN (Fully Connected network)
서로 다른 데이터에 다른 가중치   
이미지의 일부분만 넣어주게 된다면, 학습데이터와 다른 데이터로 잘못된 예측을 하게 된다.


#### CNN (locally connected network)
하나의 데이터를 지역적으로 특징을 뽑는다.
parameter를 공유하기 때문에, 더 적은 수의 parameter를 쓸 수 있다.   


<br><br>

>## Data augmentation   

<br>

input data에 다양한 변환 기법을 적용하여, 모델을 학습시킨다.   


- Brightness adjustment   

- Ratete   

- flip   

- Crop   

- Affine transformation   

- cutmix   


### RandAugment   
 
위에서 나열한 다양한 변환 기법들을 임의로 지정하게 해서 최적의 변환 기법을 선택하게 한다.   

이렇게 지정된 기법들을 Policy라 칭하게 되고, policy를 적용한 학습이 더 높은 정확도를 보이게 된다.    


<br>


### Transfer learning   

한 데이터 셋에서 배운 지식을 다른 데이터 셋에 사용한다.   

<br>

 
- #### transfer knowledge from a pre-trained task to a new task      

미리 학습된 10dim의 데이터 셋을 가지고 FC layers만 업데이트하면서, 새로운 결과를 만드는 방식.

<br>

- #### Fine-tuning the whole model   

기존의 transfer learning에서 Convolution layer도 같이 학습한다.   

단, Convolution layer는 낮은 learning rate를 가지고, FC layer는 높은 learning rate를 가진다.   


<br>


### Knowledge distillation   

이미 학습된 teacher network의 모델을 작은 모델에 지식을 전달하는 방법.   


<img src = "https://user-images.githubusercontent.com/52434993/110266823-fc3b4f00-8001-11eb-9b34-147acecd2316.png" width ="780px">


- Hard label   
  - 일반적으로 원-핫 벡터로 생각하면 된다.

- Hard Prediction(Normal Softmax)   
  
$$
\frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$


- Soft label
  - 소프트맥스 함수를 거쳐 나온 값이라고 생

- Soft Prediction(Softmax with temperature)   

$$
\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$


<br>

- Distillation Loss   
  - KL-div(Soft label, Soft prediction)   
  - Loss = difference between the teacher and student network's inference

- Student Loss   
  - CrossEntropy(Hard label, Soft prediction)   
  - Loss = difference between the student network's inference and true label 




 

 
