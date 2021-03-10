---
layout: post
title:  Computer Vision/object detection
summary: "Object detection"
author: jiyun
date: '2021-03-10 16:35:23 +0530'
category: AI
thumbnail: /assets/img/posts/CV.jpg
keywords: pytorch, AI, CV  
permalink: /blog/object_detection/
usemathjax: true
---         


>## Object detection   
>#### classification + Bounding box   

<br>

### R-CNN  

<img src="https://user-images.githubusercontent.com/52434993/110566046-cc6f8100-8192-11eb-86eb-cd431cd1623c.jpg" width="780px"> 

각각의 region proposal을 모델에 넣어서 시간이 너무 오래 걸린다.    

또한 region proposal은 고전적인 머신러닝 방법을 따른다.(학습을 통해 성능 향상 힘듬)   

<br>

### Fast R-CNN   

<img src="https://user-images.githubusercontent.com/52434993/110566558-8b2ba100-8193-11eb-845e-fe68798bf70b.jpg">

이미 학습된 CNN을 통과시켜 feature map을 추출한다.   

Selective search통해서 찾은 RoI(Region of Interest)에 RoI pooling을 거친다.(resizing 과정)    

pooling를 거친 벡터를 하나는 softmax(classification), 하나는 bounding box regression을 거친다.   

기존의 R-CNN보다 18배 빠름. (하지만, selective search의 한계)   


<br>


### Faster R-CNN   

<img src="https://user-images.githubusercontent.com/52434993/110567146-64219f00-8194-11eb-8142-3cfe4663e65a.jpg">

최초의 end-to-end 모델   

<br>


- IoU (Intersesction over Union)      

  - 두 영역의 합집합과 교집합으로 두 영역을 판단. (교집합/합집합)   



- Anchor boxes   

  - 미리 정의한 bounding boxes

<br>


- #### RPN(Region Proposal Network)      

<img src="https://user-images.githubusercontent.com/52434993/110567830-76e8a380-8195-11eb-9fa6-eeb3e3f54366.jpg">


CNN을 통해 추출한 feature map을 이용한다.   

3x3 convolution을 256채널만큼 수행한다.(intermediate layer)   

intermediate layer를 **classification**과 **bounding box regression**의 예측값을 계산한다.   

<br>


**classification을 위한 1x1 convolution의 채널 수**    
`(Object vs. Non-object) * (anchor 개수) = 2k`


**bounding box regression을 위한 1x1 convolution의 채널 수**       
`(바운딩 박스 꼭지점 x,y좌표 와 너비 높이 정보) * (anchor 개수) = 4k`


<br>

**NMS(Non-Maximum Suppression)**   

classification을 통해서 높은 점수를 가진 순으로 anchor를 정렬하고, bounding box regression을 적용한다.    

이를 통해 나온 점수를 통해 높은 점수를 가진 box만 남기고, 나머지의 중복되는 box를 제거한다.


<br>

 
```
one-stage detector는 pooling과정이 없다.(모든 영역의 loss를 계산)    

영상의 의미없는 배경으로 인해 class imbalance 문제가 일어난다.   

- Focal loss   
  - class imbalance 문제를 해결하기 위함.
     
  - 잘못된 class 큰 loss를 준다 (갱신하기 위함).   
```

<br>


### DETR   

transformer를 CV영역으로 확장     

따로 anchor를 설정하고, NMS과정을 거치지 않는다.    

보다 end-to-end에 가까운 모습을 보인다.    

하지만 transformer를 거치기 때문에 수행시간은 늘어난다.   


<img src="https://user-images.githubusercontent.com/52434993/110570818-e3fe3800-8199-11eb-93d6-e4b15c5c103d.png" width="780px">
  








