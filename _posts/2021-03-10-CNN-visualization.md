---
layout: post
title: Computer Vision/cnn visualization
subtitle: "CNN visualization"
categories: AI
tags: [ai]
---
>## CNN visualization   
>#### black box를 시각화하여, 어떻게 수행되고 있는지 어떻게 성능을 높일지를 생각할 수 있다.    


<br>


### class visualization - Gradient ascent   


영상 데이터를 넣었을 때, CNN 모델을 거쳐서 출력된 점수 중 가장 높은 결과물을 고른다.     

이후, regularization term을 빼게 된다. 


#### Loss   

$$
I^{*} = arg\underset{I}max\, f(I) - Reg(I)
$$

$$
Reg(I) \,=\, \lambda||I||_{2}^{2} 
$$

 
<br>

임의의 영상에 대한 prediction score를 구한다.   

이후에, score를 높이는 방향으로 역전파를 통해 입력값을 개선한다.    

영상을 update하고, 과정을 반복한다.    

<br><br>


### Model decision explanation   

모델이 특정 입력을 받았을 때, 입력을 해석하는 방법.     

<br>

- #### Saliency test     

  영상이 특정 영역의 중요도를 추출하는 방법.   
 
  - Occlusion map   
    임의의 영상에 mask를 넣어서 object를 식별할 score를 구한다.   
    위치에 따라서 변하는 score를 통해서 위치의 중요도를 확인할 수 있다.   
  <br>
  
  - via Backpropagation    
    특정 이미지의 classification 이후, class 점수를 활용.   
    
    입력 영상을 넣고, class score를 구한다.   
    이를 역전파를 통해 입력 데이터까지 계산한다. (부호보다는 절대적인 크기가 중요하기에 제곱을 이용한다.)   
    
    
  <br>
  
  - backpropagation-based saliency   
    일반적으로 forward path에서 활성함수를 이용해서 masking을 하지만,   
    gradient 자체에 masking을 실행한다.(deconvolution)   
    
    $$
    h^{l+1} = max(0,h^l) 
    $$
    
    
    기존의 backward pass (forward path에서의 패턴을 사용)    
    $$
    \frac{\partial L}{\partial h^l} = [(h^l > 0)]\frac{\partial L}{\partial h^{l+1}}
    $$
    
    
    deconvolution backward pass       
    $$
    \frac{\partial L}{\partial h^l} = [(h^{l+1} > 0)]\frac{\partial L}{\partial h^{l+1}}
    $$
    
    
    guided backpropagation (기존과 deconvolution 방식을 합친 것.)   
    $$
    \frac{\partial L}{\partial h^l} = [(h^l > 0) \& (h^{l+1} > 0)]\frac{\partial L}{\partial h^{l+1}}
    $$
    
   
    
 <br>
 
- #### CAM(Class activation mapping)   
  
  <img src="https://user-images.githubusercontent.com/52434993/110584145-1b78de80-81b2-11eb-87d0-a066746e15ef.jpg" width="780px">
  
  - feature map을 FC layer 전에 GAP(Global average pooling)에 통과시킨다.    
    

    $$
    S_c \, = \,  \sum_{k}w_{k}^{c}F_k \; = \; \sum_{k}w_{k}^c\, \sum_{(x,y)}f_k(x,y) \; = \; \sum_{(x,y)} \sum_{k}w_{k}^{c}f_k(x,y)   
    $$
  
  - 뒷 부분의 sumation은 공간에 대한 정보가 남아있다. (GAP 부분을 적용하기 전)   
  
  - CAM을 적용하기 위해서는 마지막 layer가 FC layer이어야 한다.   

<br>


- #### Grad-CAM   
  
  - GAP이 없는 모델에도 사용이 가능하고, 기존 학습된 network를 재학습하거나 수정할 필요가 없다.
  
  - 입력 영상까지 역전파를 하지 않고, 활성함수 전까지만 진행한다.   

    $$
    \alpha_{k}^{c} = \frac{1}{Z}\sum_i \sum_j\, \frac{\partial y^c}{\partial A_{ij}^k}
    $$
    
    $$
    L_{Grad-CAM}^{c} = ReLU(\sum_{k} \alpha_{k}^{c}A^{k})
    $$   
    
    
    **guided Backprop과 Grad-CAM을 결합한 모델**    
    
    
    
    <img src="https://user-images.githubusercontent.com/52434993/110586007-d73b0d80-81b4-11eb-9384-b569d54c2d7c.jpg" width="780px">

