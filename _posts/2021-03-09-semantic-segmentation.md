---
layout: post
title: Computer Vision/image classification
subtitle: "image classification"
categories: AI
tags: [ai]
---
>## GoogleNet   
>#### 구글에서 발표한 Inception계통의 network   
>#### 1x1 Convolution을 통해서 연산량을 줄였다.   

<br>

### 1x1 Convolution   

- Channel 수 조절   

  파라미터의 수가 급격하게 증가한느 것을 예방하기 때문에, channel수를 조절할 수 있다.   
  
<br>


- 계산량 감소   

  channel수를 조절할 수 있기 때문에, 계산량 자체도 감소할 수 있다.   
  1x1 convolution의 채널수를 줄였다가 늘였다 하는 구조는 Bottleneck구조라고 부른다.   
  
<br>

- 비선형성

  파라미터 수가 감소하게 됨에 따라, 모델을 더 깊게 구성할 수 있고,    
  이는 비선형성 활성화 함수를 더 많이 사용할 수 있음을 의미한다.   
  
  즉, 더 구체적인 패턴을 파악할 수 있게 된다.
  
  
<br>


### Auxiliary classifier   

vanishing gradient 문제를 해결하기 위한 방법이다.   

lower layers에 추가적인 gradients를 주입시킨다.   

학습시에만 사용하는 방법이다.   


<img src="https://user-images.githubusercontent.com/52434993/110406298-56511880-80c5-11eb-8432-fef7319f5a0c.png" width="780px">






<br><br>



>## ResNet   
>#### 최초로 깊은 layer를 쌓은 network   
>#### VGGNet보다 8배정도 깊은 층 수.   


<br>

network가 깊어질 수록 정확도가 높아지지 않고 최적화가 안된다.   

이는 degrade rapidly문제이며, overfitting에 의한 문제는 아니였다.(최적화에 의한 문제)    

**Residual block**을 이용해서 그 자신에 대한 identity mapping 뿐만 아니라, **잔차에 대한 학습**을 진행한다.   


<img src="https://user-images.githubusercontent.com/52434993/110406761-150d3880-80c6-11eb-972e-0c771d2b95c9.png">   






<br><br>


>## Semantic segmentation    
>#### 영상 단위가 아닌, 하나의 영상에서 물체를 구분함.   


<br>

### Fully Convolutional Networks(FCN)   

입력에서 부터 출력까지 모두 미분가능한 network 구조이다.(end-to-end)    

사람의 조작없이 학습을 통해 최적화 가능하다.(?)   


<br>

|Fully connected layer|Fully convolutional layer|
|--|--|
|공간 정보 고려 x|입력도 tensor 출력도 tensor|
|fix된 차원으로 출력|1x1 convolution 사용|   
|각 위치마다 채널을 축으로 벡터를 쌓아서 Flattening 적용|각 위치마다 채널을 축으로 필터 개수마다 결과값을 채운다|

<br>

### Upsampling   

영상의 저해상도 문제를 해결하기 위한 기법.   

input data의 크기를 feature map에 의해 줄어든다. 이를 upsampling을 통해 다시 키우게 된다.   


`Upsampling 기법들`

- Unpooling

- **Transposed convolution**   

- **Upsample and convolution**     

<br>


#### Transposed convolution     

<img src="https://user-images.githubusercontent.com/52434993/110412390-7ab1f280-80cf-11eb-9528-506e53a87e32.png" width="780px">   

`kernel 사이즈가 중첩되는 곳이 일부분인데 이렇게 진행해도 되는 것인가??`   

**kernel size와 stride를 잘 조절해서 중첩이 생기지 않게 해야 한다.**     


<br><br>


>## U-Net   
>#### fully convolutional network 이고, skip connections으로 낮은 층과 높은층의 feature을 결합한다.   
>#### **contracting path** 와 **expanding path**로 이루어져 있다.   

<br>

<img src="https://user-images.githubusercontent.com/52434993/110413412-260f7700-80d1-11eb-866b-82c0c9ac89d9.png" width="780px">   


<br>

높은 해상도에서의 **공간상의 구분적 특징**을 expanding path의 **layer에 전달**해줄 수 있다.    

feature sizes와 input size가 홀수라면, **짝수**로 맞춰주게 된다.   
이 과정에서 **해상도의 저하**가 있을수 있다.   











