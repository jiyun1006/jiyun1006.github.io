---
layout: post
title: NLP/Transformer
subtitle: "Transformer"
categories: AI
tags: [ai]
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

- ### Scaled dot-product Attention   

`Q(queries), K(keys), V(values)를 이용해서 가중평균을 구한다.`   

`Q벡터와 K벡터에 대해서 attention score를 구하고, V벡터를 가중합하여 context vector를 구한다.`   


$$
score(q,\,k)\;=\; q\,\cdot\,k/ \sqrt{n} 
$$

<br>

- #### "i am a student" 에 대한 scaled dot-product Attention 과정 예시

<br>

`"i" 에 대한 Q벡터 기준, score(q,k)의 과정을 보여준다.`   

`이후 각각의 attention score에 활성함수를 적용하여 attention 분포를 구하고 각각의 V벡터와 가중합하여 attention value를 구한다.`\
`이 attention value가 단어 "i"에 대한 context vector이다.`   


<img src="https://user-images.githubusercontent.com/52434993/108466557-28976180-72c7-11eb-84c4-74cac32cc170.jpg" width="780px">


*해당 연산을 각 단어에 대해 벡터연산으로 나누지 말고, 행렬 연산을 이용해서 일괄 계산을 한다.*   


$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$



<br>

- ### Multi-Head Attention   

`동일한 sequence에서 다른 측면에서 여러 정보가 필요할 때 사용`   

`여러 번의 attention을 병렬로 사용하는 방법 (head개의 병렬 attention 수행)`

$$
MultiHead(Q,\,K,\,V) = Concat(head_1,\,...\,,\,head_h)W^O 
$$
$$
where \;\; head_i = Attention(QW_{i}^{Q},\,KW_{i}^{K},\,VW_{i}^{V}) 
$$


<br>


`rnn과 달리 순서에 대한 특징을 가져오지 못하기 때문에(?) 각각의 문자의 순서를 구별하게 되는 순서 벡터를 encoding 벡터에 적용한다.`\
`이러한 특징을 Positional Encoding 이라고 한다.`   

<img src = "https://user-images.githubusercontent.com/52434993/108445426-bb240a80-729f-11eb-889a-0d50f8f35a5e.png" width="780px">   



<br><br>

>## Multihead attention 구현 실습   

<br>

*import 및 데이터 전처리 과정 생략*    


- ### Linear transformation & 여러 head로 나누기   

`embedding 벡터를 linear transfor시켜줄 matrix 생성`   

```python
--- 각각 query, key, value를 위한 linear ---
w_q = nn.Linear(d_model, d_model)
w_k = nn.Linear(d_model, d_model)
w_v = nn.Linear(d_model, d_model)


--- 마지막 attention value를 최종적으로 합치기 위한 linear ---
w_0 = nn.Linear(d_model, d_model)


--- 선형변환 적용 ---
q = w_q(batch_emb)  # (B, L, d_model)
k = w_k(batch_emb)  # (B, L, d_model)
v = w_v(batch_emb)  # (B, L, d_model)
```     

<br>

`multihead attention 이기 때문에, num_head의 개수에 따라 차원을 분할시켜 여러 vector로 생성`   

```python
batch_size = q.shape[0]
d_k = d_model // num_heads

q = q.view(batch_size, -1, num_heads, d_k)  # (B, L, num_heads, d_k)
k = k.view(batch_size, -1, num_heads, d_k)  # (B, L, num_heads, d_k)
v = v.view(batch_size, -1, num_heads, d_k)  # (B, L, num_heads, d_k)


--- 각 head가 L x d_k 개의 행렬을 가지게 되는 꼴로 만들어 준다. ---
q = q.transpose(1, 2)  # (B, num_heads, L, d_k) 
k = k.transpose(1, 2)  # (B, num_heads, L, d_k)
v = v.transpose(1, 2)  # (B, num_heads, L, d_k)
```   


<br>

- ### Scaled dot-product self-attention    


`각 head에서 self-attention을 실행한다.`   

```python
attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B, num_heads, L, L)
attn_dists = F.softmax(attn_scores, dim=-1)  # (B, num_heads, L, L)  
```   

`구해진 attention 분포를 통해 가중합을 구한다.`   

```python
attn_values = torch.matmul(attn_dists, v)
```


`각각 head의 결과물들을 concat하고 linear-transformation과정을 거친다.`   

`불연속적인 주소들을 연속적인 메모리 주소로 재설정하기 위해서 contiguous를 사용한다.`   


```python
attn_values = attn_values.transpose(1, 2)  # (B, L, num_heads, d_k)
attn_values = attn_values.contiguous().view(batch_size, -1, d_model)  # (B, L, d_model)


outputs = w_0(attn_values)
```   


<br><br>

>## Masked Multihead attention 구현 실습    


`위의 전처리과정과 동일`   

- ### Masking 적용   

`padding 처리된 벡터에 attention을 하는 것이 불필요한 과정이기 때문에, 이를 masking한다.`    


```python
--- pad_id 와 같은 곳에 False를 주는 tensor를 만든다. ---
padding_mask = (batch != pad_id).unsqueeze(1)  # (B, 1, L)


--- torch.tril를 이용해서 삼각형 모양의 Boolean 벡터를 만든다 (반은 True, 반은 False).
nopeak_mask = torch.ones([1, max_len, max_len], dtype=torch.bool)  # (1, L, L)
nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L)


--- padding_mask 와 nopeak_mask의 and연산을 통해 padding에 masking을 적용한다. ---
mask = padding_mask & nopeak_mask  # (B, L, L)
```

<br>

- ### Masking 적용된 self-attention   


`masking된 곳에 매우 작은 수를 줌으로써 attention score를 계산하고 나서, softmax함수를 취했을 대, 0이 나오게 한다.`   


```python
masks = mask.unsqueeze(1)  # (B, 1, L, L)


# masked_fill_ : False 라고 나온곳은 매우 작은 수를 준다.
masked_attn_scores = attn_scores.masked_fill_(masks == False, -1 * inf)  # (B, num_heads, L, L)
```






