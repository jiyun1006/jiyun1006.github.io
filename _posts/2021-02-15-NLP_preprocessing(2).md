---
layout: post
title: NLP/NLP Preprocessing(2)
subtitle: "NLP Preprocessing - Word2Vec"
categories: AI
tags: [ai]
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



>## Word2Vec 실습   
>#### word2vec의 두 가지 모델 CBOW, Skip-gram모델을 구현한다.   
>#### 실행환경 : colab

<br>


- ### 필요 패키지 설치 및 import    

```python
!pip install konlpy

from tqdm import tqdm
from konlpy.tag import Okt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

import torch
import copy
import numpy as np
```   

<br>



- ### 데이터 전처리    

*데이터 생성*   

```
train_data = [
  "정말 맛있습니다. 추천합니다.",
  "기대했던 것보단 별로였네요.",
  "다 좋은데 가격이 너무 비싸서 다시 가고 싶다는 생각이 안 드네요.",
  "완전 최고입니다! 재방문 의사 있습니다.",
  "음식도 서비스도 다 만족스러웠습니다.",
  "위생 상태가 좀 별로였습니다. 좀 더 개선되기를 바랍니다.",
  "맛도 좋았고 직원분들 서비스도 너무 친절했습니다.",
  "기념일에 방문했는데 음식도 분위기도 서비스도 다 좋았습니다.",
  "전반적으로 음식이 너무 짰습니다. 저는 별로였네요.",
  "위생에 조금 더 신경 썼으면 좋겠습니다. 조금 불쾌했습니다."       
]

test_words = ["음식", "맛", "서비스", "위생", "가격"]
```    

*토큰화 및 빈도수로 저장*   

```python
tokenizer = Okt() 

def make_tokenized(data):
  tokenized = []
  for sent in tqdm(data):
    tokens = tokenizer.morphs(sent, stem=True)  #morphs 형태소 추출
    tokenized.append(tokens)

  return tokenized
  
train_tokenized = make_tokenized(train_data)


word_count = defaultdict(int)

for tokens in tqdm(train_tokenized):
  for token in tokens:
    word_count[token] += 1
    
word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)


# 빈도수가 가장 높은 것 순으로 저장.
w2i = {}
for pair in tqdm(word_count):
  if pair[0] not in w2i:
    w2i[pair[0]] = len(w2i)
```   

<br>
 
- CBOW와 Skip-Gram 데이터 클래스 정의   
  - CBOW : 주변 단어를 이용해서 중심 단어를 예측.   
  - Skip-Gram : 중심 단어를 이용해서 주변 단어를 예측.



```python
# CBOW
class CBOWDataset(Dataset):
  def __init__(self, train_tokenized, window_size=2):
    self.x = []
    self.y = []

    for tokens in tqdm(train_tokenized):
      token_ids = [w2i[token] for token in tokens]
      for i, id in enumerate(token_ids):
        if i-window_size >= 0 and i+window_size < len(token_ids): 
          self.x.append(token_ids[i-window_size:i] + token_ids[i+1:i+window_size+1])
          self.y.append(id)

    self.x = torch.LongTensor(self.x)  # (전체 데이터 개수, 2 * window_size)
    self.y = torch.LongTensor(self.y)  # (전체 데이터 개수)

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]
    


#Skip-Gram
class SkipGramDataset(Dataset):
  def __init__(self, train_tokenized, window_size=2):
    self.x = []
    self.y = []

    for tokens in tqdm(train_tokenized):
      token_ids = [w2i[token] for token in tokens]
      for i, id in enumerate(token_ids):
        if i-window_size >= 0 and i+window_size < len(token_ids):
          self.y += (token_ids[i-window_size:i] + token_ids[i+1:i+window_size+1])
          self.x += [id] * 2 * window_size

    self.x = torch.LongTensor(self.x)  # (전체 데이터 개수)
    self.y = torch.LongTensor(self.y)  # (전체 데이터 개수)

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]


```    

<br>


- 모델 학습 및 테스트   

```python
#CBOW
class CBOW(nn.Module):
  def __init__(self, vocab_size, dim):
    super(CBOW, self).__init__()
    self.embedding = nn.Embedding(vocab_size, dim, sparse=True)
    self.linear = nn.Linear(dim, vocab_size)

  # B: batch size, W: window size, d_w: word embedding size, V: vocab size
  def forward(self, x):  # x: (B, 2W)
    embeddings = self.embedding(x)  # (B, 2W, d_w)
    embeddings = torch.sum(embeddings, dim=1)  # (B, d_w)
    output = self.linear(embeddings)  # (B, V)
    return output




#Skip-Gram
class SkipGram(nn.Module):
  def __init__(self, vocab_size, dim):
    super(SkipGram, self).__init__()
    self.embedding = nn.Embedding(vocab_size, dim, sparse=True)
    self.linear = nn.Linear(dim, vocab_size)

  # B: batch size, W: window size, d_w: word embedding size, V: vocab size
  def forward(self, x): # x: (B)
    embeddings = self.embedding(x)  # (B, d_w)
    output = self.linear(embeddings)  # (B, V)
    return output
```   

*객체 설정 및 하이퍼 파라미터 설정*   

```python
cbow = CBOW(vocab_size=len(w2i), dim=256)
skipgram = SkipGram(vocab_size=len(w2i), dim=256)


batch_size=4
learning_rate = 5e-4
num_epochs = 5
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cbow_loader = DataLoader(cbow_set, batch_size=batch_size)
skipgram_loader = DataLoader(skipgram_set, batch_size=batch_size)
```

*CBOW 학습*

```python
cbow.train()
cbow = cbow.to(device)
optim = torch.optim.SGD(cbow.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

for e in range(1, num_epochs+1):
  print("#" * 50)
  print(f"Epoch: {e}")
  for batch in tqdm(cbow_loader):
    x, y = batch
    x, y = x.to(device), y.to(device) # (B, W), (B)
    output = cbow(x)  # (B, V)
 
    optim.zero_grad()
    loss = loss_function(output, y)
    loss.backward()
    optim.step()

    print(f"Train loss: {loss.item()}")

print("Finished.")
```

*Skip-Gram 학습*   
```python
skipgram.train()
skipgram = skipgram.to(device)
optim = torch.optim.SGD(skipgram.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

for e in range(1, num_epochs+1):
  print("#" * 50)
  print(f"Epoch: {e}")
  for batch in tqdm(skipgram_loader):
    x, y = batch
    x, y = x.to(device), y.to(device) # (B, W), (B)
    output = skipgram(x)  # (B, V)

    optim.zero_grad()
    loss = loss_function(output, y)
    loss.backward()
    optim.step()

    print(f"Train loss: {loss.item()}")

print("Finished.")
```  

<br><br>

- 테스트   

<br>

```python
for word in test_words:
  input_id = torch.LongTensor([w2i[word]]).to(device)
  emb = cbow.embedding(input_id)

  print(f"Word: {word}")
  print(emb.squeeze(0))

```   

*테스트 결과*   

```
Word: 음식
tensor(2.8897, device='cuda:0', grad_fn=<UnbindBackward>)
Word: 맛
tensor(3.1442, device='cuda:0', grad_fn=<UnbindBackward>)
Word: 서비스
tensor(2.4952, device='cuda:0', grad_fn=<UnbindBackward>)
Word: 위생
tensor(2.8184, device='cuda:0', grad_fn=<UnbindBackward>)
Word: 가격
tensor(2.9439, device='cuda:0', grad_fn=<UnbindBackward>)
```



