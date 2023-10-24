---
layout: post
title: NLP/NLP Preprocessing(1)
subtitle: "NLP Preprocessing - Bag-of-words"
categories: AI
tags: [ai]
---

>## Bag-of-Words   

- 중복되지 않은(unique) 단어를 저장 한다.(categorical variable 범주형 자료)     

- 원-핫 인코딩을 이용해서 벡터를 나타낸다. (단어의 의미에 상관없이 벡터로 나타낸다.)   


<br><br>

>## Naive bayes classification   
>#### Bag-of-Words를 분류할 수 있는 기법.   

<br>

- Naive bayes classification 역시 각 단어가 독립이라 가정하기 때문에, 본질적으로 Bag-of-words와 같다.(순서 무시, 빈도만 생각)      

<br>

*특정 카테고리 c가 고정되었을 때, 문서 d가 나타날 확률*   


$$
P(d|c)P(c) = P(w_1, w_2, ... , w_n | c)P(c) \rightarrow  P(c)\prod_{w_i\in W}P(w_i|c) 
$$   


<br><br>




>## naive_bayes 코드 실습    
>#### 주어진 데이터를 전처리하고, NaiveBayes 분류기 모델을 구현한다.    
>#### 실행환경 : colab

<br>

-  ### 기본적으로 필요한 패키지를 설치, import   

```python
!pip install konlpy

from tqdm import tqdm


from konlpy import tag 
from collections import defaultdict
import math
```   

<br>

- ### 데이터 전처리   
  - 학습 데이터 테스트 데이터의 전처리를 진행한다. (class는 긍정(1), 부정(0)이 있다.)   


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
train_labels = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0]

test_data = [
  "정말 좋았습니다. 또 가고 싶네요.",
  "별로였습니다. 되도록 가지 마세요.",
  "다른 분들께도 추천드릴 수 있을 만큼 만족했습니다.",
  "서비스가 좀 더 개선되었으면 좋겠습니다. 기분이 좀 나빴습니다."
]
```

<br>

*KoNLPy 패키지에서 Okt(Open Korea Text)를 선택한다. (형태소 분석기)*   

```python
tokenizer = tag.Okt()

def make_tokenized(data):
  tokenized = []  # 단어 단위로 나뉜 리뷰 데이터.

  for sent in tqdm(data):
    tokens = tokenizer.morphs(sent)  # 형태소 추출 (morphs 메서드)
    tokenized.append(tokens)

  return tokenized
  
  
# 학습 데이터, 테스트 데이터 토큰화
train_tokenized = make_tokenized(train_data)
test_tokenized = make_tokenized(test_data)
```   

*토큰화된 학습 데이터의 첫번째 값*      

```
['정말', '맛있습니다', '.', '추천', '합니다', '.']
```

<br>


- ### 학습데이터를 기준으로 가장 많이 등장한 단어부터 순서대로 vocab에 추가.    



```python
word_count = defaultdict(int)  # Key: 단어, Value: 등장 횟수

for tokens in tqdm(train_tokenized):
  for token in tokens:
    word_count[token] += 1
       
word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
```

*나온 횟수가 가장 많은 단어 순으로 정리*   

```python
w2i = {}  # Key: 단어, Value: 단어의 index
for pair in tqdm(word_count):
  if pair[0] not in w2i:
    w2i[pair[0]] = len(w2i)
```

<br>


- ### NaiveBayes Classifier 모델 클래스 구현.   


$$
\underset{c\in C}{\arg\max} P(c)\prod_{x\in X}P(x|c)
$$   

*클래스 c에서 x가 등장할 확률을 구해야 한다.*   

$$
\frac{count(x_i, c_j)}{\sum_{x\in V}count(x_i, c_j)}
$$   


<br>

**likelihoods 에 저장.  {key : token, value : [key : class, value : likelihood(각 토큰이 클래스에 나타날 확률)]}**      


<br>


**특정 단어가 특정 클래스에 등장하지 않을 확률에 대한 대비**
**(Laplace Smoothing)**   

<br>

*`self.k` : Smoothing을 위한 상수*   
*`self.w2i` : 사전에 구한 vocab*   
*`self.priors` : 각 class의 prior 확률*   
*`self.likelihoods` : 각 token의 특정 class 조건 내에서의 likelihood*   


```python

class NaiveBayesClassifier():
  def __init__(self, w2i, k=0.1):
    self.k = k
    self.w2i = w2i
    self.priors = {}
    self.likelihoods = {}

  def train(self, train_tokenized, train_labels):
    self.set_priors(train_labels)  # Priors 계산.
    self.set_likelihoods(train_tokenized, train_labels)  # Likelihoods 계산.

  def inference(self, tokens):
    log_prob0 = 0.0
    log_prob1 = 0.0

    for token in tokens:
      if token in self.likelihoods:  # 학습 당시 추가했던 단어에 대해서만 고려.
        log_prob0 += math.log(self.likelihoods[token][0])
        log_prob1 += math.log(self.likelihoods[token][1])

    # 마지막에 prior를 고려.
    log_prob0 += math.log(self.priors[0])
    log_prob1 += math.log(self.priors[1])

    if log_prob0 >= log_prob1:
      return 0
    else:
      return 1

  def set_priors(self, train_labels):
    class_counts = defaultdict(int)
    for label in tqdm(train_labels):
      class_counts[label] += 1
    
    for label, count in class_counts.items():
      self.priors[label] = class_counts[label] / len(train_labels)   # 실제 카운트되는 label 개수 / 전체 label 개수 

    # 특정 데이터가 클래스에서 등장할 확률      

  def set_likelihoods(self, train_tokenized, train_labels):
    token_dists = {}  # 각 단어의 특정 class 조건 하에서의 등장 횟수.
    class_counts = defaultdict(int)  # 특정 class에서 등장한 모든 단어의 등장 횟수.

    for i, label in enumerate(tqdm(train_labels)):
      count = 0
      for token in train_tokenized[i]:
        if token in self.w2i:  # 학습 데이터로 구축한 vocab에 있는 token만 고려.
          if token not in token_dists:
            token_dists[token] = {0:0, 1:0}
          token_dists[token][label] += 1
          count += 1
      class_counts[label] += count

    for token, dist in tqdm(token_dists.items()):
      if token not in self.likelihoods:
        self.likelihoods[token] = {
            0:(token_dists[token][0] + self.k) / (class_counts[0] + len(self.w2i)*self.k),
            1:(token_dists[token][1] + self.k) / (class_counts[1] + len(self.w2i)*self.k),
        }
```

<br><br>

- ### 모델 객체를 만들고 학습 및 테스트   

```python
classifier = NaiveBayesClassifier(w2i)
classifier.train(train_tokenized, train_labels)

preds = []
for test_tokens in tqdm(test_tokenized):
  pred = classifier.inference(test_tokens)
  preds.append(pred)
```


