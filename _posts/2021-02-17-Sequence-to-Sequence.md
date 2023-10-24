---
layout: post
title: NLP/Sequence to Sequence
subtitle: "Sequence to Sequence 모델"
categories: AI
tags: [ai]
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


<br><br>

>## Seq2Seq 모델 구현 실습   
>#### Encoder, Decoder 를 구현하고 이를 이용해서 seq2seq모델을 구현한다.   

<br>


- ### 패키지 import 및 데이터 전처리   

```python
from tqdm import tqdm
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import randomon
```   

<br>
 
*데이터 설정 및 전처리*

```python
vocab_size = 100
pad_id = 0
sos_id = 1 # start token
eos_id = 2 # end token

src_data = [
  [3, 77, 56, 26, 3, 55, 12, 36, 31],
  [58, 20, 65, 46, 26, 10, 76, 44],
  [58, 17, 8],
  [59],
  [29, 3, 52, 74, 73, 51, 39, 75, 19],
  [41, 55, 77, 21, 52, 92, 97, 69, 54, 14, 93],
  [39, 47, 96, 68, 55, 16, 90, 45, 89, 84, 19, 22, 32, 99, 5],
  [75, 34, 17, 3, 86, 88],
  [63, 39, 5, 35, 67, 56, 68, 89, 55, 66],
  [12, 40, 69, 39, 49]
]

trg_data = [
  [75, 13, 22, 77, 89, 21, 13, 86, 95],
  [79, 14, 91, 41, 32, 79, 88, 34, 8, 68, 32, 77, 58, 7, 9, 87],
  [85, 8, 50, 30],
  [47, 30],
  [8, 85, 87, 77, 47, 21, 23, 98, 83, 4, 47, 97, 40, 43, 70, 8, 65, 71, 69, 88],
  [32, 37, 31, 77, 38, 93, 45, 74, 47, 54, 31, 18],
  [37, 14, 49, 24, 93, 37, 54, 51, 39, 84],
  [16, 98, 68, 57, 55, 46, 66, 85, 18],
  [20, 70, 14, 6, 58, 90, 30, 17, 91, 18, 90],
  [37, 93, 98, 13, 45, 28, 89, 72, 70]
]
```

```python
trg_data = [[sos_id]+seq+[eos_id] for seq in tqdm(trg_data)] # target data에 start, end token 추가

# 패딩 추가 함수
def padding(data, is_src=True):
  max_len = len(max(data, key=len))
  print(f"Maximum sequence length: {max_len}")

  valid_lens = []
  for i, seq in enumerate(tqdm(data)):
    valid_lens.append(len(seq))
    if len(seq) < max_len:
      data[i] = seq + [pad_id] * (max_len - len(seq))

  return data, valid_lens, max_len
  
  
#패딩 추가 적용.
src_data, src_lens, src_max_len = padding(src_data)
trg_data, trg_lens, trg_max_len = padding(trg_data)

# B: batch size, S_L: source maximum sequence length, T_L: target maximum sequence length
src_batch = torch.LongTensor(src_data)  # (B, S_L)
src_batch_lens = torch.LongTensor(src_lens)  # (B)
trg_batch = torch.LongTensor(trg_data)  # (B, T_L)
trg_batch_lens = torch.LongTensor(trg_lens)  # (B)
```   

<br>

*이전의 PackedSequence를 사용*     

```python
# packedSequence사용을 위해 정렬
src_batch_lens, sorted_idx = src_batch_lens.sort(descending=True)
src_batch = src_batch[sorted_idx]
trg_batch = trg_batch[sorted_idx]
trg_batch_lens = trg_batch_lens[sorted_idx]
```   

<br>

- ### Encoder 구현   

*파라미터 설정*   

```python
embedding_size = 256
hidden_size = 512
num_layers = 2
num_dirs = 2
dropout = 0.1  
```   

*Encoder 클래스*   


`nn.Embedding을 학습시키고, encoder 역할을 할 gru를 먼저 설정한다.`   

`gru 모델에 만들어 둔, 미리 설정한 packed_input과 h_0를 넣어준다.`   

`forward의 마지막 hidden state vector와 backward의 마지막 hidden state vector를 합쳐서 새로운 hidden state를 만든다.`   

```python
class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()

    self.embedding = nn.Embedding(vocab_size, embedding_size)
    
    # encoder 역할을 하는 Bi-GRU.
    self.gru = nn.GRU(
        input_size=embedding_size, 
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=True if num_dirs > 1 else False,
        dropout=dropout
    )
    
    # decoder와의 크기를 맞추기 위한 작업
    self.linear = nn.Linear(num_dirs * hidden_size, hidden_size)

  def forward(self, batch, batch_lens):  # batch: (B, S_L), batch_lens: (B)
    # d_w: word embedding size
    batch_emb = self.embedding(batch)  # (B, S_L, d_w)
    batch_emb = batch_emb.transpose(0, 1)  # (S_L, B, d_w)

    packed_input = pack_padded_sequence(batch_emb, batch_lens)

    h_0 = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (num_layers*num_dirs, B, d_h) = (4, B, d_h)
    
    # gru 모델에 만들어 둔, packed_input과 h_0를 넣어준다.
    packed_outputs, h_n = self.gru(packed_input, h_0)  # h_n: (4, B, d_h)
    outputs = pad_packed_sequence(packed_outputs)[0]  # outputs: (S_L, B, 2d_h)

    forward_hidden = h_n[-2, :, :]
    backward_hidden = h_n[-1, :, :]
    
    #forward 관점에서의 마지막 hidden state vector와 backward 관점에서의 마지막 hidden state vector를 합친다.   
    hidden = self.linear(torch.cat((forward_hidden, backward_hidden), dim=-1)).unsqueeze(0)  # (1, B, d_h)

    return outputs, hidden
    
 
 encoder = Encoder()
```   

<br>

- ### Decoder 구현   

`각 time step의 hidden state Vector를 구한다.`

```python
class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()

    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.gru = nn.GRU(
        input_size=embedding_size, 
        hidden_size=hidden_size,
    )
    
    # 각 time step의 hidden state vector
    self.output_layer = nn.Linear(hidden_size, vocab_size)

  def forward(self, batch, hidden):  # batch: (B), hidden: (1, B, d_h)
    batch_emb = self.embedding(batch)  # (B, d_w)
    batch_emb = batch_emb.unsqueeze(0)  # (1, B, d_w)

    outputs, hidden = self.gru(batch_emb, hidden)  # outputs: (1, B, d_h), hidden: (1, B, d_h)
    
    # V: vocab size
    outputs = self.output_layer(outputs)  # (1, B, V)

    return outputs.squeeze(0), hidden
    
 decoder = Decoder()
```

<br><br>

- ### Seq2Seq 모델   

`encoder에 src_batch, src_batch_lens를 넣어서 마지막 hidden state vector를 만든다.`   

`for 문을 통해 <SoS> 토큰부터 시작해서 길이 1짜리 hidden state vector를 만든다.`   

`현재 time step에서의 예측 배치 텐서를 top_ids에 저장한다.`   

`현재의 top_ids는 특정 teacher_forching_prob를 사용해서 사용할지 말지를 결정한다.`   

```python 
class Seq2seq(nn.Module):
  def __init__(self, encoder, decoder):
    super(Seq2seq, self).__init__()

    self.encoder = encoder
    self.decoder = decoder

  def forward(self, src_batch, src_batch_lens, trg_batch, teacher_forcing_prob=0.5):
    # src_batch: (B, S_L), src_batch_lens: (B), trg_batch: (B, T_L)

    _, hidden = self.encoder(src_batch, src_batch_lens)  # hidden: (1, B, d_h)

    input_ids = trg_batch[:, 0]  # (B)
    batch_size = src_batch.shape[0]
    outputs = torch.zeros(trg_max_len, batch_size, vocab_size)  # (T_L, B, V)

    for t in range(1, trg_max_len):
      decoder_outputs, hidden = self.decoder(input_ids, hidden)  # decoder_outputs: (B, V), hidden: (1, B, d_h)

      outputs[t] = decoder_outputs
      _, top_ids = torch.max(decoder_outputs, dim=-1)  # top_ids: (B)

      input_ids = trg_batch[:, t] if random.random() > teacher_forcing_prob else top_ids

    return outputs
    
 seq2seq = Seq2seq(encoder, decoder)
```    

<br>

- ### Seq2Seq 모델 사용   

`src_batch와 trg_batch를 seq2seq모델에 넣어본다.`   

`모델에 적용 후 output의 형태는 max_length, batch_size, vocab_size 이다.`

```python
outputs = seq2seq(src_batch, src_batch_lens, trg_batch)

print(outputs.shape)


--- outputs.shape ---
# 차례대로 max_length, batch_size, vocab_size 이다.
torch.Size([22, 10, 100])
```   
<br>

<img src="https://user-images.githubusercontent.com/52434993/108151169-74a4a380-7119-11eb-84e7-71de01cd5676.jpg" width="780px">

`decoder를 통해 만든 결과에서 마지막 <EoS>토큰을 예측한 "?"항목은 필요가 없으므로 제거하고,`\
`실제 trg_batch에서는 앞의 <SoS>토큰을 제거한다.`   

`loss값을 구한다음, 역전파 연산을 해주고 업데이트를 하면서 seq2seq모델을 학습시킨다.`   


```python
loss_function = nn.CrossEntropyLoss()

preds = outputs[:-1, :, :].transpose(0, 1)  # (B, T_L-1, V)
loss = loss_function(preds.contiguous().view(-1, vocab_size), trg_batch[:,1:].contiguous().view(-1, 1).squeeze(1))
```   

<br><br>


>## Seq2Seq Attention 실습   

<br>

`위에서 진행했던 데이터전처리(데이터, packedSequence)는 모두 동일하다.`   

`Encoder를 구현하는 부분부터 차이점이 생긴다.`   

- ### Encoder 구현   

`hidden state vector뿐만 아니라, 기존의 output도 linear를 적용한다.`   

```python
class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()

    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.gru = nn.GRU(
        input_size=embedding_size, 
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=True if num_dirs > 1 else False,
        dropout=dropout
    )
    self.linear = nn.Linear(num_dirs * hidden_size, hidden_size)

  def forward(self, batch, batch_lens):  # batch: (B, S_L), batch_lens: (B)
    # d_w: word embedding size
    batch_emb = self.embedding(batch)  # (B, S_L, d_w)
    batch_emb = batch_emb.transpose(0, 1)  # (S_L, B, d_w)

    packed_input = pack_padded_sequence(batch_emb, batch_lens)

    h_0 = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (num_layers*num_dirs, B, d_h) = (4, B, d_h)
    packed_outputs, h_n = self.gru(packed_input, h_0)  # h_n: (4, B, d_h)
    outputs = pad_packed_sequence(packed_outputs)[0]  # outputs: (S_L, B, 2d_h)
    # 차원을 줄이고, non linear 인 tanh를 거친다.
    outputs = torch.tanh(self.linear(outputs))  # (S_L, B, d_h)

    forward_hidden = h_n[-2, :, :]
    backward_hidden = h_n[-1, :, :]
    hidden = torch.tanh(self.linear(torch.cat((forward_hidden, backward_hidden), dim=-1))).unsqueeze(0)  # (1, B, d_h)

    return outputs, hidden
    
encoder = Encoder()
```   

<br>

- ### Dot-product Attention 구현   

`decoder hidden state와 encoder hidden state간의 내적을 구해 유사도를 구한 다음, 해당 유사도를 소프트맥스 함수를 적용한다.`   

`그 후, 가중치를 구해서 encoder hidden state 가중합하여 Attention value를 끌어낸다.`   

```python
class DotAttention(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, decoder_hidden, encoder_outputs):  # (1, B, d_h), (S_L, B, d_h)
    query = decoder_hidden.squeeze(0)  # (B, d_h)
    key = encoder_outputs.transpose(0, 1)  # (B, S_L, d_h)

    # 내적 --> 유사도 (key와 query)
    # 각 encoder hidden state의 길이만큼 반복적으로 각 차원을 곱하고, 더한다.
    energy = torch.sum(torch.mul(key, query.unsqueeze(1)), dim=-1)  # (B, S_L)

    # 내적(유사도)를 소프트맥스 함수에 적용한다.
    attn_scores = F.softmax(energy, dim=-1)  # (B, S_L)
    
    # 그 후에 가중치를 구한다.
    # 각 encoder hidden state에 적용되는 가중치로 나타내기 위해 해당 연산을 한다.
    attn_values = torch.sum(torch.mul(encoder_outputs.transpose(0, 1), attn_scores.unsqueeze(2)), dim=1)  # (B, d_h)

    return attn_values, attn_scores
    
dot_attn = DotAttention()
```   

<br>


- ### Decoder 구현
  - 기본적인 seq2seq의 decoder에다가 Attention 을 추가.       


```python
class Decoder(nn.Module):
  def __init__(self, attention):
    super().__init__()

    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.attention = attention
    self.rnn = nn.GRU(
        embedding_size,
        hidden_size
    )
    self.output_linear = nn.Linear(2*hidden_size, vocab_size)

  def forward(self, batch, encoder_outputs, hidden):  # batch: (B), encoder_outputs: (L, B, d_h), hidden: (1, B, d_h)  
    batch_emb = self.embedding(batch)  # (B, d_w)
    batch_emb = batch_emb.unsqueeze(0)  # (1, B, d_w)

    outputs, hidden = self.rnn(batch_emb, hidden)  # (1, B, d_h), (1, B, d_h)

    attn_values, attn_scores = self.attention(hidden, encoder_outputs)  # (B, d_h), (B, S_L)
    concat_outputs = torch.cat((outputs, attn_values.unsqueeze(0)), dim=-1)  # (1, B, 2d_h)

    return self.output_linear(concat_outputs).squeeze(0), hidden  # (B, V), (1, B, d_h)
    
decoder = Decoder(dot_attn)
```   

<br>

- ### Seq2Seq 모델 구현   
  - decoder에 attention을 넣은 모델   


```python
class Seq2seq(nn.Module):
  def __init__(self, encoder, decoder):
    super(Seq2seq, self).__init__()

    self.encoder = encoder
    self.decoder = decoder

  def forward(self, src_batch, src_batch_lens, trg_batch, teacher_forcing_prob=0.5):
    # src_batch: (B, S_L), src_batch_lens: (B), trg_batch: (B, T_L)

    encoder_outputs, hidden = self.encoder(src_batch, src_batch_lens)  # encoder_outputs: (S_L, B, d_h), hidden: (1, B, d_h)

    input_ids = trg_batch[:, 0]  # (B)
    batch_size = src_batch.shape[0]
    outputs = torch.zeros(trg_max_len, batch_size, vocab_size)  # (T_L, B, V)

    for t in range(1, trg_max_len):
      decoder_outputs, hidden = self.decoder(input_ids, encoder_outputs, hidden)  # decoder_outputs: (B, V), hidden: (1, B, d_h)

      outputs[t] = decoder_outputs
      _, top_ids = torch.max(decoder_outputs, dim=-1)  # top_ids: (B)

      input_ids = trg_batch[:, t] if random.random() > teacher_forcing_prob else top_ids

    return outputs
    
seq2seq = Seq2seq(encoder, decoder)
```

<br><br>


> ## Bahdanau Attention   
> #### 해당 시점의 decode hidden vector와 전체 encode hidden vector를 내적하는 것이 아니다.   
> #### concat을 통해서 특정 layer를 통해 score를 계산하는 방식    


$$s_{t-1}와 모든 h_1 , ... , h_{T_z}사이의 연관성을 가중치로 보고, 이 가중치의 합을 구해서 Context \, vector를 구한다. $$

<br>

$$
Context\,vector : c_t = \sum_{j=1}^{T_z}a_{tj}h_j = Ha_t
$$

$$
Attention\,score : a_t = Softmax((Score(s_{t-1}, h_j))_{j=1}^{T_z}) \in \mathbb{R}^T_z
$$

$$
Similarity(유사도) : Score(s_{t-1}, h_j) =  \upsilon^Ttanh(W_as_{t-1} + U_aH_j)
$$

<br>

- ### Encoder 구현   

```python
class ConcatAttention(nn.Module):
  def __init__(self):
    super().__init__()

    self.w = nn.Linear(2*hidden_size, hidden_size, bias=False)
    self.v = nn.Linear(hidden_size, 1, bias=False)

  def forward(self, decoder_hidden, encoder_outputs):  # (1, B, d_h), (S_L, B, d_h)
    src_max_len = encoder_outputs.shape[0]

    decoder_hidden = decoder_hidden.transpose(0, 1).repeat(1, src_max_len, 1)  # (B, S_L, d_h)
    encoder_outputs = encoder_outputs.transpose(0, 1)  # (B, S_L, d_h)

    concat_hiddens = torch.cat((decoder_hidden, encoder_outputs), dim=2)  # (B, S_L, 2d_h)
    energy = torch.tanh(self.w(concat_hiddens))  # (B, S_L, d_h)

    attn_scores = F.softmax(self.v(energy), dim=1)  # (B, S_L, 1)
    attn_values = torch.sum(torch.mul(encoder_outputs, attn_scores), dim=1)  # (B, d_h)

    return attn_values, attn_scores
```   

<br>

- ### Decoder 구현   
  - embedding 과 attention을 수행해서 attention value를 embedding과 concat을 하여, input_size가 embedding_size + hidden_size가 된다.   


```python
class Decoder(nn.Module):
  def __init__(self, attention):
    super().__init__()

    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.attention = attention
    self.rnn = nn.GRU(
        embedding_size + hidden_size,
        hidden_size
    )
    self.output_linear = nn.Linear(hidden_size, vocab_size)

  def forward(self, batch, encoder_outputs, hidden):  # batch: (B), encoder_outputs: (S_L, B, d_h), hidden: (1, B, d_h)  
    batch_emb = self.embedding(batch)  # (B, d_w)
    batch_emb = batch_emb.unsqueeze(0)  # (1, B, d_w)

    attn_values, attn_scores = self.attention(hidden, encoder_outputs)  # (B, d_h), (B, S_L)

    concat_emb = torch.cat((batch_emb, attn_values.unsqueeze(0)), dim=-1)  # (1, B, d_w+d_h)

    outputs, hidden = self.rnn(concat_emb, hidden)  # (1, B, d_h), (1, B, d_h)

    return self.output_linear(outputs).squeeze(0), hidden  # (B, V), (1, B, d_h)

```




