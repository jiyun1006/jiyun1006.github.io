>## Word Embedding   
>#### 각 단어들을 특정한 차원으로 이루어진 공간상의 점, 벡터로 변환하는 것.      
>#### 비슷한 의미를 가진 단어를 비슷한 공간의 점에 매핑되도록 한다. (의미상 유사도를 고려)   

<br>

>### Word2Vec   
>#### 같은 문장에서의 인접한 단어들이 관련성이 높다 라는 개념 사용.   

<br>


<img src="https://user-images.githubusercontent.com/52434993/107907750-d9d08b80-6f97-11eb-9681-e024208c5c40.jpg">


*Word2Vec과정을 시각화해서 볼 수 있는 곳*   <a href="http://ronxin.github.io/wevi/">[클릭]</a>

<br><br>

>### GloVe   
>#### 입/출력 단어 쌍을 학습데이터가 한 윈도우내에서 몇 번 등장했는지 사전에 계산   

<br>

*중복되는 단어에 대해 더 잘 대응할 수 있다.*   


$$ J(\theta) = \frac{1}{2}\sum_{i,j = 1}^{W}f(P_{ij})(u_{i}^{T}v_{j} - \log{P_{ij}})^{2} $$   


<br><br>
