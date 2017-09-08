해당 문서는
----------

Wide & Deep Model for Recommendation 문서입니다.

* 상품 구매정보 데이터를 핸들링하는 과정입니다.  

[[Wide&Deep] Data proprecessing.ipynb](https://github.com/Park-Ju-hyeong/Wide-Deep-Learning/blob/master/Agile/%5BWide%26Deep%5D%20Data%20proprecessing.ipynb)  
[Data_proprecessing.py](https://github.com/Park-Ju-hyeong/Wide-Deep-Learning/blob/master/Agile/Data_proprecessing.py)

* 정제된 데이터를 Wide & Deep 모형에 넣고 상품 추천을 합니다.  

[[Wide&Deep] Recommendation.ipynb](https://github.com/Park-Ju-hyeong/Wide-Deep-Learning/blob/master/Agile/%5BWide%26Deep%5D%20Recommendation.ipynb)
[Recommendation.py](https://github.com/Park-Ju-hyeong/Wide-Deep-Learning/blob/master/Agile/Recommendation.py)


---
## 작동

우선적으로 데이터 전처리 과정인 `Data_proprecessing` 파일을 먼저 실행시켜야 합니다. `Agile_data`모듈을 통해 데이터가 자동으로 다운로드 되어 단순히 `Run` 하면 됩니다.  

혹시 형식의 데이터를 가지고있으신 분들은 제가 보여드리는 형식으로 핸들링해서 집어넣으시면 됩니다.

이후 `Recommendation`파일을 통해 전처리된 데이터를 Wide & Deep 모형을 실행합니다. Test 데이터를 통해 Hit-rate(precision)이 출력됩니다.

* `Lpoint_data` `WnD_model` `D2V_model` 폴더가 자동으로 생성됩니다.
* 제가 돌린 Doc2Vec과 Tensorflow 모델은 `ckpt` 이곳에 있습니다. 
---

## OS requirements

```
Linux 계열
UBUNTU
```
> 윈도우에서 `urllib` 패키지가 정상작동하지 않는 것 같습니다.

## Install requirements

```
pip install tensorflow-gpu # cpu 버전도 상관 없음 
pip install gensim==0.13.4.1
```
또는 

```
pip install -r requirements.txt
```

## Run test recommendation

```
python Data_proprecessing.py # 전처리 먼저 실행
python Recommendation.py # 정확도를 보여줌
```

## Reference Implementations

[[1] Recommender System with Distributed Representation](https://www.slideshare.net/rakutentech/recommender-system-with-distributed-representation)   
[[2] Distributed Representations of Sentences and Documents](https://arxiv.org/abs/1405.4053)   
[[3] Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)   
[[4] TensorFlow Wide & Deep Learning Tutorial](https://www.tensorflow.org/tutorials/wide_and_deep)   
[[5] Korean NLP in Python](http://konlpy.org/en/v0.4.4/)