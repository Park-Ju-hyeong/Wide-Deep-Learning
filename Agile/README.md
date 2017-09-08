해당 문서는
----------

Wide & Deep Model for Recommendation 문서입니다.

* 상품 구매정보 데이터를 핸들링하는 과정입니다.
[[Wide&Deep] Data proprecessing]()  

* 정제된 데이터를 Wide & Deep 모형에 넣고 상품 추천을 합니다.
[[Wide&Deep] Recommendation]()

---

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