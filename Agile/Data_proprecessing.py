import copy as cp
import pandas as pd
import Agile_data
import os
from gensim.models import doc2vec
from collections import namedtuple

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# 데이터 다운로드
Agile_data.Lpoint_data_auoto_download()

train_data, train_categorical, train_continuous, train_Segment, train_label = Agile_data.load_train_data()
test_data, test_categorical, test_continuous, test_Segment, test_label, test_label_eval = Agile_data.load_test_data()


# doc2vec 시작
train_docs = [(str(row['item']).split(), row['고객번호']) for idx, row in train_data.iterrows()]
test_docs = [(str(row['item']).split(), row['고객번호']) for idx, row in test_data.iterrows()]

TaggedDocument = namedtuple('TaggedDocument', 'words tags')

tagged_train_docs = [TaggedDocument(d, [c]) for d, c in train_docs]
tagged_test_docs = [TaggedDocument(d, [c]) for d, c in test_docs]

if True:
    
    doc_vectorizer = doc2vec.Doc2Vec(

    dm = 0,            # PV-DBOW
    dbow_words = 1,    # w2v simultaneous with DBOW d2v / default 0
    window = 8,        # distance between the predicted word and context words 
    size = 300,        # vector size 
    sample = 1e-5,     # threshold for configuring
    min_count = 3,     # ignore with freq lower
    negative = 10,     # negative sampling / default 5

    seed = 2017,
    workers = 1,       # single cpu -> reproducible 
    alpha = 0.05,      # learning-rate
    min_alpha = 0.05   # min learning-rate

    )
    
    doc_vectorizer.build_vocab(tagged_train_docs)
    print(str(doc_vectorizer))
    
    for epoch in range(20):
        doc_vectorizer.train(tagged_train_docs)
        doc_vectorizer.alpha -= 0.002
        doc_vectorizer.min_alpha = doc_vectorizer.alpha
    
    print("Doc2Vec 학습 완료.")

save_dir = 'D2V_model/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# To save
doc_vectorizer.save('./D2V_model/Item2Vec.model')
# To load
doc_vectorizer=doc2vec.Doc2Vec.load('./D2V_model/Item2Vec.model')


# 학습된 모델에서 벡터를 추론

train_item = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_train_docs]
train_item_pd = pd.DataFrame(train_item)

test_item = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_test_docs]
test_item_pd = pd.DataFrame(test_item)



def cross_columns(data, cross_colnames):
    # 교차 특성을 생성
    del_col = data.columns
    for i in cross_colnames:
        tmp = 0
        for j in i:
            tmp += 1
            if tmp == 1:
                cross_data = data[j]
                columns = j
            else:
                cross_data = cross_data + data[j]
                columns = columns + "+" + j
        data[columns] = cross_data
        
    data.drop(del_col, inplace=True, axis=1)
        
    return data

# 상위 카테고리들의 모든 2쌍의 조합 만들기
cross_col = [[train_Segment.columns[i], train_Segment.columns[i + j + 1]]
             for i in range(16) for j in range(16-i)]


# 데이터 타입이 {1, 0} int 인데 AND 방식으로 만들기 위해 str 으로 만듬
train_Segment_ = cp.deepcopy(train_Segment)
test_Segment_ = cp.deepcopy(test_Segment)
train_Segment_str = train_Segment_.astype(str)
test_Segment_str = test_Segment_.astype(str)

Segment = pd.concat([train_Segment_str, test_Segment_str], axis=0)

# 교차
train_Segment_cross = cross_columns(train_Segment_str, cross_col)
test_Segment_cross = cross_columns(test_Segment_str, cross_col)

# 더미화
train_Segment_cross_d = pd.get_dummies(train_Segment_cross)
test_Segment_cross_d = pd.get_dummies(test_Segment_cross)

# 문자의 숫자화
train_categorical["성별"] = le.fit_transform(train_categorical["성별"])
train_categorical["연령대"] = le.fit_transform(train_categorical["연령대"])
train_categorical["A거주지역"] = le.fit_transform(train_categorical["A거주지역"])

train_user_feartures = pd.concat([train_continuous, train_categorical], axis=1)


test_categorical["성별"] = le.fit_transform(test_categorical["성별"])
test_categorical["연령대"] = le.fit_transform(test_categorical["연령대"])
test_categorical["A거주지역"] = le.fit_transform(test_categorical["A거주지역"])

test_user_feartures = pd.concat([test_continuous, test_categorical], axis=1)


# 위에서 핸들링한 모든 데이터  column bind
Wide_data = pd.concat([train_item_pd, train_Segment_cross_d], axis=1)
Wide_data_test = pd.concat([test_item_pd, test_Segment_cross_d], axis=1)


Deep_data = pd.concat([train_item_pd, train_Segment, train_user_feartures], axis=1)
Deep_data_test = pd.concat([test_item_pd, test_Segment, test_user_feartures], axis=1)


print("Wide & Deep 데이터를 저장합니다.")

Wide_data.to_csv("./Lpoint_data/Wide_data.csv", encoding='cp949', index=False)
Deep_data.to_csv("./Lpoint_data/Deep_data.csv", encoding='cp949', index=False)
Wide_data_test.to_csv("./Lpoint_data/Wide_data_test.csv", encoding='cp949', index=False)
Deep_data_test.to_csv("./Lpoint_data/Deep_data_test.csv", encoding='cp949', index=False)


