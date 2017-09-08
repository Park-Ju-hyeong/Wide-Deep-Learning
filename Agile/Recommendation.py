import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import Agile_data
import tensorflow as tf
from tensorflow.contrib import layers
from datetime import datetime
import os

Wide_data, Deep_data,train_label = Agile_data.load_Wide_Deep_train_data()
Wide_data_test, Deep_data_test, test_label = Agile_data.load_Wide_Deep_test_data()

learning_rate = 1e-4
training_epochs = 10
batch_size = 256

wide_size = len(Wide_data.columns)
deep_size = len(Deep_data.columns) 
deep_embed_size = 2  # 임베딩할 컬럼 개수

output_size = len(train_label.columns)

He = tf.contrib.layers.variance_scaling_initializer()

# input place holders
tf.reset_default_graph()
Wide = tf.placeholder(tf.float32, [None, wide_size])
Deep_un_embed = tf.placeholder(tf.float32, [None, deep_size - deep_embed_size])
Deep_embed = tf.placeholder(tf.int32, [None, deep_embed_size])

Y = tf.placeholder(tf.float32, [None, output_size])
train_mode = tf.placeholder(tf.bool, name='train_mode')


def get_acc(score_matrix, top_n, test_matix):
    avg_acc = 0
    for i in range(len(score_matrix)):
        top = score_matrix.iloc[i].nlargest(top_n).index
        tmp = 0
        for j in range(len(top)):
            if top[j] in test_matix["item"][i].split():
                tmp += 1

        acc = tmp / len(top)
        avg_acc += acc / len(score_matrix)

    return avg_acc

no_class_max = 42
embedding_size = 5
Deep_col = Deep_data.columns



# `embedding_lookup` 함수를 이용해서 categorical 변수를 Embedding 합니다.
deep_embed_var = tf.Variable(tf.random_uniform([no_class_max, embedding_size], -1.0, 1.0))
deep_embed = tf.nn.embedding_lookup(deep_embed_var, Deep_embed)
deep_embed_re = tf.reshape(deep_embed, [-1, embedding_size * deep_embed_size])

Deep = tf.concat([Deep_un_embed, deep_embed_re], axis=1)


# 원래 Deep learning에 들어갈 input의 column 수는 333 개 였는데
# 그 중 2개의 categorical 변수를 각각 5차원으로 Embedding하면 333 - 2 + 10 = 341 이 됩니다.


A = tf.get_variable("A", shape=[wide_size, output_size])
b = tf.Variable(tf.random_normal([output_size]))

wide = tf.add(tf.matmul(Wide, A), b)


W1 = tf.get_variable("W1", shape=[341, 512], initializer=He)
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(Deep, W1) + b1)

W2 = tf.get_variable("W2", shape=[512, 256], initializer=He)
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.get_variable("W3", shape=[256, 128], initializer=He)
b3 = tf.Variable(tf.random_normal([128]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

W4 = tf.get_variable("W4", shape=[128, output_size], initializer=He)
b4 = tf.Variable(tf.random_normal([output_size]))

deep = tf.add(tf.matmul(L3, W4), b4)

hypothesis = wide + deep

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

Wide_n_Deep_ACC_plot = []

def Wide_n_Deep_Model(training_epochs = 10):
    
    feed_dict_test = {Wide: Wide_data_test,
                      Deep_un_embed: Deep_data_test[Deep_col[:-2]],
                      Deep_embed: Deep_data_test[Deep_col[-2:]]}
    
    try:
        print("가장 Hit-rate가 높았던 체크포인트를 로드합니다.")
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
        saver.restore(sess, save_path=last_chk_path)
        print("Restored checkpoint from:", last_chk_path)
        
        score = sess.run(tf.sigmoid(hypothesis), feed_dict=feed_dict_test)
        WD_score = pd.DataFrame(score, columns=train_label.columns)

        print("Hit rate of Top 3: {:>.5f}".format(get_acc(WD_score, 3, test_label)))
        print("Hit rate of Top 10: {:>.5f}".format(get_acc(WD_score, 10, test_label)))
        print("Hit rate of Top 20: {:>.5f}".format(get_acc(WD_score, 20, test_label)))
        
        
        
    except:
        print("체크포인트가 없습니다. 변수를 초기화 합니다.")
        print("----------------------------------------------")
        sess.run(tf.global_variables_initializer())

        max_acc = 0
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(len(Deep_data) / batch_size)

            
            for i in range(0, len(Deep_data), batch_size):
                batch_wide = Wide_data[i:i+batch_size]
                batch_deep = Deep_data[i:i+batch_size][Deep_col[:-2]]
                batch_deep_embed = Deep_data[i:i+batch_size][Deep_col[-2:]]

                batch_label = train_label[i:i+batch_size]

                feed_dict_train = {Wide: batch_wide,
                                   Deep_un_embed: batch_deep,
                                   Deep_embed: batch_deep_embed,
                                   Y: batch_label}

                opt, c = sess.run([optimizer, cost], feed_dict=feed_dict_train)

                avg_cost += c/total_batch

                
            # 2번마다 Hit-rate를 계산합니다.
            # if (epoch % 2 == 0) or (epoch == training_epochs - 1):

            score = sess.run(tf.sigmoid(hypothesis), feed_dict=feed_dict_test)
            
            score_pd = pd.DataFrame(score, columns=train_label.columns)
            acc = get_acc(score_pd, 10, test_label)
            Wide_n_Deep_ACC_plot.append(acc)


            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            main = "[Epoch: {:>4}][Cost: {:>.5f}][Hit rate of Top 10: {:>.5f}]"
            print(main.format(epoch, avg_cost, acc))


            # Hit-rate가 가장 높은 score를 저장합니다.
            if acc > max_acc:
                WD_score = score_pd
                saver.save(sess, save_path=save_path, global_step=epoch)
                print("Saved checkpoint.")

            max_acc = max(max_acc, acc)
                

        print('Learning Finished!')
        print("----------------------------------------------")
        print("가장 Hit-rate가 높았던 체크포인트를 로드합니다.")
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
        saver.restore(sess, save_path=last_chk_path)

        print("Hit rate of Top 3: {:>.5f}".format(get_acc(WD_score, 3, test_label)))
        print("Hit rate of Top 10: {:>.5f}".format(get_acc(WD_score, 10, test_label)))
        print("Hit rate of Top 20: {:>.5f}".format(get_acc(WD_score, 20, test_label)))
        

    return WD_score


# Session

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


# Saver

save_dir = 'WnD_model/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'WnD_tensorflow')
saver = tf.train.Saver()

# Run

WD_score = Wide_n_Deep_Model(31)

