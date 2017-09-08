import pandas as pd
import numpy as np
import pickle
import sys
import os
import urllib.request
import tarfile
import zipfile


def load_train_data():

    # Train

    train_data = pd.read_csv("./Lpoint_data/train_data.csv", encoding="cp949")
    train_categorical = pd.read_csv("./Lpoint_data/train_categorical.csv", encoding="cp949")
    train_continuous = pd.read_csv("./Lpoint_data/train_continuous.csv", encoding="cp949")
    train_Segment = pd.read_csv("./Lpoint_data/train_Segment.csv", encoding="cp949")
    
    train_label = pd.read_csv("./Lpoint_data/train_label.csv", encoding="cp949")


    # Fill NA

    train_label = train_label.fillna(0)
    train_Segment = train_Segment.fillna(0)
    train_continuous = train_continuous.fillna(0)
    
    return train_data, train_categorical, train_continuous, train_Segment, train_label

    
def load_test_data():

    # Test

    test_data = pd.read_csv("./Lpoint_data/test_data.csv", encoding="cp949")
    test_categorical = pd.read_csv("./Lpoint_data/test_categorical.csv", encoding="cp949")
    test_continuous = pd.read_csv("./Lpoint_data/test_continuous.csv", encoding="cp949")
    test_Segment = pd.read_csv("./Lpoint_data/test_Segment.csv", encoding="cp949")
    
    test_label = pd.read_csv("./Lpoint_data/test_label.csv", encoding="cp949")
    test_label_eval = pd.read_csv("./Lpoint_data/test_label_real.csv", encoding="cp949")


    # Fill NA

    test_label_eval = test_label_eval.fillna(0)
    test_Segment = test_Segment.fillna(0)
    test_continuous = test_continuous.fillna(0)
    
    return test_data, test_categorical, test_continuous, test_Segment, test_label, test_label_eval


def load_pred_data():

    # Pred

    pred_label = pd.read_csv("./Lpoint_data/pred_label.csv", encoding="cp949")
    pred_label_eval = pd.read_csv("./Lpoint_data/pred_label_real.csv", encoding="cp949")
    pred_categorical = pd.read_csv("./Lpoint_data/pred_categorical.csv", encoding="cp949")
    pred_continuous = pd.read_csv("./Lpoint_data/pred_continuous.csv", encoding="cp949")

    pred_continuous = pred_continuous.fillna(0)

    return pred_label, pred_label_eval, pred_categorical, pred_continuous



def load_Wide_Deep_train_data():

    Wide_data = pd.read_csv("./Lpoint_data/Wide_data.csv", encoding='cp949')
    Deep_data = pd.read_csv("./Lpoint_data/Deep_data.csv", encoding='cp949')
    train_label = pd.read_csv("./Lpoint_data/train_label.csv", encoding="cp949")

    train_label = train_label.fillna(0)

    return Wide_data, Deep_data, train_label


def load_Wide_Deep_test_data():

    Wide_data_test = pd.read_csv("./Lpoint_data/Wide_data_test.csv", encoding='cp949')
    Deep_data_test = pd.read_csv("./Lpoint_data/Deep_data_test.csv", encoding='cp949')
    test_label = pd.read_csv("./Lpoint_data/test_label.csv", encoding="cp949")

    return Wide_data_test, Deep_data_test, test_label


def print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download 진행중: {0:.0%}".format(pct_complete)

    sys.stdout.write(msg)
    sys.stdout.flush()


def Lpoint_data_auoto_download():

    url = "https://www.dropbox.com/s/q1vc3lxp45skz98/Lpoint_data.zip?dl=1"

    download_dir = "Lpoint_data/"

    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    if not os.path.exists(file_path):
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        file_path, _ = urllib.request.urlretrieve(
            url=url, filename=file_path, reporthook=print_download_progress)

        print()
        print("다운로드 완료.")

        if file_path.endswith(".zip?dl=1"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)

        print("언팩 완료.")

    else:
        print("데이터가 이미 다운로드 되어있습니다..")