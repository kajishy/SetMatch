import tensorflow as tf
import pickle
import glob
import numpy as np
import pdb
import os
import sys

def convert_label(label):
    """
    label が numpy 配列の場合、その中の値を取り出す。
    すでにスカラーであればそのまま返す。
    """
    if isinstance(label, np.ndarray):
        # 要素数が1なら、その要素を返す
        if label.size == 1:
            return label.item()
        else:
            # 複数要素の場合はリストに変換（用途に合わせて調整してください）
            return label.tolist()
    return label

def ensure_numeric(labels):
    new_labels = []
    for lab in labels:
        lab_conv = convert_label(lab)
        # ここで文字列の場合に数値に変換（マッピング）
        if isinstance(lab_conv, (str, np.str_)):
            new_labels.append(lab_conv)
        else:
            new_labels.append(lab_conv)
    return new_labels
#-------------------------------
class trainDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, year=2017, split=0, batch_size=20, max_item_num=8, max_data=np.inf):
        data_path = f"pickle_data"
        self.max_item_num = max_item_num
        self.batch_size = batch_size
        
        # load train data
        with open(f'{data_path}/train.pkl', 'rb') as fp:
            self.x_train = pickle.load(fp)
            self.y_train = pickle.load(fp)
        self.train_num = len(self.x_train)
        # 各要素が numpy 配列の場合、スカラー文字列に変換
        self.y_train = ensure_numeric(self.y_train)
        # もし文字列ならユニークな値に基づいて数値ラベルに変換
        if isinstance(self.y_train[0], (str, np.str_)):
            unique = sorted(set(self.y_train))
            self.label_map = {lab: i for i, lab in enumerate(unique)}
            self.y_train = [self.label_map[label] for label in self.y_train]
        

        # limit data
        if self.train_num > max_data:
            self.train_num = max_data

        # load validation data
        with open(f'{data_path}/valid.pkl', 'rb') as fp:
            self.x_valid = pickle.load(fp)
            self.y_valid = pickle.load(fp)
        self.valid_num = len(self.x_valid)    
        self.y_valid = ensure_numeric(self.y_valid)
        if isinstance(self.y_valid[0], (str, np.str_)):
            unique_val = sorted(set(self.y_valid))
            self.label_map_val = {lab: i for i, lab in enumerate(unique_val)}
            self.y_valid = [self.label_map_val[label] for label in self.y_valid]
        
        
        # width and height of image
        self.dim = len(self.x_train[0][0])

        # shuffle index
        self.inds = np.arange(len(self.x_train))
        self.inds_shuffle = np.random.permutation(self.inds)

    def __getitem__(self, index):
        x, x_size, y = self.data_generation(self.x_train, self.y_train, self.inds_shuffle, index)
        return (x, x_size), y

    def data_generation(self, x, y, inds, index):
        #pdb.set_trace()
        if index >= 0:
            # extract x and y
            start_ind = index * self.batch_size
            batch_inds = inds[start_ind:start_ind+self.batch_size]
            x_tmp = [x[i] for i in batch_inds]
            y_tmp = [y[i] for i in batch_inds]
            batch_size = self.batch_size
        else:
            x_tmp = x
            y_tmp = y
            batch_size = len(x_tmp)

        # split x
        x_batch = []
        x_size_batch = []
        y_batch =[]
        split_num = 2
        for ind in range(batch_size):
            x_tmp_split = np.array_split(x_tmp[ind][np.random.permutation(len(x_tmp[ind]))],split_num)
            x_tmp_split_pad = [np.vstack([x, np.zeros([np.max([0,self.max_item_num-len(x)]),self.dim])])[:self.max_item_num] for x in x_tmp_split] # zero padding

            x_batch.append(x_tmp_split_pad)
            x_size_batch.append([len(x_tmp_split[i]) for i in range(split_num)])
            y_batch.append(np.ones(split_num)*y_tmp[ind])

        x_batch = np.vstack(x_batch)
        x_size_batch = np.hstack(x_size_batch).astype(np.float32)
        y_batch = np.hstack(y_batch)

        return x_batch, x_size_batch, y_batch

    def data_generation_val(self):
        #pdb.set_trace()
        x_valid, x_size_val, y_valid = self.data_generation(self.x_valid, self.y_valid, self.inds, -1)
        return x_valid, x_size_val, y_valid

    def __len__(self):
        # number of batches in one epoch
        batch_num = int(self.train_num/self.batch_size)

        return batch_num

    def on_epoch_end(self):
        # shuffle index
        self.inds_shuffle = np.random.permutation(self.inds)
#-------------------------------

#-------------------------------
class testDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, year=2017, split=0, cand_num=4):
        self.data_path = f"pickle_data"
        self.cand_num = cand_num
        # (number of groups in one batch) = (cand_num) + (one query)
        self.batch_grp_num = cand_num + 1

        # load data
        with open(f'{self.data_path}/test_example_cand{self.cand_num}.pkl', 'rb') as fp:
            self.x = pickle.load(fp)
            self.x_size = pickle.load(fp)
            self.y = pickle.load(fp)

"""
#train_far.pickleなどを使用する場合のデータローダ
import tensorflow as tf
import pickle
import numpy as np
import pdb
import os
import sys
from collections import Counter
import random

def interleave_arrays(A, B):
    ""
    #AとBを交互に結合する。
    #A[0], B[0], A[1], B[1], ..., A[-1], B[-1]という順に並ぶようにする。
    ""
    interleaved = np.empty((A.shape[0] + B.shape[0], *A.shape[1:]), dtype=A.dtype)
    interleaved[0::2] = A  # 偶数番目にAを配置
    interleaved[1::2] = B  # 奇数番目にBを配置
    return interleaved

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, year=2017, split=0, batch_size=20, max_item_num=8, max_data=np.inf, mlp_flag=False, set_loss=False, whitening_path=None, seed_path=None):
        data_path = "pickle_data"  # train, valid, test.pickle のあるフォルダパス
        self.max_item_num = max_item_num    # ここでは例として8に設定（0パディング後の固定サイズ）
        self.batch_size = batch_size
        self.isMLP = mlp_flag
        self.set_loss = set_loss
        
        # load train data
        with open(f'{data_path}/train_fur.pkl', 'rb') as fp:
            self.query_tr, self.positive_tr, self.y_tr, self.y_catQ_tr, self.y_catP_tr, self.x_size_tr, self.y_size_tr = pickle.load(fp)
        self.x_train = interleave_arrays(self.query_tr, self.positive_tr)
        self.y_train = np.repeat(self.y_tr, 2)
        self.x_size_train = interleave_arrays(self.x_size_tr, self.y_size_tr)
        # self.x_size_tr, self.y_size_tr は各サンプルの元々の要素数（例：8未満や8以上の値）を保持している
        
        self.train_num = len(self.x_train)
        # feature vector dimension
        self.dim = len(self.query_tr[0][0])
        
        # limit data if needed
        if self.train_num > max_data:
            self.train_num = max_data
        
        # load validation data
        with open(f'{data_path}/validation_fur.pkl', 'rb') as fp:
            self.query_val, self.positive_val, self.y_val, self.y_catQ_val, self.y_catP_val, self.x_size_val, self.y_size_val = pickle.load(fp)
        self.x_valid = interleave_arrays(self.query_val, self.positive_val)
        self.y_valid = np.repeat(self.y_val, 2)
        self.x_size_valid = interleave_arrays(self.x_size_val, self.y_size_val)
        
        # load test data
        with open(f'{data_path}/test_fur.pkl', 'rb') as fp:
            self.query_test, self.positive_test, self.y_test, self.y_catQ_test, self.y_catP_test, self.x_size_test, self.y_size_test, self.x_id_test, self.y_id_test, self.scene_id_dict = pickle.load(fp) 
        self.x_test = interleave_arrays(self.query_test, self.positive_test)
        self.xy_size_test = interleave_arrays(self.x_size_test, self.y_size_test)
        self.y_test = np.repeat(self.y_test, 2)
        #pdb.set_trace()

        # 初期のインデックスは順番そのまま（並び順は固定）  
        self.inds = np.arange(len(self.x_train))
        # 初期状態ではシフトなし（つまり元の順序）を使う
        self.inds_shifted = self.inds.copy()
    
    def __len__(self):
        # バッチ数
        return self.train_num // self.batch_size
    
    def __getitem__(self, index):
        # self.inds_shuffle からバッチ用のインデックスを抽出してデータ生成
        x, x_size, y, b_size = self.data_generation(self.x_train, self.y_train, self.x_size_train, self.inds_shifted, self.batch_size, index)
        return (x, x_size), y
    
    def validation_generator(self):
        # validation はシャッフルせずに全件（またはバッチごと）返す
        #pdb.set_trace()
        x, x_size, y, b_size = self.data_generation(self.x_valid, self.y_valid, self.x_size_valid, np.arange(len(self.x_valid)), self.batch_size, -1)
        return (x, x_size), y
    
    def on_epoch_end(self):
        # エポック終了時にランダムな開始位置だけを変更して、並び順は固定のままにする
        shift = np.random.randint(0, len(self.inds))
        self.inds_shifted = np.roll(self.inds, shift)
    
    def data_generation(self, x, y, sizes, inds, b_size, index):
        ""
        #x: すでに集合として与えられたデータの全体 (リストまたは配列)
        #y: それに対応するラベルの全体 (リストまたは配列)
        #sizes: 各サンプルの元の要素数を保持した配列（例: self.x_size_tr）
        #inds: シャッフル済みのインデックス配列
        #b_size: バッチサイズ
        #index: バッチの開始インデックス（-1なら全データを返す）
        ""
        if index >= 0:
            start_ind = index * b_size
            end_ind = start_ind + b_size
            batch_inds = inds[start_ind:end_ind]
            x_batch = [x[i] for i in batch_inds]
            y_batch = [y[i] for i in batch_inds]
            size_batch = [sizes[i] for i in batch_inds]
        else:
            # index == -1 の場合、inds の順に全データを返す
            x_batch = [x[i] for i in inds]
            y_batch = [y[i] for i in inds]
            size_batch = [sizes[i] for i in inds]
            b_size = len(x_batch)
        
        b_size = len(x_batch)
        # 各 x を max_item_num に合わせて zero padding（各 x は集合なのでそのまま扱う） 
        x_batch_pad = [
            np.vstack([xi, np.zeros([max(0, self.max_item_num - len(xi)), self.dim])])[:self.max_item_num]
            for xi in x_batch
        ]
        # x_size_batch は、元の要素数から、実際にネットワークに入力される数（最大値は max_item_num）を返す
        x_size_batch = np.array([min(s, self.max_item_num) for s in size_batch], dtype=np.float32)
        
        # numpy 配列に変換
        x_batch_pad = np.array(x_batch_pad)
        y_batch = np.array(y_batch)
        
        return x_batch_pad, x_size_batch, y_batch, b_size
    
    def test_generator(self, b_size):
        # テストデータを、シャッフルせずインデックス順にバッチサイズ分切り出して返す
        x, x_size, y, size = self.data_generation(self.x_test, self.y_test, self.xy_size_test, np.arange(len(self.x_test)), b_size, -1)
        return x, x_size, y, b_size+1

"""
