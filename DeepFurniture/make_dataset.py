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
    def __init__(self, batch_size=20, max_item_num=8, max_data=np.inf):
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
    def __init__(self, cand_num=4):
        self.data_path = f"pickle_data"
        self.cand_num = cand_num
        # (number of groups in one batch) = (cand_num) + (one query)
        self.batch_grp_num = cand_num + 1

        # load data
        with open(f'{self.data_path}/test_example_cand{self.cand_num}.pkl', 'rb') as fp:
            self.x = pickle.load(fp)
            self.x_size = pickle.load(fp)
            self.y = pickle.load(fp)
            self.set_id = pickle.load(fp)
            self.id = pickle.load(fp)


            
#-------------------------------
# gather and save pickle data
def make_packed_pickle(files, save_path):
    X = []
    Y = []
    for file in files:
        with open(file, 'rb') as f:
            try:
                x = np.array(pickle.load(f))
            except:
                pass
                #print(f"{os.path.basename(file)}: empty")
            else:
                #print(f"{os.path.basename(file)}: {x.shape}")
                X.append(x)
                Y.append(int(os.path.basename(file).split('.')[0]))

    print(f"save to {save_path}")
    with open(save_path,'wb') as f:
        pickle.dump(X,f)
        pickle.dump(Y,f)
#-------------------------------



#-------------------------------
def make_packed_example_test_pickle(files, save_path, max_item_num):
    X = []
    X_size = []
    Y = []
    set_ids = []
    image_ids = [] 
    for file in files:
        with open(file, 'rb') as f:
            try:
                x = pickle.load(f)
                ids = pickle.load(f)
            except:
                pass
            else:
                dim = len(x[0][0])
                stack = lambda z: np.vstack([np.vstack(z),np.zeros([np.max([0,max_item_num - len(z)]),dim])])[:max_item_num]
                X.append(stack(x[0]))
                X_size.append(np.float(len(x[0])))
                [X.append(stack(x[1][i])) for i in range(len(x[1]))]
                [X_size.append(np.float(len(x[1][i]))) for i in range(len(x[1]))]
                Y.append(np.hstack([0,np.arange(len(x[1]))]))
                set_ids.append(str(os.path.basename(file).split('.')[0]))
                image_ids.append(ids[0])
                image_ids.extend([i if isinstance(i[0], int) else i[0] for i in ids[1]])
        
    X = np.array(X)
    X_size = np.array(X_size)
    Y = np.hstack(Y)

    with open(save_path,'wb') as f:
        pickle.dump(X,f)
        pickle.dump(X_size,f)
        pickle.dump(Y,f)      
        pickle.dump(set_ids,f)  
        pickle.dump(image_ids,f) 
    print("saving pickle files to " + str(save_path))
#-------------------------------

#-------------------------------        
if __name__ == '__main__':   
    # parameters
    split = 0
    max_item_num = 8
    n_cands=5

    # gen = testDataGenerator()

    # path
    data_path = f"pickle_data"
    train_path = f"{data_path}/train"
    test_path = f"{data_path}/test"
    valid_path = f"{data_path}/valid"
    test_example_path = f"json_data/pickles/test_examples_ncomb_1_ncands_{n_cands}"

    # file list
    #train_files = glob.glob(f"{train_path}/*.pkl")
    #test_files = glob.glob(f"{test_path}/*.pkl")
    #valid_files = glob.glob(f"{valid_path}/*.pkl")
    test_example_files = glob.glob(f"{test_example_path}/*.pkl")        
    
    # save packed pickle
    #make_packed_pickle(train_files, f"{data_path}/train.pkl")
    #make_packed_pickle(test_files, f"{data_path}/test.pkl")
    #make_packed_pickle(valid_files, f"{data_path}/valid.pkl")
    make_packed_example_test_pickle(test_example_files, f"{data_path}/test_example_cand{n_cands}.pkl", max_item_num=max_item_num)