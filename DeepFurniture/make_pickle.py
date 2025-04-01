import pickle
import numpy as np
import os
import argparse
from collections import Counter
import pdb

def load_data(path):
    path = os.path.expanduser(path)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def split_collections(data, train_ratio=0.8, valid_ratio=0.1, seed=0, min_size=4, max_size=16):
    """
    data は各集合の名前をキーとする辞書とする。
    各集合は {'features': (n,4096), 'category_ids': (n,), ...} の構造を持つ。
    
    集合（キー）自体を8:1:1に分割し、分割された各集合のデータを
    リスト形式で返す。
    
    出力:
      train_x, train_y, train_z: 訓練用集合のリスト（各要素は各集合の features, 集合名配列, category_ids）
      valid_x, valid_y, valid_z: 検証用集合のリスト
      test_x,  test_y,  test_z : テスト用集合のリスト
    """
    # data のキー（集合名）をリスト化し、シャッフル
    #collection_names = list(data.keys())
    # 指定したサイズ範囲にある集合のみを取得
    #pdb.set_trace()
    collection_names = [name for name in data.keys() if min_size <= len(data[name]['features']) <= max_size]
    rng = np.random.default_rng(seed)
    rng.shuffle(collection_names)
    
    n_total = len(collection_names)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    n_test = n_total - n_train - n_valid  # 残り
    
    train_names = collection_names[:n_train]
    valid_names = collection_names[n_train:n_train+n_valid]
    test_names  = collection_names[n_train+n_valid:]
    
    train_x, train_y, train_z, train_id = [], [], [], []
    valid_x, valid_y, valid_z, valid_id = [], [], [], []
    test_x,  test_y,  test_z,  test_id  = [], [], [], []
    
    # 各集合ごとに、features, category_ids を抽出し、y は集合名の配列として用意
    for name in train_names:
        coll = data[name]
        features = np.array(coll['features'])       # (n, 4096)
        category_ids = np.array(coll['category_ids']) # (n,)
        item_ids = np.array(coll['item_ids'])
        train_x.append(features)
        train_y.append(np.array([name]))
        train_z.append(category_ids)
        train_id.append(item_ids)
        
    for name in valid_names:
        coll = data[name]
        features = np.array(coll['features'])
        category_ids = np.array(coll['category_ids'])
        item_ids = np.array(coll['item_ids'])
        valid_x.append(features)
        valid_y.append(np.array([name]))
        valid_z.append(category_ids)
        valid_id.append(item_ids)
        
    for name in test_names:
        coll = data[name]
        features = np.array(coll['features'])
        category_ids = np.array(coll['category_ids'])
        item_ids = np.array(coll['item_ids'])
        test_x.append(features)
        test_y.append(np.array([name]))
        test_z.append(category_ids)
        test_id.append(item_ids)
    
    lengths = [len(sublist) for sublist in train_z]
    length_counts = Counter(lengths)
    print(f'train_num: {len(train_x)}')
    print(f'valid_num: {len(valid_x)}')
    print(f'test_num: {len(test_x)}')
    print(f'Number of items in the train_set: {length_counts}')

    return (train_x, train_y, train_z, train_id), (valid_x, valid_y, valid_z, valid_id), (test_x, test_y, test_z, test_id)

def save_pickle_split(data_tuple, save_path):
    """
    data_tuple は (x, y, z) のタプルとする。
    個別に3回 pickle.dump() して保存するので、読み出す際は順に load してください。
    """
    with open(save_path, 'wb') as f:
        pickle.dump(data_tuple[0], f)
        pickle.dump(data_tuple[1], f)
        pickle.dump(data_tuple[2], f)
        pickle.dump(data_tuple[3], f)
    print(f"Saved {save_path}")

def main(args):
    # data.pkl を読み込む
    data = load_data(args.input)
    
    # 各集合単位で分割
    train, valid, test = split_collections(data, train_ratio=0.8, valid_ratio=0.1, seed=args.seed, min_size=args.min_num, max_size=args.max_num)
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 分割データの保存（個別に3回 dump しているので、読み出し時は
    # with open(..., 'rb') as fp: x = pickle.load(fp); y = pickle.load(fp); z = pickle.load(fp) となる）
    save_pickle_split(train, os.path.join(args.output_dir, 'train2.pkl'))
    save_pickle_split(valid, os.path.join(args.output_dir, 'valid2.pkl'))
    save_pickle_split(test,  os.path.join(args.output_dir, 'test2.pkl'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split collections in data.pkl into train/valid/test sets (by collection) with x: features, y: set_name, z: category_ids"
    )
    parser.add_argument("--input", type=str, default="pickle_data/deep_furniture4096.pkl", help="Path to input data.pkl file")
    parser.add_argument("--output_dir", type=str, default="pickle_data", help="Directory to save train.pkl, valid.pkl, test.pkl")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--min_num", type=int, default=4, help="Minimum number of items in the set")
    parser.add_argument("--max_num", type=int, default=16, help="Maximum number of items in the set")
    args = parser.parse_args()
    main(args)
