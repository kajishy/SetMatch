import os
import pickle
import numpy as np
import pdb
# 出力先ディレクトリ（適宜変更してください）
OUT_DIR = "pickle_data"

def pad_or_truncate(group, pad_size=8, feature_dim=4096):
    """
    group: NumPy 配列 (n, feature_dim)
    n が pad_size 未満なら 0 パディング、以上なら先頭 pad_size 件にトリミングして返す。
    また、元の n を返す。
    """
    n = group.shape[0]
    if n >= pad_size:
        #print("item")
        return group[:pad_size], pad_size
    else:
        if feature_dim == 1:
            return group, n
        padded = np.zeros((pad_size, feature_dim), dtype=group.dtype)
        padded[:n] = group
        return padded, n

def main(label_dir, n_comb=1, n_cands=8, pad_size=8, seed=0):
    # test.pkl は以下のように保存されている前提：
    #   test_x: 各集合の features, リスト（各要素は (num_items, 4096) の NumPy 配列）
    #   test_y, test_z はその他の情報（ここでは使用しません）
    test_pkl = os.path.join(label_dir, "test2.pkl")
    with open(test_pkl, "rb") as f:
        test_x = pickle.load(f)
        test_y = pickle.load(f)
        test_z = pickle.load(f)
        test_id = pickle.load(f)
    
    num_collections = len(test_x)
    np.random.seed(seed)
    
    # 各候補グループごとに、最終的に出力する x, x_size, y を格納するリスト
    all_x = []       # 各候補グループ: (pad_size, 4096)
    all_x_size = []  # 元のアイテム数（パディング前のサイズ）
    all_y = []       # マッチングラベル
    all_id = []      # アイテム画像のid
    
    # 各テストケース（＝各集合を対象とする）
    for i in range(num_collections):
        # --- 正例グループの生成 ---
        # 対象集合（正例用）は、現在の i 番目と、追加で n_comb-1 個をランダムに選ぶ
        indices = np.delete(np.arange(num_collections), i)
        if n_comb - 1 > 0:
            others = np.random.choice(indices, n_comb - 1, replace=False).tolist()
        else:
            others = []
        target = [i] + others  # 正例候補となる集合のインデックス
        
        # 各対象集合から、画像群をランダムに半分に分割して取得
        query_list = []   # クエリ側（setX）用
        pos_list = []     # 正解候補側（setY）用
        query_id_list = []   # クエリ側（setX）の画像id
        pos_id_list = []     # 正解候補側（setY）の画像id
        for j in target:
            images = np.array(test_x[j])  # (num_items, 4096)
            ids = np.array(test_id[j])
            n_images = images.shape[0]
            # 半分の件数（切り捨て）
            y_size = n_images // 2
            # 作成するためのマスク： (n_images - y_size) True, y_size False
            mask = np.array([True]*(n_images - y_size) + [False]*y_size)
            mask = np.random.permutation(mask)
            query_list.append(images[mask])
            pos_list.append(images[~mask])
            query_id_list.append(ids[mask])
            pos_id_list.append(ids[~mask])
        # 正例グループは、各対象集合ごとの結果を連結して一つにまとめる
        query_group = np.concatenate(query_list, axis=0)   # クエリ側
        pos_group = np.concatenate(pos_list, axis=0)       # 正解候補側
        query_id_group = np.concatenate(query_id_list, axis=0)   # クエリ側
        pos_id_group = np.concatenate(pos_id_list, axis=0)       # 正解候補側

        
        # --- 負例グループの生成 ---
        neg_groups = []  # リストに各負例候補グループを格納
        neg_id_groups = []  # リストに各負例候補グループidを格納
        num_neg = n_cands - 1  # 負例候補の数
        current_target = set(target)  # すでに選んだ正例集合
        for neg_idx in range(num_neg):
            # 正例集合に含まれていない集合から n_comb 個選ぶ
            avail = np.delete(np.arange(num_collections), list(current_target))
            negatives = np.random.choice(avail, n_comb, replace=False).tolist()
            # 更新して重複選択を防止
            current_target.update(negatives)
            # 各負例集合から、画像群をランダムに並べ替え、先頭半分を取得
            neg_subgroups = []
            neg_id_subgroups = []
            for k in negatives:
                images = np.array(test_x[k])
                ids = np.array(test_id[k])
                perm_indices = np.random.permutation(images.shape[0])
                # インデックスを使って画像とIDを同じ順序で並び替える
                permuted_images = images[perm_indices]
                permuted_ids = ids[perm_indices]  # ← images と同じ並び替えを適用
                n_imgs = permuted_images.shape[0]
                sub_count = n_imgs // 2
                neg_subgroups.append(permuted_images[:sub_count])
                neg_id_subgroups.append(permuted_ids[:sub_count])
            # 負例グループは各負例集合の結果を連結
            neg_group = np.concatenate(neg_subgroups, axis=0)
            neg_groups.append(neg_group)
            neg_id_group = np.concatenate(neg_id_subgroups, axis=0)
            neg_id_groups.append(neg_id_group)
        
        # --- 最終候補グループのまとめ ---
        # ここでは、クエリグループと正例グループがマッチする（同じラベル）とする
        # 負例グループはそれぞれ別のラベルを割り当てる
        candidate_groups = []
        candidate_labels = []
        candidate_id_groups = []
        
        # 正例グループ：クエリと正解候補を別々に出力し、どちらもラベル 1 とする
        candidate_groups.append(query_group)
        candidate_labels.append(1)
        candidate_groups.append(pos_group)
        candidate_labels.append(1)
        candidate_id_groups.append(query_id_group)
        candidate_id_groups.append(pos_id_group)
        
        # 負例グループ：ラベルは 2, 3, ... とする
        label_counter = 2
        for neg_group in neg_groups:
            candidate_groups.append(neg_group)
            candidate_labels.append(label_counter)
            label_counter += 1
        for neg_id_group in neg_id_groups:
            candidate_id_groups.append(neg_id_group)
        
        #pdb.set_trace()
        # --- 各候補グループを 0 パディングまたはトリミングしてサイズを揃える ---
        # ここで、各候補グループは個別に (pad_size, 4096) に整形し、その元の件数も記録
        for group, label,group_ids in zip(candidate_groups, candidate_labels, candidate_id_groups):
            # group: (n, 4096)
            padded, orig_size = pad_or_truncate(group, pad_size=pad_size, feature_dim=4096)
            padded_id, orig_size_id = pad_or_truncate(group_ids, pad_size=pad_size, feature_dim=1)
            all_x.append(padded)
            all_x_size.append(orig_size)
            all_y.append(label)
            all_id.append(padded_id)
    
    # 全テストケースの候補グループを縦に結合（候補グループごとに 1 サンプルとなる）
    final_x = np.stack(all_x, axis=0)         # shape: (total_samples, pad_size, 4096)
    final_x_size = np.array(all_x_size)         # shape: (total_samples,)
    final_y = np.array(all_y)                   # shape: (total_samples,)
    final_id = all_id#np.stack(all_id, axis=0)         # shape: (total_samples, pad_size, 1)
    
    # 出力ファイル名例：test_example_cand{n_cands}.pkl
    output_path = os.path.join(label_dir, f"test2_example_cand{n_cands}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(final_x, f)
        pickle.dump(final_x_size, f)
        pickle.dump(final_y, f)
        pickle.dump(final_id, f)
    
    print(f"Saved: {output_path}")
    
if __name__ == "__main__":
    # 例: label_dir に test.pkl があるディレクトリを指定、n_comb=1, n_cands=5 など
    main(label_dir="pickle_data", n_comb=1, n_cands=5, pad_size=8, seed=0)


"""
import os
import pickle 
import numpy as np
import json
import gzip

# 出力ディレクトリなどの定数（適宜設定してください）
OUT_DIR = "pickle_data"

def main(label_dir, n_comb=1, n_cands=8):
    # test.pkl を読み込み（各集合の features, 集合名, category_ids）
    with open(os.path.join(label_dir, "test.pkl"), "rb") as f:
        test_x = pickle.load(f)
        test_y = pickle.load(f)
        test_z = pickle.load(f)
        
    # test_x は各集合ごとの features を格納したリスト
    num_collections = len(test_x)
    
    # 各テストケースを生成
    for i in range(10):
        data = {}

        # 現在のテストケース（集合 i）以外のインデックスをリスト化
        lst = np.delete(np.arange(num_collections), i)
        # n_comb - 1 個をランダムに選び、現在の i と合わせて target とする
        others = np.random.choice(lst, n_comb - 1, replace=False).tolist()
        target = [i] + others

        setX_images, setY_images = [], []
        # target 内の各集合について、features を直接利用してクエリと正解候補を作成
        for j in target:
            images = np.array(test_x[j])  # shape: (num_items, 4096)
            n_images = images.shape[0]
            # 半分を正解候補側 (setY), 残りをクエリ側 (setX) にする
            y_size = n_images // 2
            
            # True: クエリ側, False: 正解候補側 とするマスクを作成し、ランダムに並び替え
            xy_mask = np.array([True] * (n_images - y_size) + [False] * y_size)
            xy_mask = np.random.permutation(xy_mask)
            
            setX_images.extend(images[xy_mask].tolist())
            setY_images.extend(images[~xy_mask].tolist())

        data["query"] = setX_images
        answers = [setY_images]  # 最初の候補は正解

        # 追加で負例候補を n_cands - 1 個生成する
        for j in range(n_cands - 1):
            # 既に target に含まれている集合以外から n_comb 個選ぶ
            lst = np.delete(np.arange(num_collections), target)
            negatives = np.random.choice(lst, n_comb, replace=False).tolist()
            # 重複がないことを確認
            assert len(set(target) & set(negatives)) == 0, "Negative candidates overlap with target."
            # 選ばれた negatives を target に追加して再選択されないようにする
            target += negatives

            setY_images_neg = []
            for k in negatives:
                images = np.array(test_x[k])
                # 画像群をランダムに並べ替え
                images = np.random.permutation(images)
                n_images = images.shape[0]
                y_size = n_images // 2
                # 負例候補として、先頭 y_size 枚を使用
                setY_images_neg.extend(images[:y_size].tolist())
            answers.append(setY_images_neg)

        data["answers"] = answers

        # 出力先ディレクトリの作成
        out_path="pickle_data"
        file_path = os.path.join(out_path, "{0:05d}.pkl".format(i))
        
        # data を pickle 形式で保存する
        with open(file_path, "wb") as file:
            pickle.dump(data, file)
            
    print("Test candidate files generated successfully.")

if __name__ == "__main__":
    # 例: label_dir に test.pkl が存在するディレクトリを指定、n_comb=1, n_cands=8（必要に応じて変更）
    main(label_dir="pickle_data", n_comb=1, n_cands=8)



#----------------------------------------------------------------------
""
def _read_feature(path):
    with gzip.open(path, mode="rt", encoding="utf-8") as f:
        feature = json.loads(f.read())
    return feature


def main(label_dir, n_comb=1, n_cands=8):
    test_data = pkl.load(open(os.path.join(label_dir, "test.pkl")))

    for i in range(len(test_data)):
        data = {}

        lst = np.delete(np.arange(len(test_data)), i)
        others = np.random.choice(lst, n_comb - 1, replace=False).tolist()
        target = [i] + others

        setX_images, setY_images = [], []
        for j in target:
            items = test_data[j]["items"]
            images = []
            for img in items:
                path = os.path.join(INPUT_DIR, str(img["item_id"]) + ".json.gz")
                images.append(_read_feature(path))
            images = np.array(images)

            y_size = len(images) // 2

            xy_mask = [True] * (len(images) - y_size) + [False] * y_size
            xy_mask = np.random.permutation(xy_mask)
            setX_images.extend(images[xy_mask].tolist())
            setY_images.extend(images[~xy_mask].tolist())

        data["query"] = setX_images
        answers = [setY_images]

        for j in range(n_cands - 1):
            lst = np.delete(np.arange(len(test_data)), target)
            negatives = np.random.choice(lst, n_comb, replace=False).tolist()
            assert len(set(target) & set(negatives)) == 0
            target += negatives  # avoid double-selecting

            setY_images = []
            for k in negatives:
                items = test_data[k]["items"]
                images = []
                for img in items:
                    path = os.path.join(INPUT_DIR, str(img["item_id"]) + ".json.gz")
                    images.append(_read_feature(path))
                images = np.random.permutation(images)

                y_size = len(images) // 2
                setY_images.extend(images[:y_size].tolist())

            answers.append(setY_images)

        data["answers"] = answers

    # 保存先ディレクトリの作成
    out_dir = os.path.join(
        OUT_DIR, "test_ncand{}".format(n_cands), os.path.basename(label_dir)
    )
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    file_path = os.path.join(out_dir, "{0:05d}_cand{1}.pkl".format(i, n_cands))

    # data を pickle 形式で保存する
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
"""

