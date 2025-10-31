import numpy as np
import glob
import gzip
import json
import pickle
import pathlib
import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import pdb

def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    else:
        return obj


def make_test_examples(
    #test: List,
    test_all: Tuple[List, List, List, List],#ラベルや画像IDなど
    path: pathlib.Path,
    n_comb: int = 1, 
    n_cands: int = 4,
    seed: int = 0,
    mode: str = "equal",          # "equal", "random_split", "random_missing"
    missing_rate: float = 0.3,    # ランダム欠損の割合
    max_item_num = 8,
):
    print(f"Make test dataset. mode={mode}")
    test, test_y, test_z, test_id = test_all
    np.random.seed(seed)

    test_examples = []
    #pdb.set_trace()
    for i in range(len(test)):
        example = {}

        lst = np.delete(np.arange(len(test)), i)
        others = np.random.choice(lst, n_comb - 1, replace=False).tolist()
        target = [i] + others

        items = test[i]
        ids = test_id[i]

        if mode == "equal":
            # ===== 等分モード =====
            y_size = len(items) // 2
            xy_mask = [True] * (len(items) - y_size) + [False] * y_size
            xy_mask = np.random.permutation(xy_mask)

            setX_items = items[xy_mask].tolist()
            setY_items = items[~xy_mask].tolist()
            setX_ids   = ids[xy_mask].tolist()
            setY_ids   = ids[~xy_mask].tolist()

            query_set_ids = [test_y[i]]

            example = {
                "query": setX_items,
                "query_set_ids": query_set_ids,
                "query_set_images_ids": setX_ids,
            }

            # ===== ネガティブ候補生成 =====
            answers = [setY_items]
            answers_set_images_ids = [setY_ids]
            answers_set_ids = [query_set_ids]

            for k in range(n_cands - 1):
                lst = np.delete(np.arange(len(test)), target)
                negatives = np.random.choice(lst, n_comb, replace=False).tolist()
                assert len(set(target) & set(negatives)) == 0
                target += negatives  # avoid double-selecting

                setY_items = []
                setY_ids = []
                setY_set_ids = []
                for neg in negatives:
                    neg_items = test[neg]
                    neg_ids = test_id[neg]
                    perm_indices = np.random.permutation(neg_items.shape[0])
                    neg_items = neg_items[perm_indices]
                    neg_ids = neg_ids[perm_indices]
                    y_size = len(neg_items) // 2
                    setY_items.append(neg_items[:y_size])
                    setY_ids.append(neg_ids[:y_size])
                    setY_set_ids.append(test_y[neg])

                answers.extend(setY_items)
                answers_set_images_ids.extend(setY_ids)
                answers_set_ids.extend(setY_set_ids)

            example["answers"] = answers
            example["answers_set_images_ids"] = answers_set_images_ids
            example["answers_set_ids"] = answers_set_ids

            test_examples.append(example)

        elif mode == "ratio":
            # ===== 比率モード =====
            n = len(items)
            ratios = [(0.3,0.7),(0.7,0.3),(0.4,0.6),(0.6,0.4),(0.5,0.5)]
            seen_splits = set()  # 重複防止

            for r1, r2 in ratios:
                s1 = round(n * r1)
                s2 = n - s1
                if s1 < 2 or s2 < 2:
                    continue
                #split = tuple(sorted([s1, s2]))
                split = (s1, s2) 
                if split in seen_splits:
                    continue
                seen_splits.add(split)

                xy_mask = [True] * s1 + [False] * s2
                xy_mask = np.random.permutation(xy_mask)

                setX_items = items[xy_mask].tolist()
                setY_items = items[~xy_mask].tolist()
                setX_ids   = ids[xy_mask].tolist()
                setY_ids   = ids[~xy_mask].tolist()

                query_set_ids = [test_y[i]]

                example = {
                    "query": setX_items,
                    "query_set_ids": query_set_ids,
                    "query_set_images_ids": setX_ids,
                }

                # ===== ネガティブ候補生成 =====
                answers = [setY_items]
                answers_set_images_ids = [setY_ids]
                answers_set_ids = [query_set_ids]

                for k in range(n_cands - 1):
                    lst = np.delete(np.arange(len(test)), target)
                    negatives = np.random.choice(lst, n_comb, replace=False).tolist()
                    target += negatives

                    setY_items = []
                    setY_ids = []
                    setY_set_ids = []
                    for neg in negatives:
                        neg_items = test[neg]
                        neg_ids = test_id[neg]
                        perm_indices = np.random.permutation(neg_items.shape[0])
                        neg_items = neg_items[perm_indices]
                        neg_ids = neg_ids[perm_indices]
                        y_size = len(neg_items) // 2
                        setY_items.append(neg_items[:y_size])
                        setY_ids.append(neg_ids[:y_size])
                        setY_set_ids.append(test_y[neg])

                    answers.extend(setY_items)
                    answers_set_images_ids.extend(setY_ids)
                    answers_set_ids.extend(setY_set_ids)

                example["answers"] = answers
                example["answers_set_images_ids"] = answers_set_images_ids
                example["answers_set_ids"] = answers_set_ids

                test_examples.append(example)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        
    # ファイル名にモードを追加
    json_name = f"test_examples_{mode}_ncomb_{n_comb}_ncands_{n_cands}_itemmax_{max_item_num}"
    with open(path / f"{json_name}.json", "w") as f:
        json.dump(convert_ndarray_to_list(test_examples), f, indent=2)

    return json_name


def save_test_examples(
    json_path: pathlib.Path, output_dir: pathlib.Path,
):

    test_examples = json.load(open(json_path))
    for i in tqdm.tqdm(range(len(test_examples))):
        example = test_examples[i]
        query = example["query"]
        gallery = example["answers"]
        #add image ids
        query_ids = example["query_set_images_ids"]
        gallery_ids = example["answers_set_images_ids"]

        query_features = query
        gallery_features = gallery

        example_features = [query_features, gallery_features]
        example_ids = [query_ids, gallery_ids]
        example_id = example["query_set_ids"][0]
        with open(output_dir / f"{example_id}_{i}.pkl", "wb") as f:
            pickle.dump(example_features, f)
            pickle.dump(example_ids, f)
            pickle.dump(example_id, f)
            pickle.dump(example_id, f)  # ここはシーンIDのみ

    assert len(glob.glob(str(output_dir / "*"))) == len(test_examples), "unmatched case"

    return


def main(args):
    split_count_by_n = {}

    for n in range(4, 17):
        items = np.arange(n)
        ids = np.arange(n)
        seen_splits = set()
        for r1, r2 in [(0.3,0.7),(0.7,0.3),(0.4,0.6),(0.6,0.4),(0.5,0.5)]:
            s1 = round(n * r1)
            s2 = n - s1
            if s1 < 2 or s2 < 2:
                continue
            split = (s1, s2)
            if split in seen_splits:
                continue
            seen_splits.add(split)
        print(n,":",seen_splits)
        split_count_by_n[n] = len(seen_splits)

    print(split_count_by_n)
    #return
    # dataset
    label_dir = "pickle_data"
    test_pkl = os.path.join(label_dir, "test.pkl")
    with open(test_pkl, "rb") as f:
        test_x = pickle.load(f)
        test_y = pickle.load(f)
        test_z = pickle.load(f)
        test_id = pickle.load(f)

    output_root = pathlib.Path(args.data_root)
    output_root.mkdir(parents=True, exist_ok=True)
    print("saving a json file to " + str(output_root))
    json_name = make_test_examples(
            (test_x, test_y, test_z, test_id), 
            path=output_root, 
            n_comb=1, 
            n_cands=args.n_cands, 
            seed=args.split,
            mode=args.mode,
            missing_rate=args.missing_rate,
            max_item_num=args.max_item_num,
        )
    json_path = output_root /  f"{json_name}.json" 
    print("saved: " + str(json_path))

    output_pickles_dir = output_root / "pickles" / json_name
    output_pickles_dir.mkdir(parents=True, exist_ok=True)
    print("saving pickle files to " + str(output_pickles_dir))
    save_test_examples(json_path, output_pickles_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--data_root", type=str, default="json_data")
    parser.add_argument("--n_cands", "-c", type=int, default=5)
    parser.add_argument("--mode", type=str, choices=["equal", "ratio"], default="equal")
    parser.add_argument("--missing_rate", type=float, default=0.0)
    parser.add_argument("--max_item_num", type=int, default=8)
    args = parser.parse_args()

    main(args)