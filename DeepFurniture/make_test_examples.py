import numpy as np
import glob
import gzip
import json
import pickle
import pathlib
import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union
import os

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
):
    print("Make test dataset.")
    test, test_y, test_z, test_id = test_all
    np.random.seed(seed)

    test_examples = []
    for i in range(len(test)):
        example = {}

        lst = np.delete(np.arange(len(test)), i)
        others = np.random.choice(lst, n_comb - 1, replace=False).tolist()
        target = [i] + others

        setX_items, setY_items = [], []
        setX_ids, setY_ids = [], [] #image id
        query_set_ids = [] #scene id
        for j in target:
            #items = [str(item["item_id"]) for item in test[j]["items"]]
            items = test[j]
            #items = np.array(items)
            ids = test_id[j]

            y_size = len(items) // 2

            xy_mask = [True] * (len(items) - y_size) + [False] * y_size
            xy_mask = np.random.permutation(xy_mask)
            setX_items.extend(items[xy_mask].tolist())
            setY_items.extend(items[~xy_mask].tolist())
            setX_ids.extend(ids[xy_mask].tolist())
            setY_ids.extend(ids[~xy_mask].tolist())
            #query_set_ids.append(test[j]["set_id"])
            query_set_ids.append(test_y[j])

        example["query"] = setX_items
        example["query_set_ids"] = query_set_ids
        example["query_set_images_ids"] = setX_ids

        answers = [setY_items]
        answers_set_images_ids = [setY_ids]
        answers_set_ids = [query_set_ids]
        for j in range(n_cands - 1):
            lst = np.delete(np.arange(len(test)), target)
            negatives = np.random.choice(lst, n_comb, replace=False).tolist()
            assert len(set(target) & set(negatives)) == 0
            target += negatives  # avoid double-selecting

            setY_items = []
            setY_ids = [] #image id
            setY_set_ids = [] #scene id
            for k in negatives:
                #items = [str(item["item_id"])for item in test[k]["items"]]
                items = test[k]
                ids = test_id[k]

                # インデックスを使って画像とIDを同じ順序で並び替える
                perm_indices = np.random.permutation(items.shape[0])
                items = items[perm_indices]
                ids = ids[perm_indices]
                y_size = len(items) // 2
                setY_items.append(items[:y_size])
                setY_ids.append(ids[:y_size])

                setY_set_ids.append(test_y[k])
            answers.extend(setY_items)
            answers_set_images_ids.extend(setY_ids)
            answers_set_ids.extend(setY_set_ids)

        example["answers"] = answers
        example["answers_set_images_ids"] = answers_set_images_ids
        example["answers_set_ids"] = answers_set_ids

        test_examples.append(example)

    with open(
        path / f"test_examples_ncomb_{n_comb}_ncands_{n_cands}.json", "w"
    ) as f:
        json.dump(convert_ndarray_to_list(test_examples), f, indent=2)

    return f"test_examples_ncomb_{n_comb}_ncands_{n_cands}"


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
        with open(output_dir / f"{example_id}.pkl", "wb") as f:
            pickle.dump(example_features, f)
            pickle.dump(example_ids, f)
            pickle.dump(example_id, f)

    assert len(glob.glob(str(output_dir / "*"))) == len(test_examples), "unmatched case"

    return


def main(args):
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
    json_name = make_test_examples((test_x, test_y, test_z, test_id), path=output_root, n_comb=1, n_cands=args.n_cands, seed=args.split)
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

    args = parser.parse_args()

    main(args)
