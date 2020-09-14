import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split


def parse_json(filename):
    """
    Extract landmarks coordinates from file with annotations
    :param filename: absolute path to json file
    :return: ndarray of landmarks with shape (6,2,2),
            indicator variable which defines if malformed coordinates were observed in json
    """
    landmarks = np.zeros((6, 2, 2))
    target_metrics = ["C", "CTI_A", "CTI_B", "MOOR_A", "LP_A", "LP_B"]
    class_position = {"C": 0, "CTI_A": 1, "CTI_B": 2, "MOOR_A": 3, "LP_A": 4, "LP_B": 5}
    is_line = True
    with open(filename, "r") as ann:
        lines = json.load(ann)["objects"]
        for line in lines:
            if line["classTitle"] in target_metrics:  # filter redundant metrics
                landmark = np.array(line["points"]["exterior"])
                if len(landmark) == 2:  # only use line metrics
                    landmarks[class_position[line["classTitle"]]] = landmark
                elif len(landmark) == 3:  # finger slip, let's try to fix
                    is_line = False
                    landmarks[class_position[line["classTitle"]]] = landmark[:2]
            else:
                print(line["classTitle"], filename)
    landmarks = landmarks.reshape(-1, 2)
    for i in range(
            0, landmarks.shape[0], 2
    ):  # swap point if the righthand side point appears on the left from
        if landmarks[i][0] > landmarks[i + 1][0]:  # lefthand side point
            landmarks[[i, i + 1]] = landmarks[[i + 1, i]]
    landmarks = landmarks.reshape(len(class_position), 2, 2)

    return landmarks, is_line


def is_valid_json(filename):
    """
    JSON file is considered as invalid if at least one coordinate is equal to zero.
    This event usually means that some lines are not identified on image or
    the image was not annotated at all
    :param filename:
    :return:
    """
    return parse_json(filename)[0].all()


def is_valid_line(filename):
    return parse_json(filename)[1]


def get_width(path):
    with Image.open(path) as img:
        width, height = img.size
    return width


def get_height(path):
    with Image.open(path) as img:
        width, height = img.size
    return height


def get_indices(ann_path=None, landmarks=None):
    if ann_path is not None:
        landmarks, _ = parse_json(ann_path)
    if landmarks is None:
        print('One of the params must be not None', file=sys.stderr)
        raise AttributeError
    landmarks = np.array(landmarks).reshape(-1, 2, 2)

    class_position = {0: "C", 1: "CTI_A", 2: "CTI_B", 3: "MOOR_A", 4: "LP_A", 5: "LP_B"}
    distacnes = {}

    for class_idx, line in enumerate(landmarks):
        point1 = line[0]
        point2 = line[1]
        distacnes[class_position[class_idx]] = euclidean(point1, point2)
    if distacnes["C"] == 0: return 0, 0, 0
    cti = (distacnes["CTI_A"] + distacnes["CTI_B"]) / distacnes["C"]
    moor = 2 * distacnes["MOOR_A"] / distacnes["C"]
    lupi = (distacnes["LP_A"] + distacnes["LP_B"]) / distacnes["C"]
    return cti, moor, lupi


def get_moor_idx(ann_path):
    _, moor, _ = get_indices(ann_path=ann_path)
    return moor


def get_cti_idx(ann_path):
    cti, _, _ = get_indices(ann_path=ann_path)
    return cti


def get_lupi_idx(ann_path):
    _, _, lupi = get_indices(ann_path=ann_path)
    return lupi


def wrap_points_array_into_dict(a):
    class_position = {0: "C", 1: "CTI_A", 2: "CTI_B", 3: "MOOR_A", 4: "LP_A", 5: "LP_B"}
    d = {}
    for i, line in enumerate(a.tolist()):
        d[class_position[i]] = line
    return str(d)


def create_csv(root, tag="init"):
    """
    structure
    root---subdir1---img--image1.jpg
        |         |--ann--image1.jpg.json
        |--subdir2---...
    """
    ret = []
    df = pd.DataFrame(columns=["imgPath", "annPath"])
    hash_keys = dict()
    duplicates = []
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath.endswith("ann"):
            ret += filenames
            for ann_name in filenames:
                img_name = dirpath[:-3] + "img" + os.sep + ann_name[:-5]
                # check for duplicates
                with open(os.path.abspath(img_name), "rb") as f:
                    filehash = hashlib.md5(f.read()).hexdigest()
                if filehash not in hash_keys:
                    hash_keys[filehash] = img_name
                    df = df.append(
                        {
                            "imgPath": img_name,
                            "annPath": os.path.join(dirpath, ann_name),
                        },
                        ignore_index=True,
                    )
                else:
                    duplicates.append((img_name, hash_keys[filehash]))
    print("{} duplicate images were found".format(len(duplicates)))
    df["width"] = df["imgPath"].apply(get_width)
    df["height"] = df["imgPath"].apply(get_height)
    df.sort_values(["width"], inplace=True)
    df["validAnnotation"] = df["annPath"].apply(is_valid_json)
    df["validLineDescription"] = df["annPath"].apply(is_valid_line)
    df.to_csv(os.path.join(root, "about{}.csv".format(tag)), index=False)
    df = df[df["validAnnotation"] == True]
    df.to_csv(os.path.join(root, "dataset{}.csv".format(tag)), index=False)
    df["isPathological"] = df["imgPath"].apply(
        lambda x: False if x.split(root)[1][1] == "n" else True
    )
    df["CTI"] = df["annPath"].apply(get_cti_idx)
    df["MOOR"] = df["annPath"].apply(get_moor_idx)
    df["LP"] = df["annPath"].apply(get_lupi_idx)

    df['target'] = df["annPath"].apply(
        lambda x: wrap_points_array_into_dict(parse_json(x)[0])
    )
    # # write points as well
    # class_position = {0: "C", 1: "CTI_A", 2: "CTI_B", 3: "MOOR_A", 4: "LP_A", 5: "LP_B"}
    # l0 = {0: "left", 1: "right"}
    # l1 = {0: "x", 1: "y"}
    # for i in class_position:
    #     for j in l0:
    #         for k in l1:
    #             df["_".join([class_position[i], l0[j], l1[k]])] = df["annPath"].apply(
    #                 lambda x: parse_json(x)[0][i, j, k]
    #             )
    #
    # # Assert that points were sorted right
    # for i in class_position:
    #     assert (
    #         df.loc[:, "_".join([class_position[i], "left", "x"])]
    #         < df.loc[:, "_".join([class_position[i], "right", "x"])]
    #     ).all()

    df.to_csv(os.path.join(root, "dataset{}.csv".format(tag)), index=False)
    return df


def split(root):
    """
    Perform stratified splitting of dataset. SEED # 17 used to create same
    :param root: path to the dataset root folder
    :return: Creates 3 csv files with train/val/test split. 60/20/20%
    """
    # is is assumed that 'dataset.csv' is produced by 'create_csv' function
    csv_path = os.path.join(root, 'dataset.csv')
    df = pd.read_csv(csv_path)
    print("Dataset length: {}".format(len(df)))
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        shuffle=True,
        test_size=0.2,
        random_state=17,
        stratify=df.loc[:, "isPathological"],
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        shuffle=True,
        test_size=0.25,
        random_state=17,
        stratify=df.loc[train_idx, "isPathological"],
    )
    print(len(train_idx), "-train, ", len(val_idx), "-val, ", len(test_idx), "-test")
    train_df = df.iloc[train_idx, :]
    val_df = df.iloc[val_idx, :]
    test_df = df.iloc[test_idx, :]
    train_df.to_csv(
        os.path.join(root, "train.csv"), index=False
    )
    val_df.to_csv(
        os.path.join(root, "val.csv"), index=False
    )
    test_df.to_csv(
        os.path.join(root, "test.csv"), index=False
    )


if __name__ == "__main__":
    root_folder = "/home/semyon/cardiomethry/ChestXrayIndex"
    if len(sys.argv) > 1:
        root = sys.argv[1]

    create_csv(root_folder, tag="")
    split(root_folder)
