import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from pandas.core.indexes import category
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm

dataset_prefix = "/home/lab/Python_pro/Tianchi/Dataset"
csv_path = "/home/lab/Python_pro/Tianchi/Dataset/3train_rname.csv"

label2num = {
    "监护袖章(红only)":1,
    "offground":2,
    "ground":3,
    "safebelt":4
}

def get_area(bbox):
    x1, y1, x2, y2 = bbox
    assert x2>x1 and y2>y1
    return (x2-x1)*(y2-y1)

def get_box(bbox):
    x1, y1, x2, y2 = bbox
    assert x2>x1 and y2>y1
    return [x1, y1, x2-x1, y2-y1]

def csv2coco(df_img_path, df_img_ann):
    true = True
    false = False
    images = []
    annotations = []

    ann_id = 0
    for img_id, img_ann in enumerate(df_img_ann):
        # 构建 image
        img = Image.open(os.path.join(dataset_prefix, df_img_path[img_id]))
        w, h = img.size
        image = {"file_name": df_img_path[img_id], 
                "height": h,
                "width": w,
                "id": img_id}
        
        # 构建 annotation
        img_ann = eval(img_ann)["items"]
        for obj in img_ann:
            label = obj["labels"]["标签"]
            bbox = obj["meta"]["geometry"]
            annotation = {"area": get_area(bbox), 
                        "iscrowd": 0, 
                        "image_id": img_id, 
                        "bbox": bbox, 
                        "category_id": label2num[label], 
                        "id": ann_id}
            ann_id += 1
            annotations.append(annotation)
        
        images.append(image)
    
    # 构建 categories
    categories = [{"id":1, "name":"bage"},
                {"id":2, "name":"offground"}, 
                {"id":3, "name":"ground"}, 
                {"id":4, "name":"safebelt"}]
    
    # 构建 instance
    instance = {"images": images, 
                "annotations": annotations, 
                "categories": categories}
    
    return instance


if __name__ == '__main__':
    # 会在目标路径下生成两个文件夹 annotations 和 images
    target_dir = "/home/lab/Python_pro/Tianchi/DianWang-object-detection/Cascade-RPN/Coco-Dataset"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    annotations_dir = os.path.join(target_dir, "annotations")
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)
    
    train_images_dir = os.path.join(target_dir, "images/train")
    val_images_dir = os.path.join(target_dir, "images/val")
    if not os.path.exists(train_images_dir):
        os.makedirs(train_images_dir)
    if not os.path.exists(val_images_dir):
        os.makedirs(val_images_dir)


    df = pd.read_csv(csv_path, header=None)
    df_img_path_ann = df.iloc[:, 4:]

    train_df_img_path_ann, val_df_img_path_ann = train_test_split(df_img_path_ann, test_size=0.2)

    # 构建训练集
    print("Build training dataset ...")
    train_img_paths = train_df_img_path_ann.iloc[:, 0]
    train_img_anns = train_df_img_path_ann.iloc[:, 1]
    target_train_img_paths = []
    for origin_path in tqdm(train_img_paths):
        target_train_img_path = os.path.join(train_images_dir, origin_path.split("/")[-1])
        target_train_img_paths.append(target_train_img_path)
        shutil.copy(os.path.join(dataset_prefix, origin_path), target_train_img_path)

    instances_train = csv2coco(target_train_img_paths, train_img_anns)
    with open(os.path.join(annotations_dir, "instances_train.json"), 'w') as f:
        json.dump(instances_train, f, ensure_ascii=False, indent=2)

    # 构建测试集
    print("Build validation dataset ...")
    val_img_paths = val_df_img_path_ann.iloc[:, 0]
    val_img_anns = val_df_img_path_ann.iloc[:, 1]
    target_val_img_paths = []
    for origin_path in tqdm(val_img_paths):
        target_val_img_path = os.path.join(val_images_dir, origin_path.split("/")[-1])
        target_val_img_paths.append(target_val_img_path)
        shutil.copy(os.path.join(dataset_prefix, origin_path), target_val_img_path)

    instances_val = csv2coco(target_val_img_paths, val_img_anns)
    with open(os.path.join(annotations_dir, "instances_val.json"), 'w') as f:
        json.dump(instances_val, f, ensure_ascii=False, indent=2)


    








    

    