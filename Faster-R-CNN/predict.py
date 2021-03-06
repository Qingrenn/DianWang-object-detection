import os
import json
import time

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import numpy as np

from PIL import Image, ExifTags

from torchvision import transforms
from my_log import logInit
from network_files import FasterRCNN
from backbone import resnet50_fpn_backbone
from my_dataset import DwDataset
from train_utils import get_coco_api_from_dataset, CocoEvaluator

logger = logInit("buildDataset")

class TestDataset(Dataset):

    def __init__(self, root_path, transforms):
        self.root_path = root_path
        self.transforms = transforms
        self.annotations_path = os.path.join(self.root_path,'3_testa_user.csv')
        with open(self.annotations_path, 'r') as f:
            self.img_list = [l.strip() for l in f.readlines()[1:]]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image_id = index
        img_path = os.path.join(self.root_path, self.img_list[index])
        image = Image.open(img_path)
        
        # rotate original image 
        val2rotate = {3:180, 6:270, 8:90}
        try:
            for k,v in image._getexif().items():
                if k == 274 and v !=1 :
                    image = image.rotate(val2rotate[v], expand=True)
                    break
        except Exception:
            logger.warning("{} no items ".format(img_path))

        if image.format != "JPEG":
            logger.warning("{} format not JPEG, is {}".format(img_path, image.format))
            image = image.convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)
            
        return image, image_id
    
    @staticmethod
    def collate_fn(batch):
            return tuple(zip(*batch))

def create_model(num_classes):
    backbone = resnet50_fpn_backbone()
    # model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes,box_score_thresh=0.1,box_nms_thresh={1:0.2 ,2:0.1, 3:0.2, 4:0.1}, image_mean=[0.5000, 0.5051, 0.4928], image_std=[0.2478, 0.2489, 0.2645])

    return model

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main(parser_data):

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=5)

    # load train weights
    train_weights = "./save_weights/resNetFpn-model-14.pth"
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)

    # read class_indict
    label_json_path = './dianwang_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}

    # load images
    data_transform = transforms.Compose([transforms.ToTensor()])
    tst_dataset = TestDataset(parser_data.root_path, data_transform)
    tst_dataloader = DataLoader(tst_dataset, 
                                batch_size=parser_data.batch_size,
                                shuffle=False,
                                collate_fn=tst_dataset.collate_fn)
    
    cpu_device = torch.device("cpu")

    model.eval()
    with torch.no_grad():
        for image, image_id in tqdm(tst_dataloader, desc="predict..."):
            image = list(img.to(device) for img in image)

            outputs = model(image)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

            for i in range(len(outputs)):
                jsontext = {'image_id': image_id[i], 'objs':[]}
                for j in range(len(outputs[i]["labels"])):
                    bbox, label, score = outputs[i]["boxes"][j], outputs[i]["labels"][j], outputs[i]["scores"][j]
                    obj_dict = {"label":label.numpy().tolist(), "bbox":bbox.numpy().tolist(), "score":score.numpy().tolist()}
                    jsontext["objs"].append(obj_dict)
                jsondata = json.dumps(jsontext, indent=4, separators=(',', ': '))
                with open("./test_results/img" + str(image_id[i]) + ".json",'w') as f:
                    f.write(jsondata)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root_path', default='/home/lab/Python_pro/Tianchi/TestDataset')
    parser.add_argument('--batch_size', default=4)

    args = parser.parse_args()
    main(args)
