import torch
import json
import os
import transforms
import numpy as np
from PIL import Image, ExifTags
from my_log import logInit
from torch.utils.data import Dataset, dataloader

logger = logInit("buildDataset")

class DwImage:

    def __init__(self, csv_str, row_id):
        para_list = csv_str.split(",",5)
        # self.sd_uid = para_list[0]
        # self.sd_add_date = para_list[1]
        # self.sd_task_id = para_list[2]
        # self.sd_batch_id = para_list[3]
        self.image_path = para_list[4]
        self.info = dict()
        
        # read json string
        label2num = {'监护袖章(红only)':1, 'offground':2, 'ground':3, 'safebelt':4}
        info_dict = json.loads(para_list[5][1:-1].replace('""', '"'))
        boxes = []
        labels = []
        for obj in info_dict["items"]:
            # [xmin, ymin, xmax, ymax]
            box = obj['meta']['geometry']

            # check data
            if  box[2] <= box[0] or box[3] <= box[1]:
                logger.warning("{} there are some bbox w/h < 0".format(self.image_path))
                continue

            boxes.append(box)
            assert obj['labels']['标签'] in label2num.keys(), "[Error] {} label {} is out of keys".format(self.image_path, obj['labels']['标签'])
            label = label2num[obj['labels']['标签']]
            labels.append(label)
        
        boxes = np.array(boxes)
        
        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = 0
            logger.warning("{} no boxes, suggest delete".format(self.image_path))
        
        self.info['boxes'] = boxes
        self.info['labels'] = labels
        self.info['area'] = area
        self.info['image_id'] = row_id

    def __str__(self):
        return "ImagePath:{}\nInfo:{}".format(self.image_path, self.target)


class DwDataset(Dataset):

    def __init__(self, datset_path, transforms, txt_name: str = "train.txt"):
        self.dataset_path = datset_path
        assert os.path.exists(self.dataset_path), "[ERROR] not found {} file".format(datset_path)

        self.imgs_path = os.path.join(self.dataset_path, "3_images")
        self.annotations_path = os.path.join(self.dataset_path, "3train_rname.csv")
        
        # preprocess
        self.transforms = transforms
        
        # Image object list
        self.img_list = []
        
        # read train.txt or val.txt file
        txt_path = os.path.join(self.dataset_path, txt_name)
        assert os.path.exists(txt_path), "[ERROR] not found {} file".format(txt_name)

        with open(txt_path) as txt:
            with open(self.annotations_path) as annotations:
                lines = annotations.readlines()
                for row in txt.readlines():
                    row = int(row.strip())
                    assert row < 2549, "[ERROR] row_index {} is out of range".format(row)
                    self.img_list.append(DwImage(lines[row-1].strip(), row-1))
                        
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, self.img_list[idx].image_path)
        img_info = self.img_list[idx].info
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

        target = dict()
        target['boxes'] = torch.as_tensor(img_info['boxes'], dtype=torch.float32)
        target['labels'] = torch.as_tensor(img_info['labels'], dtype=torch.int64)
        target['area'] = torch.as_tensor(img_info['area'], dtype=torch.float32)
        target['image_id'] = torch.as_tensor(img_info['image_id'], dtype=torch.int64)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target
        
    @staticmethod
    def collate_fn(batch):
            return tuple(zip(*batch))

    def coco_index(self, idx):
        img_path = os.path.join(self.dataset_path, self.img_list[idx].image_path)
        img_info = self.img_list[idx].info
        image = Image.open(img_path)
        data_height, data_width = image.size

        target = dict()
        target['boxes'] = torch.as_tensor(img_info['boxes'], dtype=torch.float32)
        target['labels'] = torch.as_tensor(img_info['labels'], dtype=torch.int64)
        target['area'] = torch.as_tensor(img_info['area'], dtype=torch.float32)
        target['image_id'] = torch.as_tensor(img_info['image_id'], dtype=torch.int64)
        target['iscrowd'] = torch.zeros(len(img_info['boxes']), dtype=torch.int64)
        
        return (data_height, data_width), target


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from torch.utils.data import WeightedRandomSampler, DataLoader
    from tqdm import tqdm
            
    # compose_transforms = transforms.Compose([transforms.Resize(),
    #                                             transforms.ToTensor(),
    #                                             transforms.RandomHorizontalFlip()])

    compose_transforms = transforms.Compose([transforms.ToTensor()])
    
    dw_dataset = DwDataset("/home/lab/Python_pro/Tianchi/Dataset",
                            compose_transforms,
                            "train.txt")
    dataloader = DataLoader(dataset=dw_dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=dw_dataset.collate_fn)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for images, targets in tqdm(dataloader):
        for d in range(3):
            mean[d] += images[0][d, :, :].mean()
            std[d] += images[0][d, :, :].std()
    mean.div_(len(dw_dataset))
    std.div_(len(dw_dataset))
    print("mean: {}\nstd: {}".format(mean, std))
    


        
    
        

