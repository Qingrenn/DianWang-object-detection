import json
import os
from datetime import datetime
from tqdm import tqdm

def get_iou(box1, box2):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    x1,y1 = list(map(lambda b1,b2:max(b1,b2), box1[:2], box2[:2]))
    x2,y2 = list(map(lambda b1,b2:min(b1,b2), box1[2:], box2[2:]))
    if x1 >= x2 or y1 >= y2:
        return 0
    else:
        inter = (x2-x1) * (y2-y1)
        iou = inter / (area1 + area2 - inter)
    return iou

def get_ovr(big, small):
    area1 = (big[2] - big[0]) * (big[3] - big[1])
    area2 = (small[2] - small[0]) * (small[3] - small[1])
    x1,y1 = list(map(lambda b1,b2:max(b1,b2), big[:2], small[:2]))
    x2,y2 = list(map(lambda b1,b2:min(b1,b2), big[2:], small[2:]))
    if x1 >= x2 or y1 >= y2:
        return 0
    else:
        inter = (x2-x1) * (y2-y1)
        iou = inter / area2
    return iou

def reinfer(img_info):
    # img_info: {"image_id":int, "objs":[{"label":int, "box":[int,], "score":int},]}

    objs = img_info["objs"]
    
    remove_idxs = []

    bage_idxs = []
    offground_idxs = []
    ground_idxs = []
    safebelt_idxs = []

    for i, obj in enumerate(objs):
        if obj["label"] == 1:
            bage_idxs.append(i)
        elif obj["label"] == 2:
            offground_idxs.append(i)
        elif obj["label"] == 3:
            ground_idxs.append(i)
        elif obj["label"] == 4:
            safebelt_idxs.append(i)
        
        # 删除低分目标
        if obj["score"] < 0.4:
            remove_idxs.append(i)
    
    # offground ground 互斥
    if len(offground_idxs) != 0 and len(ground_idxs) !=0:
        for offground_idx in offground_idxs:
            for ground_idx in ground_idxs:
                offground_box = objs[offground_idx]["bbox"]
                ground_box = objs[ground_idx]["bbox"]
                iou = get_iou(offground_box, ground_box)
                if iou > 0.5 :
                    remove_idxs.append(ground_idx if objs[offground_idx]["score"] > objs[ground_idx]["score"] else offground_idx)

    # 防止 offground目标框重叠
    if len(offground_idxs) > 1:
        for i in range(0, len(offground_idxs)):
            box1 = objs[offground_idxs[i]]["bbox"]
            # area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            for j in range(i + 1, len(offground_idxs)):
                box2 = objs[offground_idxs[j]]["bbox"]
                # area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                if get_ovr(box1, box2) > 0.7:
                    remove_idxs.append(offground_idxs[j])
                elif get_ovr(box2, box1) > 0.7:
                    remove_idxs.append(offground_idxs[i])
    
    # 删除重复对象
    if len(remove_idxs) != 0:
        remove_idxs = set(remove_idxs)
        objs = [objs[i] for i in range(len(objs)) if i not in remove_idxs]
        
        bage_idxs = []
        offground_idxs = []
        ground_idxs = []
        safebelt_idxs = []

        # 重新编号
        for i, obj in enumerate(objs):
            if obj["label"] == 1:
                bage_idxs.append(i)
            elif obj["label"] == 2:
                offground_idxs.append(i)
            elif obj["label"] == 3:
                ground_idxs.append(i)
            elif obj["label"] == 4:
                safebelt_idxs.append(i)
    
    new_objs = []
    
    people_idxs = ground_idxs + offground_idxs

    # bage + offground/ground = guarder
    if len(bage_idxs) != 0 and len(people_idxs) != 0:
        for bage_idx in bage_idxs:
            for people_idx in people_idxs:
                bage_box = objs[bage_idx]["bbox"]
                people_box = objs[people_idx]["bbox"]
                iou = get_ovr(people_box, bage_box)
                if iou > 0.5 and objs[bage_idx]["score"] > 0.4:
                    guarder = {"category_id":1, "bbox":people_box, "score":objs[people_idx]["score"]}
                    new_objs.append(guarder)
    
    # safebelt + offground =  safebeltperson
    if len(safebelt_idxs) != 0 and len(offground_idxs) !=0:
        for safebelt_idx in safebelt_idxs:
            for offground_idx in offground_idxs:
                safebelt_box = objs[safebelt_idx]["bbox"]
                offground_box = objs[offground_idx]["bbox"]
                iou = get_iou(safebelt_box, offground_box)
                if iou > 0.01 and objs[safebelt_idx]["score"] > 0.4:
                    safebeltperson = {"category_id":2, "bbox":offground_box, "score":objs[offground_idx]["score"]}
                    new_objs.append(safebeltperson)

    # safebelt + ground =  safebeltperson
    if len(safebelt_idxs) != 0 and len(ground_idxs) !=0:
        for safebelt_idx in safebelt_idxs:
            for ground_idx in ground_idxs:
                safebelt_box = objs[safebelt_idx]["bbox"]
                ground_box = objs[ground_idx]["bbox"]
                iou = get_iou(ground_box, safebelt_box)
                if iou > 0.2 and objs[safebelt_idx]["score"] > 0.5:
                    safebeltperson = {"category_id":2, "bbox":ground_box, "score":objs[ground_idx]["score"]}
                    new_objs.append(safebeltperson)

    # off_ground
    if len(offground_idxs) != 0:
        for offground_idx in offground_idxs:
            if objs[offground_idx]["score"] > 0.4:
                offgroundperson = {"category_id":3, "bbox":objs[offground_idx]["bbox"], "score":objs[offground_idx]["score"]}
                new_objs.append(offgroundperson)

    img_info["objs"] = new_objs

    return img_info


def main(parser_data):

    # 用于可视化的json文件
    if not parser_data.submit and not os.path.exists("./reinfer_results"):
        os.mkdir("./reinfer_results")
    
    # 提交的结果
    if parser_data.submit:
        now = datetime.now().time().strftime("%H_%M")
        result_name = "result" + now + ".json"
        wf = open(result_name, "w")
    
    file_list = os.listdir(parser_data.pred_path)
    jsondata_list = []
    
    for file in tqdm(file_list, desc="infer..."):
        file_path = os.path.join(parser_data.pred_path, file.strip())
        img_info = {}
        with open(file_path) as f:
            img_info = json.load(f)
            img_info = reinfer(img_info)
            image_id = img_info["image_id"]
            
            # submit
            if parser_data.submit:
                for obj in img_info["objs"]:
                    jsondata = {"image_id":image_id, "category_id":obj["category_id"], "bbox":[round(i) for i in obj["bbox"]], "score":round(obj["score"], 3)}
                    jsondata_list.append(jsondata)
            else: # used for drawing
                jsontext = json.dumps(img_info, indent=4, separators=(',', ': '))
                with open('./reinfer_results/img' + str(image_id) + '.json', 'w') as f:
                    f.write(jsontext)
    
    if parser_data.submit:
        json.dump(jsondata_list, wf)
        wf.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--pred_path', default='./test_results', help='predicted result folder')
    parser.add_argument('--submit', default=True, help="choose output format")

    args = parser.parse_args()
    
    assert os.path.exists(args.pred_path), "[ERROR] can not find {}".format(args.data_path) 

    main(args)