from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import time

config_file = '../configs/my_custom_config.py'
checkpoint_file = '../WorkHome/epoch_20.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

images_dir = '/home/lab/Python_pro/Tianchi/DianWang-object-detection/Cascade-RPN/Coco-Dataset/images/val'
images_path = os.listdir(images_dir)

results_dir = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

print("inference begin ...")
f = open(os.path.join(results_dir, 'imagelist.txt'), 'w')
for idx, image_path in enumerate(images_path):
    image_path = os.path.join(images_dir, image_path)
    f.write(str(idx) + ": " + image_path + "\n")
    print(str(idx) + ": " + image_path)
    result = inference_detector(model, image_path)
    model.show_result(image_path, result, out_file=os.path.join(results_dir, 'result' + str(idx) + '.png'))
    if idx > 4:
        break
f.close()
print("inference end ...")
