import json
import os
import numpy as np
from pycocotools.coco import COCO
from PIL import Image,ImageDraw
import random
import matplotlib.pyplot as plt
json_path="./512-512/annotations/instances_train2017.json"
img_path="./512-512/train2017"

#load coco data
coco=COCO(annotation_file=json_path)

#获取全部图片的index
ids=list(sorted(coco.imgs.keys()))
print("图像总数量:{}".format(len(ids)))

#获取全部类别（这里只有一个耕地类别）
coco_classes=dict([(v["id"],v["name"]) for k,v in coco.cats.items()])
# randX=random.randint(0,len(ids)-1-50)
#遍历前三张图片
for img_id in ids[0:len(ids)]:
    #根据图像id    获取全部 标签id
    ann_ids=coco.getAnnIds(imgIds=img_id)

    #根据标签ids    拿对应的标注信息
    targets=coco.loadAnns(ann_ids)
    if(len(targets)==0):
        print("该图片没有标注信息，不予展示")
        continue
    #获取第img_id个图片名
    image_name=coco.loadImgs(img_id)[0]['file_name']

    #调用PIL读取图片
    img=Image.open(os.path.join(img_path,image_name))

    #调用绘制器
    draw=ImageDraw.Draw(img)
    for target in targets:
        x,y,w,h=target["bbox"]
        poly=target["segmentation"][0]
        x1,y1,x2,y2=x,y,int(x+w),int(y+h)
        draw.rectangle((x1,y1,x2,y2))
        #绘制多边形
        draw.polygon(poly,outline="Red")
        # draw.text((x1,x2),coco_classes[target["category_id"]])

    #show 图片
    # plt.imshow(img)
    img.show()
    # img.save("label/"+image_name)
print("程序结束")
