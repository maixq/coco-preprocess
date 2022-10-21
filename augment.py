import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import cv2
import numpy as np
import pandas as pd
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import os
import ast
import random
import matplotlib.pyplot as plt
import json
from PIL import Image
import cv2 
from car_detect import car_detector

def display_image(title, kps,  image):
    print(title)
    img = cv2.cvtColor(image, cv2. COLOR_BGR2RGB)
    img = kps.draw_on_image(img, size=10)
    plt.figure(figsize=(30,20))
    plt.imshow(img)
    plt.title(title)
    plt.show()

dir_path = '/Users/maixueqiao/Downloads/Project/cropped_images/front_rear_images/'

def load_image(filename):
    path = os.path.join(dir_path, filename)
    image = cv2.imread(path)
    return image

def augment_image(image, keypoints, seed, cropped, ht, wd):
    '''
    augment image and keypoints
    INPUT:
        image: original image
        keypoints: keypoints of panel
    OUTPUT:
        image_aug: image after augmentation
        kps_aug: keypoints after augmentation
    '''
    # print('Augmentation ===>')
    # print('Seed', seed)
    ia.seed(seed)
    cropx, cropy, w, h = cropped[0], cropped[1], cropped[2] , cropped[3] 
    top, right, bottom, left = cropy, wd-w, ht-h, cropx
    # rotation_value = random.randint(-10,10) 
    seq = iaa.Sequential([
    iaa.Multiply((1.2, 1.2)), # change brightness, doesn't affect keypoints
    iaa.Crop(px=((top), (right),(bottom),(left)), keep_size=False)
    # iaa.Crop(px=((cropped[0], cropped[1]), (cropped[0]+cropped[2], cropped[1]+cropped[3])), keep_size=False),
    ])
    
    # Augment keypoints and images.
    image_aug, kps_aug = seq(image=image,  keypoints=keypoints)
    return image_aug, kps_aug



def convert_keypoints2coordinates(aug_keypoints):
    '''
    convert keypoints to xy coordinates
    INPUT:
        keypoints: new keypoints after augmentation
    OUTPUT:
        coordinates: xy coordinates [x,y]
    '''
    # print('Converting kps to coor ===>')
    coordinates = []
    for i in range(len(aug_keypoints.keypoints)):
        newx = aug_keypoints.keypoints[i].x
        newy = aug_keypoints.keypoints[i].y
        coordinates.append([newx,newy])
    return coordinates

json_path = '/Users/maixueqiao/coco_split/one_class_data/annotations/single_class.json'

def read_coco(json_path):
    with open (json_path, 'r') as f:
        coco = json.load(f)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)
        print(number_of_images)
    return info, licenses, images, annotations, categories

def get_crop_cor(img_path, car_detector):
    # img_path = '/Users/maixueqiao/Desktop/output/wave1-sample1n2/'+sample[1]
    input_img = np.array(Image.open(img_path).convert('RGB')) 
    crop_img, cropped_image= car_detector.predict(input_img)
    
    return crop_img, cropped_image


def convert_coordinates2keypoints(coordinates):
    '''
    convert xy coordinates to keypoints which required by Keypoints on image
    INPUT:
        coordinates: xy coordinates of each panel 
    OUTPUT:
        keypoints: array of keypoints 
    '''
    # print('Converting coordinates to keypoints ==>')
    keypoints = []
    for coor in coordinates:
        # print('coordinates: ', coor)
        x = coor[0]
        y = coor[1]
        keypoints.append(Keypoint(x=x, y=y))
    # print('keypoints: ', keypoints)
    return keypoints

# all_keypoints = []
# Get all the annotations for the specified image.
from pycocotools.coco import COCO
coco_annotation = COCO(annotation_file=json_path)

def get_annotation_lists(im_id, image, cropped, ht, wd):
    seed = 42
    ann_ids = coco_annotation.getAnnIds(imgIds=[im_id], iscrowd=None)
    anns = coco_annotation.loadAnns(ann_ids)
    ann_per_img = []

    print(f"Annotations for Image ID {im_id}:")
    for instance in range(len(anns)):
        cropped_annotation = {}
        segmentation_list = []
        # print('Instance: ', instance)
        for polygon in range(len(anns[instance]['segmentation'])):
            # print('Polygon: ', polygon)
            # print('BBOX: ', anns[instance]['bbox'])
            per_damage = anns[instance]['segmentation'][polygon]
            bbox = anns[instance]['bbox']
            iscrowd = anns[instance]['iscrowd']
            area = anns[instance]['area']
            img_id = anns[instance]['image_id']
            cat_id = anns[instance]['category_id']
            index = anns[instance]['id'] 
            coordinate_list = zip(per_damage[::2], per_damage[1::2])
            bb_x, bb_y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            keypoints = convert_coordinates2keypoints(coordinate_list)
            bbox_points = convert_coordinates2keypoints([[bb_x, bb_y], [bb_x+w, bb_y+h]])
            # all_keypoints.extend(keypoints)
            kpsoi = KeypointsOnImage(keypoints, shape=image.shape)
            bb_pts = KeypointsOnImage(bbox_points, shape=image.shape)
            
            image_aug, keypoints_aug= augment_image(image, kpsoi, seed, cropped, ht, wd)
            image_aug, bbpoints_aug= augment_image(image, bb_pts, seed, cropped, ht, wd)

            coordinates_aug = convert_keypoints2coordinates(keypoints_aug)
            # print(keypoints_aug)
            # print(coordinates_aug)
            bbox_aug = convert_keypoints2coordinates(bbpoints_aug)
            crop_seg = [x for pair in coordinates_aug for x in pair]
            # print(crop_seg)
            # print(per_damage)
            # print(crop_seg)
            segmentation_list.append(crop_seg)
            new_bb = [x for pair in bbox_aug for x in pair] 
            new_bb_x, new_bb_y, new_w, new_h = new_bb[0], new_bb[1], new_bb[2]-new_bb[0], new_bb[3]-new_bb[1]
            new_bbox = [new_bb_x, new_bb_y, new_w, new_h]
            # print('New BBOX: ', new_bbox)
            # print('Old seg: ', per_damage)
            # print('New seg: ', crop_seg)

        # write into dictionary
            # print("SEG LIST: ", segmentation_list)
            cropped_annotation['segmentation'] = segmentation_list
            cropped_annotation['area'] = area
            cropped_annotation['iscrowd'] = iscrowd
            cropped_annotation['image_id'] = img_id
            cropped_annotation['bbox'] = new_bbox
            cropped_annotation['category_id'] = cat_id
            cropped_annotation['id'] = index
            # print("crop annotation: ", cropped_annotation)
        ann_per_img.append(cropped_annotation)
    # kpsoi = KeypointsOnImage(all_keypoints, shape=image.shape)
        # display_image(img_path, keypoints_aug, image_aug)
        # display_image(img_path, bbpoints_aug, image_aug) 
        # display_image(img_path, keypoints_aug, image_aug)
        
    return ann_per_img, image_aug

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 
            'categories': categories},
             coco, indent=2, sort_keys=False)

if __name__ == "__main__":
    json_path = '/Users/maixueqiao/coco_split/one_class_data/annotations/single_class.json'
    img_dir = '/Users/maixueqiao/coco_split/one_class_data/images/'
    dst = '/Users/maixueqiao/Downloads/Project/cropped_images/single_class_crop/'

    one_class_ims = os.listdir('/Users/maixueqiao/coco_split/one_class_data/images')
    sample = one_class_ims
    info, licenses, images, annotations, categories = read_coco(json_path)
    ann_list = []
    count = 0
    for i, im in enumerate(images):
        for x, sam in enumerate(sample):
            if im['file_name'] == sample[x]:
                im_id = im['id']
                img_path = img_dir+sample[x]
                image = load_image(img_path)
                
                ht, wd = image.shape[0], image.shape[1]
                cropped_cor, crop_img = get_crop_cor(img_path, car_detector)
                if os.path.exists(dst+im['file_name']):
                    print('Already exist')
                    pass
                else:
                    # save_im = cv2.cvtColor(img, cv2. COLOR_BGR2RGB)
                    
                    crop_img.save(dst+'{}'.format(im['file_name']))
                annotation_per_img, img_aug = get_annotation_lists(im_id, image, cropped_cor, ht, wd)
                # save_im.save(dst+'{}'.format(im['file_name']), save_im)
                # cv2.imwrite(dst+'{}').format(im['filename'])
                ann_list.extend(annotation_per_img)
                count+=1
                print(count)

    output_dir = '/Users/maixueqiao/Downloads/Project/cropped_images/cropped_jsons/single_class_crop.json'
    save_coco(output_dir, info, licenses, images, ann_list, categories)