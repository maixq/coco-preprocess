import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import cv2
import numpy as np

import os
import cv2
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import json
from PIL import Image
import cv2 
import math
from car_detect import panel_detector

panel_classes = {
                'car_license_plate': 0,
                'front_bonnet': 1,
                'front_bumper': 2,
                'front_left_bonnet': 3,
                'front_left_bumper': 4,
                'front_left_door': 5,
                'front_left_fender': 6,
                'front_left_headlight': 7,
                'front_left_side_mirror': 8,
                'front_right_bonnet': 9,
                'front_right_bumper': 10,
                'front_right_door': 11,
                'front_right_fender': 12,
                'front_right_headlight': 13,
                'front_right_side_mirror': 14,
                'rear_bonnet': 15,
                'rear_bumper': 16,
                'rear_left_bonnet': 17,
                'rear_left_bumper': 18,
                'rear_left_door': 19,
                'rear_left_fender': 20,
                'rear_left_headlight': 21,
                'rear_left_side_panel': 22,
                'rear_right_bonnet': 23,
                'rear_right_bumper': 24,
                'rear_right_door': 25,
                'rear_right_fender': 26,
                'rear_right_headlight': 27,
                'rear_right_side_panel': 28
            }

def display_image(title, kps,  image):
    print(title)
    img = cv2.cvtColor(image, cv2. COLOR_BGR2RGB)
    img = kps.draw_on_image(img, size=10)
    plt.figure(figsize=(30,20))
    plt.imshow(img)
    plt.title(title)
    plt.show()

# dir_path = '/Users/maixueqiao/Downloads/Project/cropped_images/front_rear_images/'
json_path = '/Users/maixueqiao/coco_split/batch_2_dedup_filter/annotations/instance.json'
dir_path = '/Users/maixueqiao/coco_split/batch_2_dedup_filter/images/'

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

# json_path = '/Users/maixueqiao/coco_split/one_class_data/annotations/single_class.json'

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

def get_panel_cor(img_path, detector):
    # img_path = '/Users/maixueqiao/Desktop/output/wave1-sample1n2/'+sample[1]
    input_img = np.array(Image.open(img_path).convert('RGB')) 
    crop_boxes= detector.predict(input_img)
    
    return crop_boxes


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

def get_annotation_lists(im_id, image, cropped, ht, wd, ct):
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
            img_id = ct
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
    json_path = '/Users/maixueqiao/coco_split/batch_2_dedup_filter/annotations/instance.json'
    img_dir = '/Users/maixueqiao/coco_split/batch_2_dedup_filter/images/'
    dst = '/Users/maixueqiao/Desktop/cropped/images/'

    sample = os.listdir('/Users/maixueqiao/coco_split/batch_2_dedup_filter/images/')
    info, licenses, images, annotations, categories = read_coco(json_path)
    ann_list = []
    im_list = []
    count = 0
    for i, im in enumerate(images[:3]):
        panel_pt_dic = {}
        for x, sam in enumerate(sample):
            if im['file_name'] == sample[x]:
                im_id = im['id']
                img_path = img_dir+sample[x]
                image = load_image(img_path)
                ht, wd = image.shape[0], image.shape[1]
                crop_bboxes, pred_classes = get_panel_cor(img_path, panel_detector)

                for box_idx in range(len(crop_bboxes.tolist())):
                    img_d = {
                        'license': '',
                        'file_name': '',
                        'url': '',
                        'height': '',
                        'width': '',
                        'date_captured': '',
                        'id': ''
                    }

                    crop_panel = [k for k,v in panel_classes.items() if v == pred_classes[box_idx]][0]
                    if (crop_panel.split('_')[-1] == 'headlight') or (crop_panel.split('_')[-1] == 'plate') :
                        pass
                    else:
                        crop_pts = crop_bboxes[box_idx]
                        crop_cor = [math.ceil(x) for x in crop_pts]
                        crop_img = Image.fromarray(image).crop((crop_pts[0], crop_pts[1], crop_pts[2], crop_pts[3]))
                        crop_w, crop_h =  crop_img.size
                        
                        if os.path.exists(dst+im['file_name']):
                            print('Already exist')
                        else:
                            img_d['file_name'] = crop_panel + '_' + im['file_name']
                            img_d['height'] = crop_h
                            img_d['width'] = crop_w
                            img_d['id'] = count
                            im_list.append(img_d)
                            crop_img = cv2.cvtColor(np.array(crop_img), cv2.COLOR_BGR2RGB)
                            Image.fromarray(crop_img).save(dst+'{}_{}'.format(crop_panel, im['file_name']))
                            annotation_per_img, img_aug = get_annotation_lists(im_id, image, crop_cor, ht, wd, count)
                            print(annotation_per_img)
                            ann_list.extend(annotation_per_img)
                            count+=1
    
    output_dir = '/Users/maixueqiao/Desktop/cropped/annotations/crop.json'
    save_coco(output_dir, info, licenses, im_list, ann_list, categories)
