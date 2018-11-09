import json
import numpy as np
import cv2
from numpy import array, linalg, matrix
from scipy.special import comb as nOk

import os
from pathlib import Path
from itertools import product, combinations
import argparse
import time
import argparse

def load_json(f):
    with open(f, 'r') as fp:
        return json.load(fp)


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def find_pic_path(data_path, pic_name):
    #     pic_name = 'b1c81faa-3df17267.jpg'
    for root, _, files in os.walk(data_path):
        # print(root)
        for filename in files:

            if filename == pic_name:
                if os.path.isfile(os.path.join(root, filename)):
                    return os.path.join(root, filename)
                else:
                    return 'No such file'
    return 'No such filename'


def random_color():
    return np.random.rand(3)


def Mtk(n, t, k): return t**(k)*(1-t)**(n-k)*nOk(n, k)


def get_object_attrib_in_data(data, category, timeofday='daytime'):
    pic_label_ = []
    for pic in data:
        if pic.get('attributes').get('timeofday') == timeofday:
            labels = pic.get('labels')
            label_list = [(pic.get('name'), label.get('poly2d')[0], label.get(
                'attributes'), label.get('id')) for label in labels if label.get('category') == category]
            if len(label_list):
                pic_label_.append(label_list)
    return pic_label_


def bezierM3(ts): return matrix([[Mtk(3, t, k) for k in range(4)] for t in ts])


def bezierM1(ts): return matrix([[Mtk(1, t, k) for k in range(2)] for t in ts])


def bezier_curve_pts_sampling(control_points, n_sampling=100):
    # return sampling points on bazier curve with given control points
    if len(control_points) == 4:
        return bezierM3(np.linspace(0, 1, n_sampling)) * control_points
    elif len(control_points) == 2:
        return bezierM1(np.linspace(0, 1, n_sampling)) * control_points
    else:
        raise Exception('Only support linear or cubic bezier curve.')


def return_bezier_segs(types_string):

    assert set(types_string).issubset(
        set('CL')), 'string only contains C and L'
    start_pos = 0
    check_index = start_pos + 1
    segs = []
    tries = 0
    while start_pos < len(types_string)-1 and tries < 2:
        if types_string[check_index] == 'L':
            segs.append(list(range(start_pos, check_index+1)))
            start_pos = check_index
            check_index = start_pos + 1
        else:  # check point is 'C'
            if check_index + 2 < len(types_string):
                validity = types_string[check_index +
                                        1] == 'C' and types_string[check_index+2] == 'C'
                if validity:
                    segs.append(list(range(start_pos, check_index + 3)))
                    start_pos = check_index+2
                    check_index = start_pos + 1
                else:
                    tries += 1
                    types_string = types_string[::-1]
                    start_pos = 0
                    check_index = 1
                    segs = []
            else:
                raise Exception('Invalid type string, need consecutive 3 C')
    if tries > 1:
        raise Exception('{} is invalid type string'.format(types_string))
    return segs, types_string, tries


def compound_bezier_pts_sampling(types, vertices):
    # types: string of C and L, e.g. 'CCCLLLCCCL'

    segs, types, reverse = return_bezier_segs(types)
    # reverse points order if necessary
    if reverse:
        vertices = vertices[::-1]

    seg = segs[0]
    curve_points = bezier_curve_pts_sampling(vertices[seg])
    for seg in segs[1:]:
        new_curve_seg_pts = bezier_curve_pts_sampling(vertices[seg])
        curve_points = np.append(curve_points, new_curve_seg_pts, axis=0)
    return curve_points


def curve_enclosure_area(curve_points, img_size):
    img = np.zeros(img_size, np.uint8)
    cv2.fillPoly(img, np.int_([curve_points]), 1)
    return img


def curve_length(curve):
    # inputs: ordered Nx2 np array
    length = 0
    for ii in np.arange(len(curve)-1):
        length += np.linalg.norm(curve[ii, :] - curve[ii+1, :])
    return length


def data_paths_generator(data_path):
    data_path = Path(data_path)
    subdirs_of_data_path = [x for x in data_path.iterdir() if x.is_dir()]
    label_folder_path = [x for x in data_path.iterdir(
    ) if x.is_dir() and 'labels' in str(x)][0]
    train_label_path = [x for x in label_folder_path.iterdir()
                        if 'train' in str(x)][0]
    val_label_path = [x for x in label_folder_path.iterdir()
                      if 'val' in str(x)][0]
    data_path = [x for x in data_path.iterdir() if x.is_dir()
                 and 'image' in str(x)][0]
    print('images: ', data_path)
    print('labels: ', train_label_path, val_label_path)
    return data_path, train_label_path, val_label_path


def pic_lanes_laneType_dayTime_separation(data, category, all_lane_type, timeofday):
    lane_type_day = list(product(all_lane_type, timeofday))

    pic_lane_type_day = dict(
        zip(lane_type_day, [[] for _ in range(len(lane_type_day))]))

    for time in timeofday:
        pic_lanes_before = get_object_attrib_in_data(
            data, category, time)
        for lane_type in all_lane_type:
            for pic_labels in pic_lanes_before:
                pic_labels = [label for label in pic_labels if label[2].get('laneType') == lane_type[0] and label[2].get(
                    'laneStyle') == lane_type[1] and label[2].get('laneDirection') == lane_type[2]]
                if len(pic_labels):  # if non zero
                    pic_lane_type_day[(lane_type, time)].append(pic_labels)
    return pic_lane_type_day

# curve coupling, bounding area, bounding mask


def curves_on_pic(pic_labels):
    # return list of Nx2 np array
    curves = []
    for label in pic_labels:
        _, bezier_type, need_reverse = return_bezier_segs(
            label[1].get('types'))

        vertices = np.array(label[1].get('vertices'))
        if need_reverse:
            vertices = vertices[::-1]
#         vertices.append(pts)
        curves.append(
            np.array(compound_bezier_pts_sampling(bezier_type, vertices)))
    return curves


class curve_distances():
    def __init__(self, curve1, curve2):
        self.curve1 = curve1
        self.curve2 = curve2
        self.distance_list = np.array(
            [pt_curve_distance(pt, self.curve2) for pt in self.curve1])

    def max_metric(self):
        return np.max(self.distance_list)

    def percentile_metric(self, number=80):
        # assign the percentile percentage
        return np.percentile(self.distance_list, number)


def pt_curve_distance(pt, curve):
    # pt: 1x2, curve Nx2 np arrays
    return min(np.linalg.norm(pt - curve, axis=1))


class curves_pairwise_distance():
    def __init__(self, curves):
        self.curves = curves
        self._pairs = list(combinations(range(len(self.curves)), 2))

    def max_metric(self):
        return np.array([curve_distances(c1, c2).max_metric() for c1, c2 in combinations(self.curves, 2)])

    def percentile_metric(self, number=80):
        return np.array([curve_distances(c1, c2).percentile_metric(number) for c1, c2 in combinations(self.curves, 2)])

    @property
    def pairs(self):
        return self._pairs

# def curve_pairing(curves, method, threshold=40):


def curve_pairing(curves, method, threshold=30, **kwargs):
    distances = getattr(curves_pairwise_distance(curves), method)(**kwargs)

    pairs = np.array(list(combinations(range(len(curves)), 2)))
    paired = list(range(len(curves)))
    while len(pairs):
        min_dist, min_arg = np.min(distances), np.argmin(distances)
        pair = pairs[min_arg]
#         print(min_dist)
        if min_dist > threshold:

            # print('no more grouped areas')
            break
        paired.append(tuple(pair))
        paired = list(
            filter(lambda x: not {x}.issubset(set(tuple(pair))), paired))
        update_ = [not bool(set(ii).intersection(set(pair))) for ii in pairs]
        distances = distances[update_]
        pairs = pairs[update_]
    return paired


def curves_pts_concatenation(curve1, curve2):
    min_dist = 1e5
    reverse = False
    for i, j in product((1, -1), (1, -1)):
        if np.linalg.norm(curve1[i]-curve2[j]) < min_dist:
            min_dist = np.linalg.norm(curve1[i]-curve2[j])
            reverse = i * j

    if reverse:
        return np.vstack([curve1, curve2[::-1]])
    else:
        return np.vstack([curve1, curve2])


def mask_bounding_area_by_two_curves(curve1, curve2, img_size):
    curve_points = curves_pts_concatenation(curve1, curve2)
    mask = np.zeros(img_size, np.uint8)
    cv2.fillPoly(mask, np.int_([curve_points]), 1)
    return mask


def mask_area_by_one_curve(curve, img_size, thickness=5):
    img = np.zeros(img_size, np.uint8)
    cv2.polylines(img, np.int_([curve]),
                  isClosed=False, thickness=thickness, color=1)

    return img


def timer(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time passes,  {:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds))


all_lane_type = [('single yellow', 'solid', 'parallel'),
                 ('single yellow', 'solid', 'vertical'),
                 ('single yellow', 'dashed', 'parallel'),
                 ('single yellow', 'dashed', 'vertical'),
                 ('single white', 'solid', 'parallel'),
                 ('single white', 'solid', 'vertical'),
                 ('single white', 'dashed', 'parallel'),
                 ('single white', 'dashed', 'vertical'),
                 ('double yellow', 'solid', 'parallel'),
                 ('double yellow', 'solid', 'vertical'),
                 ('double yellow', 'dashed', 'parallel'),
                 ('double yellow', 'dashed', 'vertical'),
                 ('double white', 'solid', 'parallel'),
                 ('double white', 'solid', 'vertical'),
                 ('double white', 'dashed', 'parallel'),
                 ('double white', 'dashed', 'vertical')]

timeofday = ('daytime', 'night')


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    help='The root of  dataset dir path')
parser.add_argument('--data_target', type=tuple, help='train or val', default=('train', 'val'))
parser.add_argument('--timeofday', type=tuple, help='daytime or night', default=('daytime'))
parser.add_argument('--lane_style', type=tuple, help='dashed or solid', default=('dashed', 'solid'))
parser.add_argument('--lane_type', type=tuple, help='single or double', default=('single', 'double'))
parser.add_argument('--lane_direction', type=tuple, help='vertical or parallel', default=('parallel'))

args = parser.parse_args()

root_path = args.root_path
data_target = args.data_target
timeofday = args.timeofday
allowed_lane_type = args.lane_type
allowed_lane_style = args.lane_style
allowed_lane_direction = args.lane_direction
all_lane_type = [ii for ii in all_lane_type if ii[0].split(
    ' ')[0] in allowed_lane_type]
all_lane_type = [ii for ii in all_lane_type if ii[1] in allowed_lane_style]
all_lane_type = [ii for ii in all_lane_type if ii[2] in allowed_lane_direction]

print(all_lane_type)
#
# bdd_root_path = Path(root_path)
# data_path = bdd_root_path / 'bdd100k'
data_path = Path(root_path)
lane_segs_instance_data = data_path / 'lane_binary_instances'
mkdir(lane_segs_instance_data)

segmentation_save_dir = lane_segs_instance_data / 'segments'
mkdir(segmentation_save_dir)
train_segmentation_save_dir =  segmentation_save_dir / 'train'
mkdir(train_segmentation_save_dir)
val_segmentation_save_dir = segmentation_save_dir / 'val'
mkdir(val_segmentation_save_dir)

instance_save_dir = lane_segs_instance_data / 'instances'
mkdir(instance_save_dir)  # instances_save_dir is a Path object now
train_instance_save_dir = instance_save_dir / 'train'
mkdir(train_instance_save_dir)
val_instance_save_dir = instance_save_dir / 'val'
mkdir(val_instance_save_dir)
# instances_save_dir is a Path object now
bdd_lan_train_txt = lane_segs_instance_data / 'train.txt'
bdd_lan_val_txt = lane_segs_instance_data / 'val.txt'






data_path, train_label_path, val_label_path = data_paths_generator(data_path)

label_paths = {'train': train_label_path, 'val': val_label_path}
segments_paths = {'train': train_segmentation_save_dir,
                  'val': val_segmentation_save_dir}
instance_paths = {'train': train_instance_save_dir,
                  'val': val_instance_save_dir}

save_txts = {'train': bdd_lan_train_txt, 'val': bdd_lan_val_txt}



current_data = load_json(label_paths[data_target])
print('{} json file loaded'.format(data_target))

pic_lane_type_day = pic_lanes_laneType_dayTime_separation(
    current_data, 'lane', all_lane_type, timeofday)
print('start to save')


total_in_dataset = 0
for (lane, zeit) in product(all_lane_type, timeofday):
    pic_lanes = pic_lane_type_day[(lane, zeit)]
    total_in_dataset += len(pic_lanes)
processed = 0
with open(save_txts[data_target], 'w') as fff:
    start_time = time.time()
    for iii, (lane, zeit) in enumerate(product(all_lane_type, timeofday)):
        pic_lanes = pic_lane_type_day[(lane, zeit)]
        print(lane, zeit, len(pic_lane_type_day[(lane, zeit)]))

        if len(pic_lanes):
            for ii,  pic_labels in enumerate(pic_lanes):
                if not (ii % 500):

                    timer(start_time, time.time())
                    print('We have processed {} pictures in {}-{}, dataset number  {}, {} remains in current dataset. total {} remains'.format(
                        ii, lane, zeit, iii, len(pic_lanes) - ii, total_in_dataset-processed))
                processed += 1
                pic_name = pic_labels[0][0]
                pic_path = find_pic_path(data_path, pic_name)

                img = cv2.imread(pic_path)[:, :, ::-1]
                img_size = img.shape[:2]

                curves = curves_on_pic(pic_labels)
                curve_pairs = curve_pairing(
                    curves, method='percentile_metric', threshold=30, number=80)

                area_counting = 1
                mask = np.zeros(img_size, np.uint8)

                for pair in curve_pairs:
                    if type(pair) == tuple:
                        tmp_mask = mask_bounding_area_by_two_curves(
                            curves[pair[0]], curves[pair[1]], img_size)
                        if np.sum(tmp_mask) < 2500:
                            continue
                        mask += area_counting * tmp_mask
                        # mask = cv2.dilate(mask, kernel, iterations=dilate_iter_n)
                        area_counting += 1
                    elif 'double' in lane[0]:
                        continue
                    else:
                        tmp_mask = mask_area_by_one_curve(
                            curves[pair], img_size, thickness=5)
                        if np.sum(tmp_mask) < 2500:
                            continue

                        mask += tmp_mask
                        area_counting += 1

                ins_save_name = instance_paths[data_target] / pic_name

                cv2.imwrite(str(ins_save_name), mask)
                seg_save_name = segments_paths[data_target] / pic_name

                cv2.imwrite(str(seg_save_name),
                            np.where(mask > 0, 255, mask))
                # cv2.imwrite(cropped_img_save_path, cropped)
                fff.write(str(pic_path) + ' ' +
                            str(seg_save_name) + ' '+str(ins_save_name) + '\n')
