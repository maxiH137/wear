import json
import os
import numpy as np
import pandas as pd

from utils import apply_sliding_window, label_dict, convert_labels_to_annotation_json

# define split
sbjs = [['sbj_0', 'sbj_1', 'sbj_2', 'sbj_3', 'sbj_4', 'sbj_5'], ['sbj_6', 'sbj_7', 'sbj_8', 'sbj_9', 'sbj_10', 'sbj_11'], ['sbj_12', 'sbj_13', 'sbj_14', 'sbj_15', 'sbj_16', 'sbj_17']]

# change these parameters
window_size = 50
window_overlap = 50
frames = 60
stride = 30

# change output folder
raw_inertial_folder = './data/wear/raw/inertial'
inertial_folder_flat = './data/wear/processed/inertial_features/custom_flat'
inertial_folder_custom = './data/wear/processed/inertial_features/custom'
inertial_folder_aug = './data/wear/processed/inertial_features/augment'
inertial_folder = './data/wear/processed/inertial_features/{}_frames_{}_stride'.format(frames, stride)

#i3d_folder = './data/wear/processed/i3d_features/{}_frames_{}_stride'.format(frames, stride)
#combined_folder = './data/wear/processed/combined_features/{}_frames_{}_stride'.format(frames, stride)
anno_folder = './data/wear/annotations'


# fixed dataset properties
nb_sbjs = 18
nb_sbjs_aug = nb_sbjs*6
fps = 60
sampling_rate = 50

def wear_anno(wear_annotation, number, sbj_anno):
    wear_annotation['database']['sbj_' + str(int(number))] = {
            'subset': train_test,
            'duration': duration_seconds,
            'fps': fps,
            'annotations': sbj_anno,
            } 
    number += 1
    return wear_annotation, number

def permute_subj(inertial_sbj, sbj_perm, matrix):
    
    pre_perm = np.array([inertial_sbj.right_arm_acc_x, inertial_sbj.right_arm_acc_y, inertial_sbj.right_arm_acc_z])
    
    reshape_right_arm = np.array([inertial_sbj.right_arm_acc_x, inertial_sbj.right_arm_acc_y, inertial_sbj.right_arm_acc_z])
    reshape_left_arm = np.array([inertial_sbj.left_arm_acc_x, inertial_sbj.left_arm_acc_y, inertial_sbj.left_arm_acc_z])
    reshape_right_leg = np.array([inertial_sbj.right_leg_acc_x, inertial_sbj.right_leg_acc_y, inertial_sbj.right_leg_acc_z])
    reshape_left_leg = np.array([inertial_sbj.left_leg_acc_x, inertial_sbj.left_leg_acc_y, inertial_sbj.left_leg_acc_z])

    reshape_right_arm_result = np.dot(matrix, reshape_right_arm)
    reshape_left_arm_result = np.dot(matrix, reshape_left_arm)
    reshape_right_leg_result = np.dot(matrix, reshape_right_leg)
    reshape_left_leg_result = np.dot(matrix, reshape_left_leg)
    
    sbj_perm.right_arm_acc_x, sbj_perm.right_arm_acc_y, sbj_perm.right_arm_acc_z = reshape_right_arm_result[0], reshape_right_arm_result[1], reshape_right_arm_result[2]
    sbj_perm.left_arm_acc_x, sbj_perm.left_arm_acc_y, sbj_perm.left_arm_acc_z = reshape_left_arm_result[0], reshape_left_arm_result[1], reshape_left_arm_result[2]
    sbj_perm.right_leg_acc_x, sbj_perm.right_leg_acc_y, sbj_perm.right_leg_acc_z = reshape_right_leg_result[0], reshape_right_leg_result[1], reshape_right_leg_result[2]
    sbj_perm.left_leg_acc_x, sbj_perm.left_leg_acc_y, sbj_perm.left_leg_acc_z = reshape_left_leg_result[0], reshape_left_leg_result[1], reshape_left_leg_result[2]

    return sbj_perm

def normalize_subj(inertial_sbj, sbj_perm):
    
    pre_norm= np.array([inertial_sbj.right_arm_acc_x, inertial_sbj.right_arm_acc_y, inertial_sbj.right_arm_acc_z])
    
    return sbj_perm

def save_augment(sbj_perm, sbj_string, perm):
    inertial_sbj = sbj_perm.replace({"label": label_dict}).fillna(-1).to_numpy()
    inertial_sbj[:, -1] += 1
    _, win_sbj, _ = apply_sliding_window(inertial_sbj, window_size, window_overlap)
    flipped_sbj = np.transpose(win_sbj[:, :, 1:], (0,2,1))
    
    output_inertial = flipped_sbj.reshape(flipped_sbj.shape[0], -1)

    np.save(os.path.join(inertial_folder_aug + perm, sbj_string + '.npy'), output_inertial)

number_anno = 18
number = 18

for i, split_sbjs in enumerate(sbjs):
    wear_annotations = {'version': 'Wear', 'database': {}, 'label_dict': label_dict}
  
    for sbj in split_sbjs:

        raw_inertial_sbj = pd.read_csv(os.path.join(raw_inertial_folder, sbj + '.csv'), index_col=None)  
        raw_inertial_sbj_perm1 = pd.read_csv(os.path.join(raw_inertial_folder, sbj + '.csv'), index_col=None) 
        raw_inertial_sbj_perm2 = pd.read_csv(os.path.join(raw_inertial_folder, sbj + '.csv'), index_col=None) 
        raw_inertial_sbj_perm3 = pd.read_csv(os.path.join(raw_inertial_folder, sbj + '.csv'), index_col=None) 
        raw_inertial_sbj_perm4 = pd.read_csv(os.path.join(raw_inertial_folder, sbj + '.csv'), index_col=None) 
        raw_inertial_sbj_perm5 = pd.read_csv(os.path.join(raw_inertial_folder, sbj + '.csv'), index_col=None) 

        raw_inertial_sbj_perm1 = permute_subj(raw_inertial_sbj, raw_inertial_sbj_perm1, np.array([[1,0,0], [0,0,1], [0,1,0]]))
        raw_inertial_sbj_perm2 = permute_subj(raw_inertial_sbj, raw_inertial_sbj_perm2, np.array([[0,1,0], [1,0,0], [0,0,1]]))
        raw_inertial_sbj_perm3 = permute_subj(raw_inertial_sbj, raw_inertial_sbj_perm3, np.array([[0,0,1], [0,1,0], [1,0,0]]))
        raw_inertial_sbj_perm4 = permute_subj(raw_inertial_sbj, raw_inertial_sbj_perm4, np.array([[0,0,1], [1,0,0], [0,1,0]]))
        raw_inertial_sbj_perm5 = permute_subj(raw_inertial_sbj, raw_inertial_sbj_perm5, np.array([[0,1,0], [0,0,1], [1,0,0]]))

        save_augment(raw_inertial_sbj_perm1, 'sbj_' + str(number), '/perm_all3')
        number += 1
        save_augment(raw_inertial_sbj_perm1, 'sbj_' + str(number), '/perm_all3')
        number += 1
        save_augment(raw_inertial_sbj_perm1, 'sbj_' + str(number), '/perm_all3')
        number += 1
        save_augment(raw_inertial_sbj_perm1, 'sbj_' + str(number), '/perm_all3')
        number += 1
        save_augment(raw_inertial_sbj_perm1, 'sbj_' + str(number), '/perm_all3')
        number += 1

        inertial_sbj = raw_inertial_sbj.replace({"label": label_dict}).fillna(-1).to_numpy()
        inertial_sbj[:, -1] += 1
        _, win_sbj, _ = apply_sliding_window(inertial_sbj, window_size, window_overlap)
        flipped_sbj = np.transpose(win_sbj[:, :, 1:], (0,2,1))
        unflipped_sbj = win_sbj[:, :, 1:]
        flat_win_sbj = win_sbj.reshape(win_sbj.shape[0], -1)
        output_inertial = flipped_sbj.reshape(flipped_sbj.shape[0], -1)
        #output_i3d = np.load(os.path.join(i3d_folder, sbj + '.npy'))
        #try:
        #    output_combined = np.concatenate((output_inertial, output_i3d), axis=1)
        #except ValueError:
        #    print('had to chop')
        #    output_combined = np.concatenate((output_inertial[:output_i3d.shape[0], :], output_i3d), axis=1)

        np.save(os.path.join(inertial_folder_flat, sbj + '.npy'), output_inertial)
        np.save(os.path.join(inertial_folder_custom, sbj + '.npy'), unflipped_sbj)
        np.save(os.path.join(inertial_folder, sbj + '.npy'), output_inertial)
        #np.save(os.path.join(combined_folder, sbj + '.npy'), output_combined)
      
    
    # create video annotations
    for j in range(nb_sbjs):
        curr_sbj = "sbj_" + str(j)
        raw_inertial_sbj_t = pd.read_csv(os.path.join(raw_inertial_folder, curr_sbj + '.csv'), index_col=None)
        duration_seconds = len(raw_inertial_sbj_t) / sampling_rate
        sbj_annos = convert_labels_to_annotation_json(raw_inertial_sbj_t.iloc[:, -1], sampling_rate, fps, label_dict)
        
        # Add Augmentation only in training data, not in Validation data
        if curr_sbj in split_sbjs:
            train_test = 'Validation'
            wear_annotations, _           = wear_anno(wear_annotations, j, sbj_annos)
        else:
            train_test = 'Training'
            wear_annotations, _           = wear_anno(wear_annotations, j, sbj_annos)
            wear_annotations, number_anno = wear_anno(wear_annotations, number_anno, sbj_annos)
            wear_annotations, number_anno = wear_anno(wear_annotations, number_anno, sbj_annos)
            wear_annotations, number_anno = wear_anno(wear_annotations, number_anno, sbj_annos)
            wear_annotations, number_anno = wear_anno(wear_annotations, number_anno, sbj_annos)
            wear_annotations, number_anno = wear_anno(wear_annotations, number_anno, sbj_annos)

        with open(os.path.join(anno_folder, 'wear_split_aug_' + str(int(i + 1)) +  '.json'), 'w') as outfile:
            outfile.write(json.dumps(wear_annotations, indent = 4))
