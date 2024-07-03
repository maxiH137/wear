import json
import os
import numpy as np
import pandas as pd
import argparse

from utils import apply_sliding_window, label_dict, convert_labels_to_annotation_json, load_config

# define split
#sbjs = [['sbj_0'], ['sbj_1'], ['sbj_2'], ['sbj_3'], ['sbj_4'], ['sbj_5'], ['sbj_6'], ['sbj_7'], ['sbj_8'], ['sbj_9'], ['sbj_10'], ['sbj_11'], ['sbj_12'], ['sbj_13'], ['sbj_14'], ['sbj_15'], ['sbj_16'], ['sbj_17']]
sbjs = [['sbj_0', 'sbj_1', 'sbj_2', 'sbj_3', 'sbj_4', 'sbj_5'], ['sbj_6', 'sbj_7', 'sbj_8', 'sbj_9', 'sbj_10', 'sbj_11'], ['sbj_12', 'sbj_13', 'sbj_14', 'sbj_15', 'sbj_16', 'sbj_17']]

# change these parameters
window_size = 50
window_overlap = 50
frames = 60
stride = 30

activity = []

# change output folder
raw_inertial_folder = './data/wear/raw/inertial'
inertial_folder_flat = './data/wear/processed/inertial_features/custom_flat'
inertial_folder_custom = './data/wear/processed/inertial_features/custom'
inertial_folder_aug = './data/wear/processed/inertial_features/augment'
inertial_folder = './data/wear/processed/inertial_features/{}_frames_{}_stride'.format(frames, stride)
aug_folder = '/set1'

#i3d_folder = './data/wear/processed/i3d_features/{}_frames_{}_stride'.format(frames, stride)
#combined_folder = './data/wear/processed/combined_features/{}_frames_{}_stride'.format(frames, stride)
anno_folder = './data/wear/annotations'


# fixed dataset properties
nb_sbjs = 18
fps = 60
sampling_rate = 50

def wear_anno(wear_annotation, number, sbj_anno, train_test, duration_seconds):
    wear_annotation['database']['sbj_' + str(int(number))] = {
            'subset': train_test,
            'duration': duration_seconds,
            'fps': fps,
            'annotations': sbj_anno,
            } 
    #number += 1
    return wear_annotation

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

# Normalization described in: https://dl.acm.org/doi/pdf/10.1145/3659597
def normalize_subj(inertial_sbj):  
    inertial_sbj[:,0,:] = (inertial_sbj[:,0,:] - np.mean(inertial_sbj[:,0,:]))/np.std(inertial_sbj[:,0,:])
    assert(np.std(inertial_sbj[:,0,:]) < 1.01 and np.abs(np.mean(inertial_sbj[:,0,:])) < 0.01)

    inertial_sbj[:,1,:] = (inertial_sbj[:,1,:] - np.mean(inertial_sbj[:,1,:]))/np.std(inertial_sbj[:,1,:])
    assert(np.std(inertial_sbj[:,1,:]) < 1.01 and np.abs(np.mean(inertial_sbj[:,1,:])) < 0.01)

    inertial_sbj[:,2,:] = (inertial_sbj[:,2,:] - np.mean(inertial_sbj[:,2,:]))/np.std(inertial_sbj[:,2,:])
    assert(np.std(inertial_sbj[:,2,:]) < 1.01 and np.abs(np.mean(inertial_sbj[:,2,:])) < 0.01)

    inertial_sbj[:,3,:] = (inertial_sbj[:,3,:] - np.mean(inertial_sbj[:,3,:]))/np.std(inertial_sbj[:,3,:])
    assert(np.std(inertial_sbj[:,3,:]) < 1.01 and np.abs(np.mean(inertial_sbj[:,3,:])) < 0.01)

    inertial_sbj[:,4,:] = (inertial_sbj[:,4,:] - np.mean(inertial_sbj[:,4,:]))/np.std(inertial_sbj[:,4,:])
    assert(np.std(inertial_sbj[:,4,:]) < 1.01 and np.abs(np.mean(inertial_sbj[:,4,:])) < 0.01)

    inertial_sbj[:,5,:] = (inertial_sbj[:,5,:] - np.mean(inertial_sbj[:,5,:]))/np.std(inertial_sbj[:,5,:])
    assert(np.std(inertial_sbj[:,5,:]) < 1.01 and np.abs(np.mean(inertial_sbj[:,5,:])) < 0.01)

    inertial_sbj[:,6,:] = (inertial_sbj[:,6,:] - np.mean(inertial_sbj[:,6,:]))/np.std(inertial_sbj[:,6,:])
    assert(np.std(inertial_sbj[:,6,:]) < 1.01 and np.abs(np.mean(inertial_sbj[:,6,:])) < 0.01)

    inertial_sbj[:,7,:] = (inertial_sbj[:,7,:] - np.mean(inertial_sbj[:,7,:]))/np.std(inertial_sbj[:,7,:])
    assert(np.std(inertial_sbj[:,7,:]) < 1.01 and np.abs(np.mean(inertial_sbj[:,7,:])) < 0.01)

    inertial_sbj[:,8,:] = (inertial_sbj[:,8,:] - np.mean(inertial_sbj[:,8,:]))/np.std(inertial_sbj[:,8,:])
    assert(np.std(inertial_sbj[:,8,:]) < 1.01 and np.abs(np.mean(inertial_sbj[:,8,:])) < 0.01)

    inertial_sbj[:,9,:] = (inertial_sbj[:,9,:] - np.mean(inertial_sbj[:,9,:]))/np.std(inertial_sbj[:,9,:])
    assert(np.std(inertial_sbj[:,9,:]) < 1.01 and np.abs(np.mean(inertial_sbj[:,9,:])) < 0.01)

    inertial_sbj[:,10,:] = (inertial_sbj[:,10,:] - np.mean(inertial_sbj[:,10,:]))/np.std(inertial_sbj[:,10,:])
    assert(np.std(inertial_sbj[:,10,:]) < 1.01 and np.abs(np.mean(inertial_sbj[:,10,:])) < 0.01)

    inertial_sbj[:,11,:] = (inertial_sbj[:,11,:] - np.mean(inertial_sbj[:,11,:]))/np.std(inertial_sbj[:,11,:])
    assert(np.std(inertial_sbj[:,11,:]) < 1.01 and np.abs(np.mean(inertial_sbj[:,11,:])) < 0.01)

    return inertial_sbj

# Augment methods taken from https://github.com/diheal/resampling/blob/f9f353f84c65625e82fd6691bf56c99042ea23ab/Augment.py
def inverting(inertial_sbj):
    return inertial_sbj * -1

def reversing(inertial_sbj):
    return inertial_sbj[:,::-1]

def magnify(inertial_sbj):
    mag = np.random.randint(11,14)/10
    return inertial_sbj * mag

def scaling(inertial_sbj):
    scale = np.random.randint(7,10)/10
    return inertial_sbj * scale

def noise(inertial_sbj):
    inertial_sbj = inertial_sbj + inertial_sbj * np.random.uniform(-0.1, 0.1, inertial_sbj.shape)    
    return inertial_sbj


def save_augment(sbj_perm, sbj, number, perm, augment, cfg):
    #x = sbj_perm.label._values
    inertial_sbj = sbj_perm.replace({"label": label_dict}).fillna(-1).to_numpy()
    inertial_sbj[:, -1] += 1
    _, win_sbj, _ = apply_sliding_window(inertial_sbj, window_size, window_overlap)
    flipped_sbj = np.transpose(win_sbj[:, :, 1:], (0,2,1))

    # Normalization of all 12 sensor axis of a subject, so mean is 0 and std is 1
    if(cfg['normalization']):
        flipped_sbj = normalize_subj(flipped_sbj)

    output_inertial = flipped_sbj.reshape(flipped_sbj.shape[0], -1)
    

    if(augment):
        if(cfg['noise']):
            noise_inertial = noise(output_inertial)
            np.save(os.path.join(inertial_folder_aug + perm, 'sbj_' + str(number) + '.npy'), noise_inertial)
            number += 1
        if(cfg['scaling']):
            scaling_inertial = scaling(output_inertial)
            np.save(os.path.join(inertial_folder_aug + perm, 'sbj_' + str(number) + '.npy'), scaling_inertial)
            number += 1
        if(cfg['magnify']):
            magnify_inertial = magnify(output_inertial)
            np.save(os.path.join(inertial_folder_aug + perm, 'sbj_' + str(number) + '.npy'), magnify_inertial)
            number += 1
        if(cfg['reversing']):
            reversing_inertial = reversing(output_inertial)
            np.save(os.path.join(inertial_folder_aug + perm, 'sbj_' + str(number) + '.npy'), reversing_inertial)
            number += 1
        if(cfg['inverting']):
            inverting_inertial = inverting(output_inertial)
            np.save(os.path.join(inertial_folder_aug + perm, 'sbj_' + str(number) + '.npy'), inverting_inertial)
            number += 1
        np.save(os.path.join(inertial_folder_aug + perm, 'sbj_' + str(number) + '.npy'), output_inertial)
    else:
        if(cfg['noise']):
            noise_inertial = noise(output_inertial)
            np.save(os.path.join(inertial_folder_aug + perm, 'sbj_' + str(number) + '.npy'), noise_inertial)
            number += 1
        if(cfg['scaling']):
            scaling_inertial = scaling(output_inertial)
            np.save(os.path.join(inertial_folder_aug + perm, 'sbj_' + str(number) + '.npy'), scaling_inertial)
            number += 1
        if(cfg['magnify']):
            magnify_inertial = magnify(output_inertial)
            np.save(os.path.join(inertial_folder_aug + perm, 'sbj_' + str(number) + '.npy'), magnify_inertial)
            number += 1
        if(cfg['reversing']):
            reversing_inertial = reversing(output_inertial)
            np.save(os.path.join(inertial_folder_aug + perm, 'sbj_' + str(number) + '.npy'), reversing_inertial)
            number += 1
        if(cfg['inverting']):
            inverting_inertial = inverting(output_inertial)
            np.save(os.path.join(inertial_folder_aug + perm, 'sbj_' + str(number) + '.npy'), inverting_inertial)
            number += 1
        np.save(os.path.join(inertial_folder_aug + perm, str(sbj) + '.npy'), output_inertial)
    
    return number

def activity_sbj(sbj, label):
    match label:
        case 0:
            jogging = 0
        case 1:
            jogging_rotating_arms = 0
        case 2:
            jogging_skipping = 0
        case 3:
            jogging_sidesteps = 0
        case 4:
            jogging_buttkicks = 0
        case 5:
            stretching_triceps = 0
        case 6:
            stretching_lunging = 0
        case 7:
            stretching_shoulders = 0
        case 8:
            stretching_hamstring = 0
        case 9:
            stretching_lumbar_rotation = 0
        case 10:
            push_ups = 0
        case 11:
            push_ups_complex = 0
        case 12:
            sit_ups = 0
        case 13:
            sit_ups_complex = 0
        case 14:
            burpees = 0
        case 15:
            lunges = 0
        case 16:
            lunges_complex = 0
        case 17:
            bench_dips = 0


def data_creation(args):

    config = load_config(args.config)
    number = 18
    #raw_intertial_sbj_all = None

    for i, split_sbjs in enumerate(sbjs):
        wear_annotations = {'version': 'Wear', 'database': {}, 'label_dict': label_dict}
    
        for sbj in split_sbjs:
            
            raw_inertial_sbj = pd.read_csv(os.path.join(raw_inertial_folder, sbj + '.csv'), index_col=None)      
            #raw_intertial_sbj_all = pd.concat([raw_intertial_sbj_all, raw_inertial_sbj], ignore_index=True)

            raw_inertial_sbj_perm1 = pd.read_csv(os.path.join(raw_inertial_folder, sbj + '.csv'), index_col=None) 
            raw_inertial_sbj_perm2 = pd.read_csv(os.path.join(raw_inertial_folder, sbj + '.csv'), index_col=None) 
            raw_inertial_sbj_perm3 = pd.read_csv(os.path.join(raw_inertial_folder, sbj + '.csv'), index_col=None) 
            raw_inertial_sbj_perm4 = pd.read_csv(os.path.join(raw_inertial_folder, sbj + '.csv'), index_col=None) 
            raw_inertial_sbj_perm5 = pd.read_csv(os.path.join(raw_inertial_folder, sbj + '.csv'), index_col=None) 
            
            if(config['augmentation']['permutation']):
                raw_inertial_sbj_perm1 = permute_subj(raw_inertial_sbj, raw_inertial_sbj_perm1, np.array([[1,0,0], [0,0,1], [0,1,0]]))
                raw_inertial_sbj_perm2 = permute_subj(raw_inertial_sbj, raw_inertial_sbj_perm2, np.array([[0,1,0], [1,0,0], [0,0,1]]))
                raw_inertial_sbj_perm3 = permute_subj(raw_inertial_sbj, raw_inertial_sbj_perm3, np.array([[0,0,1], [0,1,0], [1,0,0]]))
                raw_inertial_sbj_perm4 = permute_subj(raw_inertial_sbj, raw_inertial_sbj_perm4, np.array([[0,0,1], [1,0,0], [0,1,0]]))
                raw_inertial_sbj_perm5 = permute_subj(raw_inertial_sbj, raw_inertial_sbj_perm5, np.array([[0,1,0], [0,0,1], [1,0,0]]))

            # Normal Subject without Augmentation except Normalization if True
            number = save_augment(raw_inertial_sbj, sbj,number, aug_folder, False, config['augmentation'])
            
            # Augmented Data, especially Permutated data
            if(config['augmentation']['permutation']):
                number = save_augment(raw_inertial_sbj_perm1, sbj, number, aug_folder, True, config['augmentation'])
                number += 1
                number = save_augment(raw_inertial_sbj_perm2, sbj, number, aug_folder, True, config['augmentation'])
                number += 1
                number = save_augment(raw_inertial_sbj_perm3, sbj, number, aug_folder, True, config['augmentation'])
                number += 1
                number = save_augment(raw_inertial_sbj_perm4, sbj, number, aug_folder, True, config['augmentation'])
                number += 1
                number = save_augment(raw_inertial_sbj_perm5, sbj, number, aug_folder, True, config['augmentation'])
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

            #np.save(os.path.join(inertial_folder_flat, sbj + '.npy'), output_inertial)
            #np.save(os.path.join(inertial_folder_custom, sbj + '.npy'), unflipped_sbj)
            #np.save(os.path.join(inertial_folder, sbj + '.npy'), output_inertial)
            #np.save(os.path.join(combined_folder, sbj + '.npy'), output_combined)
        
        
        # create video annotations
        number_anno = 18
        for j in range(nb_sbjs):
            curr_sbj = "sbj_" + str(j)
            raw_inertial_sbj_t = pd.read_csv(os.path.join(raw_inertial_folder, curr_sbj + '.csv'), index_col=None)
            duration_seconds = len(raw_inertial_sbj_t) / sampling_rate
            sbj_annos = convert_labels_to_annotation_json(raw_inertial_sbj_t.iloc[:, -1], sampling_rate, fps, label_dict)
            
            # Add Augmentation only in training data, not in Validation data
            if curr_sbj in split_sbjs:
                train_test = 'Validation'
                wear_annotations = wear_anno(wear_annotations, j, sbj_annos, train_test, duration_seconds)

                if(config['augmentation']['permutation']):
                    number_anno += 5

                if(config['augmentation']['scaling']):
                    if(config['augmentation']['permutation']):
                        number_anno += 6
                    else:
                        number_anno += 1

                if(config['augmentation']['magnify']):
                    if(config['augmentation']['permutation']):
                        number_anno += 6
                    else:
                        number_anno += 1

                if(config['augmentation']['noise']):
                    if(config['augmentation']['permutation']):
                        number_anno += 6
                    else:
                        number_anno += 1

                if(config['augmentation']['reversing']):
                    if(config['augmentation']['permutation']):
                        number_anno += 6
                    else:
                        number_anno += 1

                if(config['augmentation']['inverting']):
                    if(config['augmentation']['permutation']):
                        number_anno += 6
                    else:
                        number_anno += 1
            else:
                train_test = 'Training'
                wear_annotations = wear_anno(wear_annotations, j, sbj_annos, train_test, duration_seconds)
                #for number_saved in range(amount_aug_saved+1):
                #    wear_annotations = wear_anno(wear_annotations, number_anno + number_saved + 1 + (j*amount_aug_saved), sbj_annos, train_test, duration_seconds)
                
                if(config['augmentation']['permutation']):
                    wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                    number_anno += 1
                    wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                    number_anno += 1
                    wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                    number_anno += 1
                    wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                    number_anno += 1
                    wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                    number_anno += 1

                if(config['augmentation']['scaling']):
                   
                    if(config['augmentation']['permutation']):
                        wear_annotations = wear_anno(wear_annotations, number_anno, sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                    else:
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1

                if(config['augmentation']['magnify']):
                   
                    if(config['augmentation']['permutation']):
                        wear_annotations = wear_anno(wear_annotations, number_anno, sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                    else:
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1

                if(config['augmentation']['noise']):
              
                    if(config['augmentation']['permutation']):
                        wear_annotations = wear_anno(wear_annotations, number_anno, sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                    else:
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1

                if(config['augmentation']['reversing']):
                  
                    if(config['augmentation']['permutation']):
                        wear_annotations = wear_anno(wear_annotations, number_anno, sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                    else:
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        
                if(config['augmentation']['inverting']):
                
                    if(config['augmentation']['permutation']):
                        wear_annotations = wear_anno(wear_annotations, number_anno, sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1
                    else:
                        wear_annotations = wear_anno(wear_annotations, number_anno , sbj_annos, train_test, duration_seconds)
                        number_anno += 1

            with open(os.path.join(anno_folder, 'wear_split_aug_' + str(int(i + 1)) +  '.json'), 'w') as outfile:
                outfile.write(json.dumps(wear_annotations, indent = 4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/60_frames_30_stride/tridet_inertial_aug.yaml')
    args = parser.parse_args()
    data_creation(args)  