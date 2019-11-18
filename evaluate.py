#! /usr/bin/env python

import argparse
import os
import numpy as np
from preprocessing import parse_annotation
from preprocessing import BatchGenerator
from frontend import YOLO
import json
import pandas as pd
from backend import FullYoloFeature

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations 
    ###############################

    # parse annotations of the training set
    train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'], 
                                                config['train']['train_image_folder'], 
                                                config['model']['labels'])

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(config['valid']['valid_annot_folder']):
        valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'], 
                                                    config['valid']['valid_image_folder'], 
                                                    config['model']['labels'])
    else:
        train_valid_split = int(0.8*len(train_imgs))
        np.random.shuffle(train_imgs)

        valid_imgs = train_imgs[train_valid_split:]
        train_imgs = train_imgs[:train_valid_split]

    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

        print('Seen labels:\t', train_labels)
        print('Given labels:\t', config['model']['labels'])
        print('Overlap labels:\t', overlap_labels)           

        if len(overlap_labels) < len(config['model']['labels']):
            print('Some labels have no annotations! Please revise the list of labels in the config.json file!')
            return
    else:
        print('No labels are provided. Train on all seen labels.')
        config['model']['labels'] = train_labels.keys()
        
    ###############################
    #   Construct the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])
    
    ###############################
    #   Load trained weights
    ###############################    
    yolo.load_weights(config['train']['saved_weights_name'])

    ###############################
    #   Create Test set generator
    ###############################
    generator_config = {
            'IMAGE_H'         : config['model']['input_size'], 
            'IMAGE_W'         : config['model']['input_size'],
            'GRID_H'          : 13,  
            'GRID_W'          : 13,
            'BOX'             : len(config['model']['anchors'])//2,
            'LABELS'          : config['model']['labels'],
            'CLASS'           : len(config['model']['labels']),
            'ANCHORS'         : config['model']['anchors'],
            'BATCH_SIZE'      : 1,
            'TRUE_BOX_BUFFER' : 2,
        }    
    test_generator = BatchGenerator(valid_imgs, 
                                    generator_config, 
                                    norm=FullYoloFeature(config['model']['input_size']).normalize,
                                    jitter=False)   

    ################################
    ##   Start the evaluation process 
    ################################
    PZ_AP = []
    Prostate_AP = []
    mAP = []
    iou_range = np.arange(0.5, 1.0, 0.05)
    
    results = pd.DataFrame(iou_range, columns=['IoU_threshold'])
        
    for iou_threshold in iou_range:
        current_results = yolo.evaluate(test_generator, 
                                        iou_threshold=iou_threshold,
                                        score_threshold=0.05,
                                        max_detections=2)
        current_pz_ap = current_results[0]
        current_prostate_ap = current_results[1]
        current_mAP = 0.5 * (current_pz_ap +  current_prostate_ap)
        
        PZ_AP.append(current_pz_ap)
        Prostate_AP.append(current_prostate_ap)
        mAP.append(current_mAP)
    
    results['PZ_AP'] = PZ_AP
    results['Prostate_AP'] = Prostate_AP
    results['mAP'] = mAP

    # Absolute distance metrics
    abs_diff_results = yolo.calc_border_distance(test_generator,
                                                 iou_threshold=0.5,
                                                 score_threshold=0.3,
                                                 max_detections=2)
    #results['PZ_abs_diff'] = abs_diff_results[0]['abs_distance']
    #results['PZ_FPs'] = abs_diff_results[0]['FPs']
    #results['Prostate_abs_diff'] = abs_diff_results[1]['abs_distance']
    #results['Prostate_FPs'] = abs_diff_results[1]['FPs']
    border_error = pd.DataFrame(abs_diff_results, columns=['left', 'top', 'right', 'bottom'])

    csv_output_path = os.path.join(config['train']['saved_weights_name'].rsplit(os.sep, 2)[0], 'quantitative_results.csv')
    results.to_csv(csv_output_path, index=False)
    csv_output_path = os.path.join(config['train']['saved_weights_name'].rsplit(os.sep, 2)[0], 'border_error_results.csv')
    print('Error distance results: ', abs_diff_results.shape)
    print(csv_output_path)
    border_error.to_csv(csv_output_path, index=False)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
