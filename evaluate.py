import os.path

import csv
import urllib.request
import cv2
import matplotlib.pyplot as plt

import measurements
import utils
from utils import *
from drawhelper import draw_poly_on_image, fill_poly_on_image

import pandas as pd

from sklearn.metrics import mean_absolute_error
import numpy as np


def create_box_plot(save_path, data_lists, colours_list, x_label, y_label, x_ticks_list):

    fig, ax = plt.subplots()
    bplot = ax.boxplot(data_lists, patch_artist=True, labels=x_ticks_list, showmeans=True, meanline=True)

    #'boxes', 'whiskers', 'fliers', 'medians', 'caps'

    for patch, colour in zip(bplot['boxes'], colours_list):
         #patch.set_facecolor(colour)
         patch.set(facecolor=colour)

    for flier, colour in zip(bplot['fliers'], colours_list):
         flier.set(markeredgecolor=colour)

    for median, colour in zip(bplot['medians'], colours_list):
         median.set_color('magenta')

    ax.yaxis.grid(True)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(save_path)
    plt.clf()


def box_plot_test():
    perc1 = 12
    perc2 = 100
    run = 0
    method = 'Patch'
    output_path = f'Results_{method}_compare_{perc1}_{perc2}_run{0}'

    path1 = f'run_SimCLR_Echo_unet_bce_True_batch8/Exp_with_30000_percent_unlabelled_data/finetuning_{perc1}'
    path2 = f'run_SimCLR_Echo_unet_bce_True_batch8/Exp_with_30000_percent_unlabelled_data/finetuning_{perc2}'

    file1 = os.path.join(path1, f'Consensus_EF_Vol_results_all_{run}.json')
    file2 = os.path.join(path2, f'Consensus_EF_Vol_results_all_{run}.json')

    dict1 = load_json(file1)
    dict2 = load_json(file2)

    ef1, ef2 = [], []
    ef_err1, ef_err2 = [], []
    colour_list = ['blue', 'green']
    for key, value in dict1.items():
        ef1.append(value['Ejection Fraction pred'])
        ef_err1.append(value['EF error'])
    for key, value in dict2.items():
        ef2.append(value['Ejection Fraction pred'])
        ef_err2.append(value['EF error'])
    create_box_plot(save_path=os.path.join(output_path, f'box1.png'),
                    data_lists=[ef1, ef2],
                    colours_list=colour_list,
                    x_ticks_list=['12%', '100%'],
                    x_label='Downstream Percentage',
                    y_label='EF')
    create_box_plot(save_path=os.path.join(output_path, f'box2.png'),
                    data_lists=[ef_err1, ef_err2],
                    colours_list=colour_list,
                    x_ticks_list=['12%', '100%'],
                    x_label='Downstream Percentage',
                    y_label='EF Error')
    return

def compare_two_percentages():
    source_images_path = 'Data/Expert Consensus dataset/Images/'
    perc1 = 12
    perc2 = 100
    run = 0
    method = 'Patch'
    output_path = f'Results_{method}_compare_{perc1}_{perc2}_run{0}'

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    path1 = f'run_SimCLR_Echo_unet_bce_True_batch8/Exp_with_30000_percent_unlabelled_data/finetuning_{perc1}'
    path2 = f'run_SimCLR_Echo_unet_bce_True_batch8/Exp_with_30000_percent_unlabelled_data/finetuning_{perc2}'

    file1 = os.path.join(path1, f'Consensus_EF_Vol_results_all_{run}.json')
    file2 = os.path.join(path2, f'Consensus_EF_Vol_results_all_{run}.json')

    file1_res = os.path.join(path1, f'Consensus_dataset_results_all_{run}.json')
    file2_res = os.path.join(path2, f'Consensus_dataset_results_all_{run}.json')

    output_images_path1 = os.path.join(path1, f'Consensus_dataset_results_all_{run} examples')
    output_images_path2 = os.path.join(path2, f'Consensus_dataset_results_all_{run} examples')

    output_disks_path1 = os.path.join(path1, f'Consensus_disks_volumes_run_{run}')
    output_disks_path2 = os.path.join(path2, f'Consensus_disks_volumes_run_{run}')

    dict1 = load_json(file1)
    dict2 = load_json(file2)

    dict_res1 = load_json(file1_res)
    dict_res2 = load_json(file2_res)

    count = 0
    count_edv_less_esv1 = 0
    count_edv_less_esv2 = 0
    for key, value in dict1.items():
        val_dict1 = value
        val_dict2 = dict2[key]

        ef_err1 = val_dict1['EF error']
        ef_err2 = val_dict2['EF error']

        # eg for fine-tuning 12%
        ED_frame_1 = val_dict1['ED frame']
        ES_frame_1 = val_dict1['ES frame']

        # eg for fine-tuning 100%
        ED_frame_2 = val_dict2['ED frame']
        ES_frame_2 = val_dict2['ES frame']

        if ef_err1 < ef_err2:
            count += 1

            if val_dict1['EDV pred'] < val_dict1['ESV pred']:
                count_edv_less_esv1 += 1
            if val_dict2['EDV pred'] < val_dict2['ESV pred']:
                count_edv_less_esv2 += 1

            val_res_dict1_ED = dict_res1[source_images_path + ED_frame_1]
            val_res_dict1_ES = dict_res1[source_images_path + ES_frame_1]

            val_res_dict2_ED = dict_res2[source_images_path + ED_frame_2]
            val_res_dict2_ES = dict_res2[source_images_path + ES_frame_2]

            temp_name = ED_frame_1[0:ED_frame_1.index('.png')] + '_2' + '.png'
            image1_ED = os.path.join(output_images_path1, temp_name)
            temp_name = ES_frame_1[0:ES_frame_1.index('.png')] + '_2' + '.png'
            image1_ES = os.path.join(output_images_path1, temp_name)

            image1_ED_disk = os.path.join(output_disks_path1, ED_frame_1)
            image1_ES_disk = os.path.join(output_disks_path1, ES_frame_1)

            temp_name = ED_frame_2[0:ED_frame_2.index('.png')] + '_2' + '.png'
            image2_ED = os.path.join(output_images_path2, temp_name)
            temp_name = ES_frame_2[0:ES_frame_2.index('.png')] + '_2' + '.png'
            image2_ES = os.path.join(output_images_path2, temp_name)

            image2_ED_disk = os.path.join(output_disks_path2, ED_frame_2)
            image2_ES_disk = os.path.join(output_disks_path2, ES_frame_2)

            img1_ED = cv2.imread(image1_ED)
            img1_ED = cv2.cvtColor(img1_ED, cv2.COLOR_BGR2RGB)
            img1_ES = cv2.imread(image1_ES)
            img1_ES = cv2.cvtColor(img1_ES, cv2.COLOR_BGR2RGB)
            img1_ED_disk = cv2.imread(image1_ED_disk)
            img1_ED_disk = cv2.cvtColor(img1_ED_disk, cv2.COLOR_BGR2RGB)
            img1_ES_disk = cv2.imread(image1_ES_disk)
            img1_ES_disk = cv2.cvtColor(img1_ES_disk, cv2.COLOR_BGR2RGB)

            img2_ED = cv2.imread(image2_ED)
            img2_ED = cv2.cvtColor(img2_ED, cv2.COLOR_BGR2RGB)
            img2_ES = cv2.imread(image2_ES)
            img2_ES = cv2.cvtColor(img2_ES, cv2.COLOR_BGR2RGB)
            img2_ED_disk = cv2.imread(image2_ED_disk)
            img2_ED_disk = cv2.cvtColor(img2_ED_disk, cv2.COLOR_BGR2RGB)
            img2_ES_disk = cv2.imread(image2_ES_disk)
            img2_ES_disk = cv2.cvtColor(img2_ES_disk, cv2.COLOR_BGR2RGB)

            fontscale = 0.8
            pos1 = (50, 40)
            pos2 = (50, 70)
            pos3 = (50, 100)
            pos4 = (270, 40)
            pos5 = (270, 70)
            pos6 = (270, 100)
            # perc1
            val_temp = val_res_dict1_ED['dice_endo']
            img1_ED = cv2.putText(img1_ED, f'ED DICE: {round(val_temp, 3)}', pos1,
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  fontscale, (255, 255, 255))
            val_temp = val_res_dict1_ES['dice_endo']
            img1_ES = cv2.putText(img1_ES, f'ES DICE: {round(val_temp, 3)}', pos1,
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  fontscale, (255, 255, 255))

            val_temp_gt = val_dict1['EDV gt']
            val_temp = val_dict1['EDV pred']
            val_temp_er = val_dict1['EDV error']
            img1_ED_disk = cv2.putText(img1_ED_disk, f'EDV gt: {round(val_temp_gt, 3)}', pos1,
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       fontscale, (255, 255, 255))
            img1_ED_disk = cv2.putText(img1_ED_disk, f'EDV pr: {round(val_temp, 3)}', pos2,
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       fontscale, (255, 255, 255))
            img1_ED_disk = cv2.putText(img1_ED_disk, f'EDV er: {round(val_temp_er, 3)}', pos3,
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       fontscale, (255, 255, 255))
            val_temp_gt = val_dict1['ESV gt']
            val_temp = val_dict1['ESV pred']
            val_temp_er = val_dict1['ESV error']
            val_temp_efgt = val_dict1['Ejection Fraction gt']
            val_temp_efpr = val_dict1['Ejection Fraction pred']
            val_temp_efer = val_dict1['EF error']
            img1_ES_disk = cv2.putText(img1_ES_disk, f'ESV gt: {round(val_temp_gt, 3)}', pos1,
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       fontscale, (255, 255, 255))
            img1_ES_disk = cv2.putText(img1_ES_disk, f'ESV pr: {round(val_temp, 3)}', pos2,
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       fontscale, (255, 255, 255))
            img1_ES_disk = cv2.putText(img1_ES_disk, f'ESV er: {round(val_temp_er, 3)}', pos3,
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       fontscale, (255, 255, 255))
            img1_ES_disk = cv2.putText(img1_ES_disk, f'EF gt: {round(val_temp_efgt, 3)}', pos4,
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       fontscale, (255, 255, 255))
            img1_ES_disk = cv2.putText(img1_ES_disk, f'EF pr: {round(val_temp_efpr, 3)}', pos5,
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       fontscale, (255, 255, 255))
            img1_ES_disk = cv2.putText(img1_ES_disk, f'EF er: {round(val_temp_efer, 3)}', pos6,
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       fontscale, (255, 255, 255))

            # perc2
            val_temp = val_res_dict2_ED['dice_endo']
            img2_ED = cv2.putText(img2_ED, f'ED DICE: {round(val_temp, 3)}', pos1,
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  fontscale, (255, 255, 255))
            val_temp = val_res_dict2_ES['dice_endo']
            img2_ES = cv2.putText(img2_ES, f'ES DICE: {round(val_temp, 3)}', pos1,
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  fontscale, (255, 255, 255))

            val_temp_gt = val_dict2['EDV gt']
            val_temp = val_dict2['EDV pred']
            val_temp_er = val_dict2['EDV error']
            img2_ED_disk = cv2.putText(img2_ED_disk, f'EDV gt: {round(val_temp_gt, 3)}', pos1,
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       fontscale, (255, 255, 255))
            img2_ED_disk = cv2.putText(img2_ED_disk, f'EDV pr: {round(val_temp, 3)}', pos2,
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       fontscale, (255, 255, 255))
            img2_ED_disk = cv2.putText(img2_ED_disk, f'EDV er: {round(val_temp_er, 3)}', pos3,
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       fontscale, (255, 255, 255))
            val_temp_gt = val_dict2['ESV gt']
            val_temp = val_dict2['ESV pred']
            val_temp_er = val_dict2['ESV error']
            val_temp_efgt = val_dict2['Ejection Fraction gt']
            val_temp_efpr = val_dict2['Ejection Fraction pred']
            val_temp_efer = val_dict2['EF error']
            img2_ES_disk = cv2.putText(img2_ES_disk, f'ESV gt: {round(val_temp_gt, 3)}', pos1,
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       fontscale, (255, 255, 255))
            img2_ES_disk = cv2.putText(img2_ES_disk, f'ESV pr: {round(val_temp, 3)}', pos2,
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       fontscale, (255, 255, 255))
            img2_ES_disk = cv2.putText(img2_ES_disk, f'ESV er: {round(val_temp_er, 3)}', pos3,
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       fontscale, (255, 255, 255))
            img2_ES_disk = cv2.putText(img2_ES_disk, f'EF gt: {round(val_temp_efgt, 3)}', pos4,
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       fontscale, (255, 255, 255))
            img2_ES_disk = cv2.putText(img2_ES_disk, f'EF pr: {round(val_temp_efpr, 3)}', pos5,
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       fontscale, (255, 255, 255))
            img2_ES_disk = cv2.putText(img2_ES_disk, f'EF er: {round(val_temp_efer, 3)}', pos6,
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       fontscale, (255, 255, 255))

            # plt.figure(figsize=(20, 20), dpi=300)
            # fig, ax = plt.subplots(2, 4)
            # ax[0, 0].imshow(img1_ED)
            # ax[0, 1].imshow(img1_ES)
            # ax[0, 2].imshow(img1_ED_disk)
            # ax[0, 3].imshow(img1_ES_disk)
            #
            # ax[1, 0].imshow(img2_ED)
            # ax[1, 1].imshow(img2_ES)
            # ax[1, 2].imshow(img2_ED_disk)
            # ax[1, 3].imshow(img2_ES_disk)

            plt.figure(figsize=(15, 8), dpi=100)
            # plt.subplots_adjust(hspace=0.00001)

            ax = plt.subplot(2, 4, 1)
            plt.imshow(img1_ED)
            plt.axis('off')
            ax = plt.subplot(2, 4, 2)
            plt.imshow(img1_ES)
            plt.axis('off')
            ax = plt.subplot(2, 4, 3)
            plt.imshow(img1_ED_disk)
            plt.axis('off')
            ax = plt.subplot(2, 4, 4)
            plt.imshow(img1_ES_disk)
            plt.axis('off')

            ax = plt.subplot(2, 4, 5)
            plt.imshow(img2_ED)
            plt.axis('off')
            ax = plt.subplot(2, 4, 6)
            plt.imshow(img2_ES)
            plt.axis('off')
            ax = plt.subplot(2, 4, 7)
            plt.imshow(img2_ED_disk)
            plt.axis('off')
            ax = plt.subplot(2, 4, 8)
            plt.imshow(img2_ES_disk)
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(os.path.join(output_path, f'{key}.png')))
            plt.clf()

            # plot each annotation individually but all on the same plot.
            # plt.figure(figsize=(15, 15), dpi=300)
            # plt.subplots_adjust(hspace=0.01)
            # i = 0
            # for i in range(0, len(img_polys)):
            #     idx = i
            #     ax = plt.subplot(5, 3, i + 1)
            #     plt.imshow(img_polys[i])
            # #plt.tight_layout()
            # plt.savefig(os.path.join(os.path.join('temp_images_consensus_eval',f'{new_file_name}_3.png')))
            # plt.clf()

    print(f'Count EF error{perc1} < EF error{perc2}: {count}')
    print(f'Count EDV < ESV {perc1}: {count_edv_less_esv1}')
    print(f'Count EDV < ESV {perc2}: {count_edv_less_esv2}')


def find_low_ef_error_with_bad_dice():
    perc1 = 7
    run = 0
    method = 'SimCLR'
    output_path = f'Results_Dice_EF_Dissociate_{method}_search_{perc1}_run{0}'

    #if not os.path.exists(output_path):
    #    os.mkdir(output_path)

    source_images_path = 'Data/Expert Consensus dataset/Images/'
    source_labels_path = 'Data/Expert Consensus dataset/Labels/'

    path1 = f'run_SimCLR_Echo_unet_bce_True_batch20/Exp_with_30000_percent_unlabelled_data/finetuning_{perc1}'

    file1 = os.path.join(path1, f'Consensus_EF_Vol_results_all_{run}.json')

    file1_res = os.path.join(path1, f'Consensus_dataset_results_all_{run}.json')

    pred_images_path = os.path.join(path1, f'Consensus_dataset_results_all_{run} examples')

    pred_disks_file = os.path.join(path1, f'Consensus_disks_volumes_run_{run}.json')

    dict1 = load_json(file1)

    dict_res1 = load_json(file1_res)

    df = pd.DataFrame(
        columns=['ED_frame', 'ES_Frame',  # not used, just for combining dataframes later
                 'Ejection_Fraction_gt', 'Ejection_Fraction_pred',
                 'EF_error',
                 'EDV_gt', 'ESV_gt',
                 'EDV_pred', 'ESV_pred',
                 'EDV_error', 'ESV_error',
                 'ED_dice', 'ES_dice'])

    df['ED_frame'] = df['ED_frame'].astype(str)
    df['ES_Frame'] = df['ES_Frame'].astype(str)
    df['Ejection_Fraction_gt'] = df['Ejection_Fraction_gt'].astype(float)
    df['Ejection_Fraction_pred'] = df['Ejection_Fraction_pred'].astype(float)
    df['EF_error'] = df['EF_error'].astype(float)
    df['EDV_gt'] = df['EDV_gt'].astype(float)
    df['ESV_gt'] = df['ESV_gt'].astype(float)
    df['EDV_pred'] = df['EDV_pred'].astype(float)
    df['ESV_pred'] = df['ESV_pred'].astype(float)
    df['EDV_error'] = df['EDV_error'].astype(float)
    df['ESV_error'] = df['ESV_error'].astype(float)
    df['ED_dice'] = df['ED_dice'].astype(float)
    df['ES_dice'] = df['ES_dice'].astype(float)

    count = 0
    count_edv_less_esv1 = 0
    count_edv_less_esv2 = 0
    for key, value in dict1.items():
        val_dict1 = value
        ef_err1 = val_dict1['EF error']

        # eg for fine-tuning 12%
        ED_frame = val_dict1['ED frame']
        ES_frame = val_dict1['ES frame']
        EF_gt = val_dict1['Ejection Fraction gt']
        EF_pred = val_dict1['Ejection Fraction pred']
        EF_error = val_dict1['EF error']
        EDV_gt = val_dict1['EDV gt']
        ESV_gt = val_dict1['ESV gt']
        EDV_pred = val_dict1['EDV pred']
        ESV_pred = val_dict1['ESV pred']
        EDV_error = val_dict1['EDV error']
        ESV_error = val_dict1['ESV error']

        if EF_error==0 or EDV_pred<ESV_pred:
            continue

        val_res_dict1_ED = dict_res1[source_images_path + ED_frame]
        val_res_dict1_ES = dict_res1[source_images_path + ES_frame]

        dice_ED = val_res_dict1_ED['dice_endo']
        dice_ES = val_res_dict1_ES['dice_endo']

        row = {'ED_frame': ED_frame, 'ES_frame': ES_frame,
               'Ejection_Fraction_gt': EF_gt, 'Ejection_Fraction_pred': EF_pred,
               'EF_error': EF_error,
               'EDV_gt': EDV_gt, 'ESV_gt': ESV_gt,
               'EDV_pred': EDV_pred, 'ESV_pred': ESV_pred,
               'EDV_error': EDV_error, 'ESV_error': ESV_error,
               'ED_dice': dice_ED, 'ES_dice': dice_ES
               }
        df = df.append(row, ignore_index=True)

    # df_top10 = df[['EF_error', 'ED_dice']]
    # print(df_top10.head(10))
    print()

    #i.e. EF_error high to low,  ED_Dice and ES_dice high to low
    print('Bad EF error with good dice scores:')
    df1 = df.sort_values(['EF_error', 'ED_dice', 'ES_dice'], ascending=[False, False, False])
    df1_top10 = df1[['EF_error', 'ED_dice', 'ES_dice']]
    print(df1_top10.head(50))
    print()



    #I manually go through the ef lists ordered above to find these.

    #7% - bad ef error but good dice:
    #eg1 = df.iloc[[55]]
    #eg2 = df.iloc[[10]]
    #eg3 = df.iloc[[12]]

    #write_dice_ef_dissociate_eg(eg1, source_images_path, source_labels_path, pred_disks_file, output_path+'_bad_ef_good_dice')
    #write_dice_ef_dissociate_eg(eg2, source_images_path, source_labels_path, pred_disks_file, output_path+'_bad_ef_good_dice')
    #write_dice_ef_dissociate_eg(eg3, source_images_path, source_labels_path, pred_disks_file, output_path+'_bad_ef_good_dice')

    # 20% - bad ef error but good dice:
    #eg1 = df.iloc[[71]]
    #eg2 = df.iloc[[55]]
    #eg3 = df.iloc[[74]]
    #write_dice_ef_dissociate_eg(eg1, source_images_path, source_labels_path, pred_disks_file, output_path+'_bad_ef_good_dice')
    #write_dice_ef_dissociate_eg(eg2, source_images_path, source_labels_path, pred_disks_file, output_path+'_bad_ef_good_dice')
    #write_dice_ef_dissociate_eg(eg3, source_images_path, source_labels_path, pred_disks_file, output_path+'_bad_ef_good_dice')

    # 100% - bad ef error but good dice:
    # eg1 = df.iloc[[83]]
    # eg2 = df.iloc[[73]]  #selected
    # print(eg2[['EF_error', 'ED_dice', 'ES_dice', 'EDV_gt', 'ESV_gt']])
    # print(eg2[['EDV_pred', 'ESV_pred', 'Ejection_Fraction_gt', 'Ejection_Fraction_pred']])
    # write_dice_ef_dissociate_eg(eg2, source_images_path, source_labels_path, pred_disks_file, output_path+'_bad_ef_good_dice')

    print()
    print()
    # i.e. EF_error low to High,  ED_Dice and ES_dice Low to high
    print('Good EF error with bad dice scores:')
    df1 = df.sort_values(['EF_error', 'ED_dice', 'ES_dice'], ascending=[True, True, True])
    df1_top10 = df1[['EF_error', 'ED_dice', 'ES_dice']]
    print(df1_top10.head(50))
    print()

    # 20% - good ef error but bad dice:
    #eg1 = df.iloc[[56]]
    #eg2 = df.iloc[[17]]
    #write_dice_ef_dissociate_eg(eg1, source_images_path, source_labels_path, pred_disks_file, output_path+'_good_ef_bad_dice')
    #write_dice_ef_dissociate_eg(eg2, source_images_path, source_labels_path, pred_disks_file, output_path+'_good_ef_bad_dice')
    #write_dice_ef_dissociate_eg(eg3, source_images_path, source_labels_path, pred_disks_file, output_path+'_good_ef_bad_dice')

    # 5% - bad ef error but gooo dice:
    # eg1 = df.iloc[[21]]
    # print(eg1[['EF_error', 'ED_dice', 'ES_dice', 'EDV_gt', 'ESV_gt']])
    # print(eg1[['EDV_pred', 'ESV_pred', 'Ejection_Fraction_gt', 'Ejection_Fraction_pred']])
    # write_dice_ef_dissociate_eg(eg1, source_images_path, source_labels_path, pred_disks_file, output_path+'_good_ef_bad_dice')

    #7% - good ef error but bad dice:
    eg1 = df.iloc[[58]]   #SELECTED
    print(eg1[['EF_error', 'ED_dice', 'ES_dice', 'EDV_gt', 'ESV_gt']])
    print(eg1[['EDV_pred', 'ESV_pred', 'Ejection_Fraction_gt', 'Ejection_Fraction_pred']])
    write_dice_ef_dissociate_eg(eg1, source_images_path, source_labels_path, pred_disks_file, output_path+'_good_ef_bad_dice')


def write_dice_ef_dissociate_eg(df_eg, source_images_path, source_labels_path, pred_disks_path, save_folder):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    ED_frame = df_eg['ED_frame'].iloc[0]
    ES_frame = df_eg['ES_frame'].iloc[0]

    image_ED = os.path.join(source_images_path, ED_frame)
    image_ES = os.path.join(source_images_path, ES_frame)

    label_ED = os.path.join(source_labels_path, ED_frame)
    label_ES = os.path.join(source_labels_path, ES_frame)

    #Read label file
    mask_ED = cv2.imread(label_ED)
    mask_ED = mask_ED.astype(np.uint8)
    mask_ED = mask_ED[:, :, 0]
    mask_ES = cv2.imread(label_ES)
    mask_ES = mask_ES.astype(np.uint8)
    mask_ES = mask_ES[:, :, 0]

    #Read image file
    img_ED = cv2.imread(image_ED)
    img_ED = cv2.cvtColor(img_ED, cv2.COLOR_BGR2RGB)
    img_ES = cv2.imread(image_ES)
    img_ES = cv2.cvtColor(img_ES, cv2.COLOR_BGR2RGB)

    # get disks file and plot disks on ed frame and es frame:
    disks_dict = load_json(pred_disks_path)
    disks_ED = disks_dict[ED_frame]
    disks_ES = disks_dict[ES_frame]

    #ED:
    (vol, poly_points, minmaxline,
     midpointline, segments) = disks_dict[ED_frame]
    if poly_points is not None:
        img = annotate_image(np.zeros(img_ED.shape), #img_ED,
                             poly_points,
                             None,
                             midpointline,
                             segments,
                             is_binary_image=False)

    save_name = ED_frame[0:ED_frame.index('.png')] + '_disks' + '.png'
    cv2.imwrite(os.path.join(save_folder, save_name), img)

    img_curves = get_overlay_image_pts(img_ED.copy(), mask_ED, poly_points)
    out_file = os.path.join(save_folder, ED_frame.replace('.png', '') + '_2.png')
    cv2.imwrite(out_file, img_curves)

    # shift_polypoints
    poly_points_shifted = []
    for pt in poly_points:
        poly_points_shifted.append([[pt[0][0] + 30, pt[0][1]]])
    img_curves = get_overlay_image_pts(img_ED.copy(), mask_ED, poly_points_shifted)
    out_file = os.path.join(save_folder, ED_frame.replace('.png', '') + '_3.png')
    cv2.imwrite(out_file, img_curves)
    mask_shift = draw_poly_on_image(np.zeros(img_ED.shape), poly_points_shifted, (255, 255, 255))
    mask_shift = fill_poly_on_image(np.zeros(img_ED.shape), poly_points_shifted, (255, 255, 255))
    mask_shift = get_binary_image(mask_shift)
    dc_shifted = measurements.compute_Dice_coefficient(mask_ED, mask_shift)
    print(f'ED_DC_Shifted: {dc_shifted}')

    #ES:
    (vol, poly_points, minmaxline,
     midpointline, segments) = disks_dict[ES_frame]
    if poly_points is not None:
        img = annotate_image(np.zeros(img_ED.shape), #img_ES,
                             poly_points,
                             None,
                             midpointline,
                             segments,
                             is_binary_image=False)

    save_name = ES_frame[0:ED_frame.index('.png')] + '_disks' + '.png'
    cv2.imwrite(os.path.join(save_folder, save_name), img)

    img_curves = get_overlay_image_pts(img_ES.copy(), mask_ES, poly_points)
    out_file = os.path.join(save_folder, ES_frame.replace('.png', '') + '_2.png')
    cv2.imwrite(out_file, img_curves)

    # shift_polypoints
    poly_points_shifted = []
    for pt in poly_points:
        poly_points_shifted.append([[pt[0][0] + 30, pt[0][1]]])
    img_curves = get_overlay_image_pts(img_ES.copy(), mask_ES, poly_points_shifted)
    out_file = os.path.join(save_folder, ES_frame.replace('.png', '') + '_3.png')
    cv2.imwrite(out_file, img_curves)
    mask_shift = draw_poly_on_image(np.zeros(img_ES.shape), poly_points_shifted, (255, 255, 255))
    mask_shift = fill_poly_on_image(np.zeros(img_ES.shape), poly_points_shifted, (255, 255, 255))
    mask_shift = get_binary_image(mask_shift)
    dc_shifted = measurements.compute_Dice_coefficient(mask_ES, mask_shift)
    print(f'ES_DC_Shifted: {dc_shifted}')





def get_mae_between_results(res_file1, res_file2, column_name1, column_name2, exclude_first = False):

    dtype_dict = {
            'percentage_unlabelled': 'int64',
            'num_unlabelled': 'int64',
            'percentage_train': 'float64',
            'train': 'int64',
            'dice_test': 'float64',
            'dice_test_std': 'float64',
            'dice_con': 'float64',
            'dice_con_std': 'float64',
            'dice_con_top100': 'float64',
            'dice_con_std_top100': 'float64',
            'hd_test': 'float64',
            'hd_test_std': 'float64',
            'hd_con': 'float64',
            'hd_con_std': 'float64',
            'hd_con_top100': 'float64',
            'hd_con_std_top100': 'float64',
            'val': 'int64',
            'test': 'int64',
            'consensus': 'int64',
            'x_label': 'str',
            'hue_label': 'str',
            'ef_error': 'float64',
            'ef_error_std': 'float64',
            'ed_vol_error': 'float64',
            'ed_vol_error_std': 'float64',
            'es_vol_error': 'float64',
            'es_vol_error_std': 'float64'
    }

    mae = 0
    start = 0

    if exclude_first:
        start = 1

    df1 = pd.read_csv(res_file1, dtype=dtype_dict)
    df2 = pd.read_csv(res_file2, dtype=dtype_dict)

    l1 = np.asarray(df1[column_name1].tolist())
    l2 = np.asarray(df2[column_name1].tolist())

    mae = mean_absolute_error(l1[start:], l2[start:])
    print(f'{column_name1}: {mae}')

    ####################################

    l1 = np.asarray(df1[column_name2].tolist())
    l2 = np.asarray(df2[column_name2].tolist())

    mae = mean_absolute_error(l1[start:], l2[start:])
    print(f'{column_name2}: {mae}')


def compute_MAE_for_Results():

    a = f'run_SimCLR_Unity_unet-enc_bce_True_batch20/results_on_enc_1900.csv'
    b = f'run_SimCLR_Unity_unet-enc_bce_True_batch64/results_on_enc_1900.csv'
    c = f'run_SimCLR_Unity_unet_bce_True_batch20/results_on_enc_1900.csv'

    print('SimCL-enc--20')
    get_mae_between_results(a,
                            a, 'dice_con', 'dice_con_top100', False)

    get_mae_between_results(a,
                            b, 'dice_con', 'dice_con_top100', False)

    get_mae_between_results(a,
                            c, 'dice_con', 'dice_con_top100', False)
    print('--------------------')
    print('SimCLR-enc-64')
    get_mae_between_results(b,
                            a, 'dice_con', 'dice_con_top100', False)

    get_mae_between_results(b,
                            b, 'dice_con', 'dice_con_top100', False)

    get_mae_between_results(b,
                            c, 'dice_con', 'dice_con_top100', False)
    print('--------------------')
    print('SimCLR-20')
    get_mae_between_results(c,
                            a, 'dice_con', 'dice_con_top100', False)

    get_mae_between_results(c,
                            b, 'dice_con', 'dice_con_top100', False)

    get_mae_between_results(c,
                            c, 'dice_con', 'dice_con_top100', False)
    print('--------------------')


    print('-------------------------------------------------------')
    print('-------------------------------------------------------')

    a = f'run_SimCLR_Unity_unet_bce_True_batch8/results_on_enc_1900.csv'
    b = f'run_SimCLR_Unity_unet_bce_True_batch16/results_on_enc_1900.csv'
    c = f'run_SimCLR_Unity_unet_bce_True_batch20/results_on_enc_1900.csv'
    print('SimCLR-8')
    get_mae_between_results(a,
                            a, 'dice_con', 'dice_con_top100')

    get_mae_between_results(a,
                            b, 'dice_con', 'dice_con_top100')

    get_mae_between_results(a,
                            c, 'dice_con', 'dice_con_top100')
    print('--------------------')
    print('SimCLR-16')
    get_mae_between_results(b,
                            a, 'dice_con', 'dice_con_top100')

    get_mae_between_results(b,
                            b, 'dice_con', 'dice_con_top100')

    get_mae_between_results(b,
                            c, 'dice_con', 'dice_con_top100')
    print('--------------------')
    print('SimCLR-20')
    get_mae_between_results(c,
                            a, 'dice_con', 'dice_con_top100')

    get_mae_between_results(c,
                            b, 'dice_con', 'dice_con_top100')

    get_mae_between_results(c,
                            c, 'dice_con', 'dice_con_top100')
    print('--------------------')

    print('-------------------------------------------------------')
    print('-------------------------------------------------------')

    a = f'run_BTwin_Echo_unet_bce_True_batch8/results_on_enc_30000.csv'
    b = f'run_BTwin_Echo_unet_bce_True_batch16/results_on_enc_30000.csv'
    c = f'run_BTwin_Echo_unet_bce_True_batch20/results_on_enc_30000.csv'
    print('BTwins-8')
    get_mae_between_results(a,
                            a, 'dice_con', 'dice_con_top100', False)

    get_mae_between_results(a,
                            b, 'dice_con', 'dice_con_top100', False)

    get_mae_between_results(a,
                            c, 'dice_con', 'dice_con_top100', False)
    print('--------------------')
    print('BTwins-16')
    get_mae_between_results(b,
                            a, 'dice_con', 'dice_con_top100', False)

    get_mae_between_results(b,
                            b, 'dice_con', 'dice_con_top100', False)

    get_mae_between_results(b,
                            c, 'dice_con', 'dice_con_top100', False)
    print('--------------------')
    print('BTwins-20')
    get_mae_between_results(c,
                            a, 'dice_con', 'dice_con_top100', False)

    get_mae_between_results(c,
                            b, 'dice_con', 'dice_con_top100', False)

    get_mae_between_results(c,
                            c, 'dice_con', 'dice_con_top100', False)
    print('--------------------')

    return


def get_ef_errors_for_base(base_folder):

    data_dict = {}

    percent_folders = os.listdir(base_folder)

    for perc_folder in percent_folders:
        if 'baseline_with_' not in perc_folder.lower():
            continue

        results_folder = os.path.join(base_folder, perc_folder)
        if not os.path.exists(results_folder):
            continue

        files = os.listdir(results_folder)

        res_file = 'labelled_dataset_counts.json'
        res_ef_file = f'baseline_model_results_run_0/Consensus_EF_Vol_results_all.json'
        res_ef_file_run_1 = f'baseline_model_results_run_1/Consensus_EF_Vol_results_all.json'
        if not (res_file in files or res_ef_file in files):
            continue

        res_dict = load_json(os.path.join(results_folder, res_file))

        percentage_train = res_dict['perc_train']
        num_train = res_dict['train']

        dict1 = load_json(os.path.join(results_folder, res_ef_file))
        dict2 = load_json(os.path.join(results_folder, res_ef_file_run_1))
        ef_err_list, edv_err_list, esv_err_list = [], [], []
        ef_list, edv_list, esv_list = [], [], []
        count_no_ef = 0
        for key, value in dict1.items():
            value1 = dict2[key]

            if key == '02-1c89c6b90d8e37afdb0e20db1fbed0127b208cdbb687922623d2e905d4a679ce':
                count_no_ef += 1
                continue
            #The above file was problematic. The EDV frame was not being segmented well, which resulted in
            #the

            if value['Ejection Fraction pred'] == 0 or value1['Ejection Fraction pred'] == 0:
                count_no_ef += 1
                continue

            ef_err_list.append((value['EF error'] + value1['EF error']) / 2.0)
            edv_err_list.append((value['EDV error'] + value1['EDV error']) / 2.0)
            esv_err_list.append((value['ESV error'] + value1['ESV error']) / 2.0)
            ef_list.append((value['Ejection Fraction pred'] + value1['Ejection Fraction pred']) / 2.0)
            edv_list.append((value['EDV pred'] + value1['EDV pred']) / 2.0)
            esv_list.append((value['ESV pred'] + value1['ESV pred']) / 2.0)

        x_tick = f'{num_train}({percentage_train}%)'

        data_dict[percentage_train] = (percentage_train, num_train,
            x_tick, ef_err_list, edv_err_list, esv_err_list, ef_list, edv_list, esv_list, count_no_ef)

    return data_dict

def get_ef_errors_for_exp(exp_folder):

    data_dict = {}

    downstream_folders = os.listdir(exp_folder)

    for downstream_folder in downstream_folders:
        if 'finetuning_' not in downstream_folder.lower():
            continue

        results_folder = os.path.join(exp_folder, downstream_folder)
        if not os.path.exists(results_folder):
            continue

        files = os.listdir(results_folder)

        res_file = 'encoder_fine_tune_results.json'
        res_ef_file = f'Consensus_EF_Vol_results_all_{0}.json'
        res_ef_file_run_1 = f'Consensus_EF_Vol_results_all_{1}.json'
        if not (res_file in files or res_ef_file in files):
            continue

        res_dict = load_json(os.path.join(results_folder, res_file))

        percentage_unlabelled = res_dict['percentage_unlabelled']
        percentage_train = res_dict['percentage_train']
        num_train = res_dict['num_train']

        dict1 = load_json(os.path.join(results_folder, res_ef_file))
        dict2 = load_json(os.path.join(results_folder, res_ef_file_run_1))
        ef_err_list, edv_err_list, esv_err_list = [], [], []
        ef_list, edv_list, esv_list = [], [], []
        count_no_ef = 0
        for key, value in dict1.items():
            value1 = dict2[key]

            if key=='02-1c89c6b90d8e37afdb0e20db1fbed0127b208cdbb687922623d2e905d4a679ce':
                count_no_ef += 1
                continue

            if value['Ejection Fraction pred'] == 0 or value1['Ejection Fraction pred'] == 0:
                count_no_ef+=1
                continue

            ef_err_list.append((value['EF error'] + value1['EF error']) / 2.0)
            edv_err_list.append((value['EDV error'] + value1['EDV error']) / 2.0)
            esv_err_list.append((value['ESV error'] + value1['ESV error']) / 2.0)
            ef_list.append((value['Ejection Fraction pred'] + value1['Ejection Fraction pred']) / 2.0)
            edv_list.append((value['EDV pred'] + value1['EDV pred']) / 2.0)
            esv_list.append((value['ESV pred'] + value1['ESV pred']) / 2.0)

            # ef_err_list.append(value['EF error'])
            # edv_err_list.append(value['EDV error'])
            # esv_err_list.append(value['ESV error'])
            # ef_list.append(value['Ejection Fraction pred'])
            # edv_list.append(value['EDV pred'])
            # esv_list.append(value['ESV pred'])

        x_tick = f'{num_train}({percentage_train}%)'

        data_dict[percentage_train] = (percentage_train, num_train,
            x_tick, ef_err_list, edv_err_list, esv_err_list, ef_list, edv_list, esv_list, count_no_ef)

    return data_dict



def get_data_lists(data_dict):

    data_lists = []
    percentages = []
    for key, value in data_dict.items():
        percent = key
        ef_errors = value[3]
        data_lists.append(ef_errors)
        percentages.append(key)

    percentages, data_lists = zip(*sorted(zip(percentages, data_lists)))

    return percentages, data_lists

# each plot returns a dictionary, use plt.setp()
# function to assign the color code
# for all properties of the box plot of particular group
# use the below function to set color for particular group,
# by iterating over all properties of the box plot
def define_box_properties(bplot, colour_code, label):
    for k, v in bplot.items():
        plt.setp(bplot.get(k), color=colour_code)

    # 'boxes', 'whiskers', 'fliers', 'medians', 'caps'
    for flier in bplot['fliers']:
         flier.set(markeredgecolor=colour_code, markersize = 13, markeredgewidth = 2.5)  #markeredgecolor  markerfacecolor

    #for patch in bplot['boxes']:
        # patch.set_facecolor(colour)
        #patch.set(facecolor=colour_code)
        #patch.set(linewidth=2.5)

    for median in bplot['medians']:
        median.set(linewidth=2.5, color='black')

    for mean in bplot['means']:
        mean.set(linewidth=2.5, color='white', linestyle = (0, (1, 1)))

    for whisker in bplot['whiskers']:
        whisker.set(linewidth=3)

    for cap in bplot['caps']:
        cap.set(linewidth=3)

    for cap in bplot['caps']:
        cap.set_xdata(cap.get_xdata() + np.array([-.10, .10]))

    # use plot function to draw a small line to name the legend.
    #Fake the legend
    plt.plot([], c=colour_code, linewidth=15, label=label)
    plt.legend()

def write_csv_ef_errors(save_folder, data_dict, method_name, unlabelled_count):
    data_dict = dict(sorted(data_dict.items()))

    df = pd.DataFrame(columns=['num_unlabelled',
                               'perc_train', 'train',
                               'count_no_ef',
                               'ef_err', 'ef_err_std',
                               'edv_err', 'edv_err_std',
                               'esv_err', 'esv_err_std',
                               'ef', 'ef_std',
                               'edv', 'edv_std',
                               'esv', 'esv_std'])

    df['num_unlabelled'] = df['num_unlabelled'].astype(int)
    df['perc_train'] = df['perc_train'].astype(int)
    df['train'] = df['train'].astype(int)
    df['count_no_ef'] = df['count_no_ef'].astype(int)
    df['ef_err'] = df['ef_err'].astype(float)
    df['ef_err_std'] = df['ef_err_std'].astype(float)
    df['edv_err'] = df['edv_err'].astype(float)
    df['edv_err_std'] = df['edv_err_std'].astype(float)
    df['esv_err'] = df['esv_err'].astype(float)
    df['esv_err_std'] = df['esv_err_std'].astype(float)
    df['ef'] = df['ef'].astype(float)
    df['ef_std'] = df['ef_std'].astype(float)
    df['edv'] = df['edv'].astype(float)
    df['edv_std'] = df['edv_std'].astype(float)
    df['esv'] = df['esv'].astype(float)
    df['esv_std'] = df['esv_std'].astype(float)

    for key, value in data_dict.items():
        (percentage_train, num_train,
         x_tick,
         ef_err_list,
         edv_err_list, esv_err_list,
         ef_list,
         edv_list, esv_list,
         count_no_ef) = value

        row = {'num_unlabelled': unlabelled_count,
               'perc_train': percentage_train, 'train': num_train,
               'count_no_ef': count_no_ef,
               'ef_err': np.mean(ef_err_list), 'ef_err_std': np.std(ef_err_list),
               'edv_err': np.mean(edv_err_list), 'edv_err_std': np.std(edv_err_list),
               'esv_err': np.mean(esv_err_list), 'esv_err_std': np.std(esv_err_list),
               'ef': np.mean(ef_list), 'ef_std': np.std(ef_list),
               'edv': np.mean(edv_list), 'edv_std': np.std(edv_list),
               'esv': np.mean(esv_list), 'esv_std': np.std(esv_list)
               }
        df = df.append(row, ignore_index=True)

    df.to_csv(os.path.join(save_folder, f'{method_name}_{unlabelled_count}_ef_avg.csv'), index=False)

def grouped_box_plot_figure_enc_vs_dec():
    plt.clf()

    names = ['Rand', 'SimCLR-enc-20', 'SimCLR-enc-64', 'SimCLR-20']
    colours = ['slategray', 'peru', 'tomato', 'orange']

    compare1 = f'run_SimCLR_Unity_unet-enc_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/'
    compare2 = f'run_SimCLR_Unity_unet-enc_bce_True_batch64/Exp_with_1900_percent_unlabelled_data/'
    compare3 = f'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/'

    output_path = f'Results_Boxplot_Encoder_vs_Full'
    if not os.path.exists(output_path):
       os.mkdir(output_path)

    base_dict = get_ef_errors_for_base('Baseline_444_Final')
    data_dict1 = get_ef_errors_for_exp(compare1)
    data_dict2 = get_ef_errors_for_exp(compare2)
    data_dict3 = get_ef_errors_for_exp(compare3)

    write_csv_ef_errors(output_path, base_dict, names[0], 0)
    write_csv_ef_errors(output_path, data_dict1, names[1], 1900)
    write_csv_ef_errors(output_path, data_dict2, names[2], 1900)
    write_csv_ef_errors(output_path, data_dict3, names[3], 1900)

    percentages_base, data_lists_base = get_data_lists(base_dict)
    percentages1, data_lists1 = get_data_lists(data_dict1)
    percentages2, data_lists2 = get_data_lists(data_dict2)
    percentages3, data_lists3 = get_data_lists(data_dict3)

    fig, ax = plt.subplots(figsize=(18, 14), dpi=300)
    ax.set_title('Consensus Dataset', fontsize=56)  # , fontweight='bold')

    label_size = 56  # 40
    tick_size = 52  # 34
    inset_tick_size = 34  # 28 #22
    legend_font_size = 24  # 26

    # test1 = np.array(np.arange(len(data_lists_base))) * 3.0 - 0.80
    # test2 = np.array(np.arange(len(data_lists_base))) * 3.0 - 0.20
    # test3 = np.array(np.arange(len(data_lists_base))) * 3.0 + 0.40
    # test4 = np.array(np.arange(len(data_lists_base))) * 3.0 + 1


    #boxprops = dict(linewidth=3)
    #whiskerprops = dict(linewidth=3) #linestyle='--'
    #capprops = dict(linewidth=3)  # linestyle='--'

    gap = 0.60

    base_plot = plt.boxplot(data_lists_base,
                           positions=np.array(
                           np.arange(len(data_lists_base))) * 3.0 - 0.80,
                           patch_artist=True, showmeans=True, meanline=True,
                           widths=0.555)

    plot1 = plt.boxplot(data_lists1,
                            positions=np.array(
                            np.arange(len(data_lists1))) * 3.0 - 0.20,
                            patch_artist=True, showmeans=True, meanline=True,
                            widths=0.555)

    plot2 = plt.boxplot(data_lists2,
                            positions=np.array(
                            np.arange(len(data_lists2))) * 3.0 + 0.40,
                            patch_artist=True, showmeans=True, meanline=True,
                            widths=0.555)

    plot3 = plt.boxplot(data_lists3,
                            positions=np.array(
                            np.arange(len(data_lists3))) * 3.0 + 1,
                            patch_artist=True, showmeans=True, meanline=True,
                            widths=0.555)
                            #, whiskerprops=whiskerprops,
                            # capprops=capprops)


    ticks = []
    for perc in percentages1:
        ticks.append(str(perc))

    # setting colors for each groups
    define_box_properties(base_plot, colours[0], names[0])
    define_box_properties(plot1, colours[1], names[1])
    define_box_properties(plot2, colours[2], names[2])
    define_box_properties(plot3, colours[3], names[3])

    # set the x label values
    plt.xticks(np.arange(0, len(ticks) * 3, 3), ticks, fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    # set the limit for x axis
    plt.xlim(-2, len(ticks) * 3)

    plt.xlabel(' % Labelled Data', fontsize=label_size, labelpad=20)
    plt.ylabel('EF Error', fontsize=label_size)

    # Make the tick lines bigger for all ticks
    ax.xaxis.set_tick_params(width=5, size=20)
    ax.yaxis.set_tick_params(width=5, size=20)

    # make ths spines thicker:
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    # PLOT LEGEND
    # pos = ax.get_position()
    # ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.92])
    # legend = plt.legend(fontsize=legend_font_size, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    ax.get_legend().remove()

    plt.savefig(os.path.join(output_path, 'boxplot_enc_vs_full.png'), dpi=300, bbox_inches='tight')
    plt.clf()


def grouped_box_plot_figure_batch_size_simclr():
    plt.clf()

    names = ['Rand', 'SimCLR-8', 'SimCLR-16', 'SimCLR-20']
    colours = ['slategray', 'tomato', 'peru', 'orange']

    compare1 = f'run_SimCLR_Unity_unet_bce_True_batch8/Exp_with_1900_percent_unlabelled_data/'
    compare2 = f'run_SimCLR_Unity_unet_bce_True_batch16/Exp_with_1900_percent_unlabelled_data/'
    compare3 = f'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/'

    output_path = f'Results_Boxplot_batch_size'
    if not os.path.exists(output_path):
       os.mkdir(output_path)

    base_dict = get_ef_errors_for_base('Baseline_444_Final')
    data_dict1 = get_ef_errors_for_exp(compare1)
    data_dict2 = get_ef_errors_for_exp(compare2)
    data_dict3 = get_ef_errors_for_exp(compare3)

    write_csv_ef_errors(output_path, base_dict, names[0], 0)
    write_csv_ef_errors(output_path, data_dict1, names[1], 1900)
    write_csv_ef_errors(output_path, data_dict2, names[2], 1900)
    write_csv_ef_errors(output_path, data_dict3, names[3], 1900)

    percentages_base, data_lists_base = get_data_lists(base_dict)
    percentages1, data_lists1 = get_data_lists(data_dict1)
    percentages2, data_lists2 = get_data_lists(data_dict2)
    percentages3, data_lists3 = get_data_lists(data_dict3)

    fig, ax = plt.subplots(figsize=(18, 14), dpi=300)
    ax.set_title('Consensus Dataset', fontsize=56)  # , fontweight='bold')

    label_size = 56  # 40
    tick_size = 52  # 34
    inset_tick_size = 34  # 28 #22
    legend_font_size = 24  # 26

    # test1 = np.array(np.arange(len(data_lists_base))) * 3.0 - 0.80
    # test2 = np.array(np.arange(len(data_lists_base))) * 3.0 - 0.20
    # test3 = np.array(np.arange(len(data_lists_base))) * 3.0 + 0.40
    # test4 = np.array(np.arange(len(data_lists_base))) * 3.0 + 1


    #boxprops = dict(linewidth=3)
    #whiskerprops = dict(linewidth=3) #linestyle='--'
    #capprops = dict(linewidth=3)  # linestyle='--'

    gap = 0.60

    base_plot = plt.boxplot(data_lists_base,
                           positions=np.array(
                           np.arange(len(data_lists_base))) * 3.0 - 0.80,
                           patch_artist=True, showmeans=True, meanline=True,
                           widths=0.555)

    plot1 = plt.boxplot(data_lists1,
                            positions=np.array(
                            np.arange(len(data_lists1))) * 3.0 - 0.20,
                            patch_artist=True, showmeans=True, meanline=True,
                            widths=0.555)

    plot2 = plt.boxplot(data_lists2,
                            positions=np.array(
                            np.arange(len(data_lists2))) * 3.0 + 0.40,
                            patch_artist=True, showmeans=True, meanline=True,
                            widths=0.555)

    plot3 = plt.boxplot(data_lists3,
                            positions=np.array(
                            np.arange(len(data_lists3))) * 3.0 + 1,
                            patch_artist=True, showmeans=True, meanline=True,
                            widths=0.555)
                            #, whiskerprops=whiskerprops,
                            # capprops=capprops)


    ticks = []
    for perc in percentages1:
        ticks.append(str(perc))

    # setting colors for each groups
    define_box_properties(base_plot, colours[0], names[0])
    define_box_properties(plot1, colours[1], names[1])
    define_box_properties(plot2, colours[2], names[2])
    define_box_properties(plot3, colours[3], names[3])

    # set the x label values
    plt.xticks(np.arange(0, len(ticks) * 3, 3), ticks, fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    # set the limit for x axis
    plt.xlim(-2, len(ticks) * 3)

    plt.xlabel(' % Labelled Data', fontsize=label_size, labelpad=20)
    plt.ylabel('EF Error', fontsize=label_size)

    # Make the tick lines bigger for all ticks
    ax.xaxis.set_tick_params(width=5, size=20)
    ax.yaxis.set_tick_params(width=5, size=20)

    # make ths spines thicker:
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    # PLOT LEGEND
    # pos = ax.get_position()
    # ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.92])
    # legend = plt.legend(fontsize=legend_font_size, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    ax.get_legend().remove()

    plt.savefig(os.path.join(output_path, 'boxplot_batch_sizes_simclr.png'), dpi=300, bbox_inches='tight')
    plt.clf()

def grouped_box_plot_figure_batch_size_btwins():
    plt.clf()

    names = ['Rand', 'BTwins-8', 'BTwins-16', 'BTwins-20']
    colours = ['slategray', 'darkturquoise', 'cornflowerblue', 'blue']

    compare1 = f'run_BTwin_Echo_unet_bce_True_batch8/Exp_with_30000_percent_unlabelled_data/'
    compare2 = f'run_BTwin_Echo_unet_bce_True_batch16/Exp_with_30000_percent_unlabelled_data/'
    compare3 = f'run_BTwin_Echo_unet_bce_True_batch20/Exp_with_30000_percent_unlabelled_data/'

    output_path = f'Results_Boxplot_batch_size'
    if not os.path.exists(output_path):
       os.mkdir(output_path)

    base_dict = get_ef_errors_for_base('Baseline_444_Final')
    data_dict1 = get_ef_errors_for_exp(compare1)
    data_dict2 = get_ef_errors_for_exp(compare2)
    data_dict3 = get_ef_errors_for_exp(compare3)

    write_csv_ef_errors(output_path, base_dict, names[0], 0)
    write_csv_ef_errors(output_path, data_dict1, names[1], 30000)
    write_csv_ef_errors(output_path, data_dict2, names[2], 30000)
    write_csv_ef_errors(output_path, data_dict3, names[3], 30000)

    percentages_base, data_lists_base = get_data_lists(base_dict)
    percentages1, data_lists1 = get_data_lists(data_dict1)
    percentages2, data_lists2 = get_data_lists(data_dict2)
    percentages3, data_lists3 = get_data_lists(data_dict3)

    fig, ax = plt.subplots(figsize=(18, 14), dpi=300)

    label_size = 34  # 36 #30
    tick_size = 30  # 32 #26
    inset_tick_size = 34  # 28 #22
    legend_font_size = 24  # 26

    # test1 = np.array(np.arange(len(data_lists_base))) * 3.0 - 0.80
    # test2 = np.array(np.arange(len(data_lists_base))) * 3.0 - 0.20
    # test3 = np.array(np.arange(len(data_lists_base))) * 3.0 + 0.40
    # test4 = np.array(np.arange(len(data_lists_base))) * 3.0 + 1


    #boxprops = dict(linewidth=3)
    #whiskerprops = dict(linewidth=3) #linestyle='--'
    #capprops = dict(linewidth=3)  # linestyle='--'

    gap = 0.60

    base_plot = plt.boxplot(data_lists_base,
                           positions=np.array(
                           np.arange(len(data_lists_base))) * 3.0 - 0.80,
                           patch_artist=True, showmeans=True, meanline=True,
                           widths=0.555)

    plot1 = plt.boxplot(data_lists1,
                            positions=np.array(
                            np.arange(len(data_lists1))) * 3.0 - 0.20,
                            patch_artist=True, showmeans=True, meanline=True,
                            widths=0.555)

    plot2 = plt.boxplot(data_lists2,
                            positions=np.array(
                            np.arange(len(data_lists2))) * 3.0 + 0.40,
                            patch_artist=True, showmeans=True, meanline=True,
                            widths=0.555)

    plot3 = plt.boxplot(data_lists3,
                            positions=np.array(
                            np.arange(len(data_lists3))) * 3.0 + 1,
                            patch_artist=True, showmeans=True, meanline=True,
                            widths=0.555)
                            #, whiskerprops=whiskerprops,
                            # capprops=capprops)


    ticks = []
    for perc in percentages1:
        ticks.append(str(perc))

    # setting colors for each groups
    define_box_properties(base_plot, colours[0], names[0])
    define_box_properties(plot1, colours[1], names[1])
    define_box_properties(plot2, colours[2], names[2])
    define_box_properties(plot3, colours[3], names[3])

    # set the x label values
    plt.xticks(np.arange(0, len(ticks) * 3, 3), ticks, fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    # set the limit for x axis
    plt.xlim(-2, len(ticks) * 3)

    plt.xlabel(' % Labelled Data', fontsize=label_size, labelpad=20)
    plt.ylabel('EF Error', fontsize=label_size)

    # Make the tick lines bigger for all ticks
    ax.xaxis.set_tick_params(width=5, size=20)
    ax.yaxis.set_tick_params(width=5, size=20)

    # make ths spines thicker:
    for axis in ['bottom', 'left']:  # ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(4)

    # PLOT LEGEND
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.92])
    legend = plt.legend(fontsize=legend_font_size, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    plt.savefig(os.path.join(output_path, 'boxplot_batch_sizes_btwin.png'), dpi=300, bbox_inches='tight')
    plt.clf()

def grouped_box_plot_figure_unlabelled():
    plt.clf()

    names = ['Rand', 'Unlabelled-B (100K)', 'Unlabelled-A (60K)', 'Unlabelled-A (2057)', 'Unlabelled-B (2057)']
    colours = ['slategray', 'crimson', 'sienna', 'orange', 'darkturquoise']

    compare1 = f'run_SimCLR_A4CHLV_unet_bce_True_batch{20}/Exp_with_100000_percent_unlabelled_data/'
    compare2 = f'run_SimCLR_UnityFrames_unet_bce_True_batch{20}/Exp_with_60000_percent_unlabelled_data/'
    compare3 = f'run_SimCLR_Unity_unet_bce_True_batch{20}/Exp_with_1900_percent_unlabelled_data/'
    compare4 = f'run_SimCLR_A4CHLV_1900_unet_bce_True_batch{20}/Exp_with_1900_percent_unlabelled_data/'


    output_path = f'Results_Boxplot_unlabelled'
    if not os.path.exists(output_path):
       os.mkdir(output_path)

    base_dict = get_ef_errors_for_base('Baseline_444_Final')
    data_dict1 = get_ef_errors_for_exp(compare1)
    data_dict2 = get_ef_errors_for_exp(compare2)
    data_dict3 = get_ef_errors_for_exp(compare3)
    data_dict4 = get_ef_errors_for_exp(compare4)

    write_csv_ef_errors(output_path, base_dict, names[0], 0)
    write_csv_ef_errors(output_path, data_dict1, names[1], 30000)
    write_csv_ef_errors(output_path, data_dict2, names[2], 30000)
    write_csv_ef_errors(output_path, data_dict3, names[3], 30000)
    write_csv_ef_errors(output_path, data_dict4, names[4], 30000)

    percentages_base, data_lists_base = get_data_lists(base_dict)
    percentages1, data_lists1 = get_data_lists(data_dict1)
    percentages2, data_lists2 = get_data_lists(data_dict2)
    percentages3, data_lists3 = get_data_lists(data_dict3)
    percentages4, data_lists4 = get_data_lists(data_dict4)

    fig, ax = plt.subplots(figsize=(18, 14), dpi=300)
    ax.set_title('Consensus Dataset', fontsize=56)  # , fontweight='bold')

    label_size = 56  # 40
    tick_size = 52  # 34
    inset_tick_size = 34  # 28 #22
    legend_font_size = 24  # 26

    # test1 = np.array(np.arange(len(data_lists_base))) * 3.0 - 0.80
    # test2 = np.array(np.arange(len(data_lists_base))) * 3.0 - 0.20
    # test3 = np.array(np.arange(len(data_lists_base))) * 3.0 + 0.40
    # test4 = np.array(np.arange(len(data_lists_base))) * 3.0 + 1


    #boxprops = dict(linewidth=3)
    #whiskerprops = dict(linewidth=3) #linestyle='--'
    #capprops = dict(linewidth=3)  # linestyle='--'

    gap = 0.60

    base_plot = plt.boxplot(data_lists_base,
                           positions=np.array(
                           np.arange(len(data_lists_base))) * 4.0 - 1.20,
                           patch_artist=True, showmeans=True, meanline=True,
                           widths=0.555)

    plot1 = plt.boxplot(data_lists1,
                            positions=np.array(
                            np.arange(len(data_lists1))) * 4.0 - 0.60,
                            patch_artist=True, showmeans=True, meanline=True,
                            widths=0.555)

    plot2 = plt.boxplot(data_lists2,
                            positions=np.array(
                            np.arange(len(data_lists2))) * 4.0 + 0,
                            patch_artist=True, showmeans=True, meanline=True,
                            widths=0.555)

    plot3 = plt.boxplot(data_lists3,
                            positions=np.array(
                            np.arange(len(data_lists3))) * 4.0 + 0.6,
                            patch_artist=True, showmeans=True, meanline=True,
                            widths=0.555)
                            #, whiskerprops=whiskerprops,
                            # capprops=capprops)

    plot4 = plt.boxplot(data_lists4,
                        positions=np.array(
                            np.arange(len(data_lists4))) * 4.0 + 1.2,
                        patch_artist=True, showmeans=True, meanline=True,
                        widths=0.555)


    ticks = []
    for perc in percentages1:
        ticks.append(str(perc))

    # setting colors for each groups
    define_box_properties(base_plot, colours[0], names[0])
    define_box_properties(plot1, colours[1], names[1])
    define_box_properties(plot2, colours[2], names[2])
    define_box_properties(plot3, colours[3], names[3])
    define_box_properties(plot4, colours[4], names[4])

    # set the x label values
    plt.xticks(np.arange(0, len(ticks) * 4, 4), ticks, fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    # set the limit for x axis
    plt.xlim(-3, len(ticks) * 4)

    plt.xlabel(' % Labelled Data', fontsize=label_size, labelpad=20)
    plt.ylabel('EF Error', fontsize=label_size)

    # Make the tick lines bigger for all ticks
    ax.xaxis.set_tick_params(width=5, size=20)
    ax.yaxis.set_tick_params(width=5, size=20)

    # make ths spines thicker:
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    # PLOT LEGEND
    # pos = ax.get_position()
    # ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.92])
    # legend = plt.legend(fontsize=legend_font_size, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    ax.get_legend().remove()

    plt.savefig(os.path.join(output_path, 'boxplot_unlabelled.png'), dpi=300, bbox_inches='tight')
    plt.clf()

def grouped_box_plot_figure_methods_compare():
    plt.clf()

    names = ['Rand', 'Barlow Twins', 'Masking-A', 'Masking-B', 'Rotation', 'SimCLR', 'Split']
    colours = ['slategray', 'cornflowerblue', 'salmon', 'orchid', 'mediumseagreen', 'orange', 'yellowgreen']

    compare1 = f'run_BTwin_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/'
    compare2 = f'run_Patch_rand_unity_patch_unet_bce_False_batch16/Exp_with_1900_percent_unlabelled_data/'
    compare3 = f'run_Patch_rand_unity_horizontal_unet_bce_False_batch16/Exp_with_1900_percent_unlabelled_data/'
    compare4 = f'run_ssl_rotation_unet_bce_False_batch16/Exp_with_1900_percent_unlabelled_data/'
    compare5 = f'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/'
    compare6 = f'run_Split2_unet_bce_False_batch16/Exp_with_1900_percent_unlabelled_data/'

    output_path = f'Results_Boxplot_method_compare'
    if not os.path.exists(output_path):
       os.mkdir(output_path)

    base_dict = get_ef_errors_for_base('Baseline_444_Final')
    data_dict1 = get_ef_errors_for_exp(compare1)
    data_dict2 = get_ef_errors_for_exp(compare2)
    data_dict3 = get_ef_errors_for_exp(compare3)
    data_dict4 = get_ef_errors_for_exp(compare4)
    data_dict5 = get_ef_errors_for_exp(compare5)
    data_dict6 = get_ef_errors_for_exp(compare6)

    write_csv_ef_errors(output_path, base_dict, names[0], 0)
    write_csv_ef_errors(output_path, data_dict1, names[1], 30000)
    write_csv_ef_errors(output_path, data_dict2, names[2], 30000)
    write_csv_ef_errors(output_path, data_dict3, names[3], 30000)
    write_csv_ef_errors(output_path, data_dict4, names[4], 30000)
    write_csv_ef_errors(output_path, data_dict5, names[5], 30000)
    write_csv_ef_errors(output_path, data_dict6, names[6], 30000)

    percentages_base, data_lists_base = get_data_lists(base_dict)
    percentages1, data_lists1 = get_data_lists(data_dict1)
    percentages2, data_lists2 = get_data_lists(data_dict2)
    percentages3, data_lists3 = get_data_lists(data_dict3)
    percentages4, data_lists4 = get_data_lists(data_dict4)
    percentages5, data_lists5 = get_data_lists(data_dict5)
    percentages6, data_lists6 = get_data_lists(data_dict6)

    fig, ax = plt.subplots(figsize=(22, 14), dpi=300)
    ax.set_title('Consensus Dataset', fontsize=56) #, fontweight='bold')

    label_size = 56#40
    tick_size = 52#34
    inset_tick_size = 34
    legend_font_size = 24

    # test1 = np.array(np.arange(len(data_lists_base))) * 3.0 - 0.80
    # test2 = np.array(np.arange(len(data_lists_base))) * 3.0 - 0.20
    # test3 = np.array(np.arange(len(data_lists_base))) * 3.0 + 0.40
    # test4 = np.array(np.arange(len(data_lists_base))) * 3.0 + 1


    #boxprops = dict(linewidth=3)
    #whiskerprops = dict(linewidth=3) #linestyle='--'
    #capprops = dict(linewidth=3)  # linestyle='--'

    gap = 0.60

    base_plot = plt.boxplot(data_lists_base,
                           positions=np.array(
                           np.arange(len(data_lists_base))) * 6.0 - 2.5,
                           patch_artist=True, showmeans=True, meanline=True,
                           widths=0.650)

    plot1 = plt.boxplot(data_lists1,
                            positions=np.array(
                            np.arange(len(data_lists1))) * 6.0 - 1.8,
                            patch_artist=True, showmeans=True, meanline=True,
                            widths=0.650)

    plot2 = plt.boxplot(data_lists2,
                            positions=np.array(
                            np.arange(len(data_lists2))) * 6.0 - 1.10,
                            patch_artist=True, showmeans=True, meanline=True,
                            widths=0.650)

    plot3 = plt.boxplot(data_lists3,
                            positions=np.array(
                            np.arange(len(data_lists3))) * 6.0 - 0.39,
                            patch_artist=True, showmeans=True, meanline=True,
                            widths=0.650)
                            #, whiskerprops=whiskerprops,
                            # capprops=capprops)

    plot4 = plt.boxplot(data_lists4,
                        positions=np.array(
                            np.arange(len(data_lists4))) * 6.0 +0.36,
                        patch_artist=True, showmeans=True, meanline=True,
                        widths=0.650)

    plot5 = plt.boxplot(data_lists5,
                        positions=np.array(
                            np.arange(len(data_lists5))) * 6.0 + 1.1,
                        patch_artist=True, showmeans=True, meanline=True,
                        widths=0.650)

    plot6 = plt.boxplot(data_lists6,
                        positions=np.array(
                            np.arange(len(data_lists6))) * 6.0 + 1.8,
                        patch_artist=True, showmeans=True, meanline=True,
                        widths=0.650)


    ticks = []
    for perc in percentages1:
        ticks.append(str(perc))

    # setting colors for each groups
    define_box_properties(base_plot, colours[0], names[0])
    define_box_properties(plot1, colours[1], names[1])
    define_box_properties(plot2, colours[2], names[2])
    define_box_properties(plot3, colours[3], names[3])
    define_box_properties(plot4, colours[4], names[4])
    define_box_properties(plot5, colours[5], names[5])
    define_box_properties(plot6, colours[6], names[6])

    # set the x label values
    plt.xticks(np.arange(0, len(ticks) * 6, 6), ticks, fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    # set the limit for x axis
    plt.xlim(-5.5, len(ticks) * 6)

    plt.xlabel(' % Labelled Data', fontsize=label_size, labelpad=20)
    plt.ylabel('EF Error', fontsize=label_size)

    # Make the tick lines bigger for all ticks
    ax.xaxis.set_tick_params(width=5, size=20)
    ax.yaxis.set_tick_params(width=5, size=20)

    # make ths spines thicker:
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    # PLOT LEGEND
    # pos = ax.get_position()
    # ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.92])
    # legend = plt.legend(fontsize=legend_font_size, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    ax.get_legend().remove()

    plt.savefig(os.path.join(output_path, 'boxplot_method_compare.png'), dpi=300, bbox_inches='tight')
    plt.clf()

def get_raw_masks_for_method(file1):
    output_dict = {}
    raw_output_dict = load_json(file1)
    frames_list = raw_output_dict['frames_list']
    raw_outputs = raw_output_dict['raw_outputs']

    for i in range(len(frames_list)):
        name = os.path.basename(frames_list[i])
        model_output = raw_outputs[i]

        mask = np.array(model_output)
        mask = mask[:, :, 0]
        mask = tf.greater(mask, 0.5)
        mask = tf.dtypes.cast(mask, tf.uint8)
        mask = mask.numpy()

        output_dict[name] = mask

    return output_dict

def get_dice_scores_for_method(file_keys, file1, file2):
    dict1 = load_json(file1)
    dict2 = load_json(file2)

    dice_dict = {}
    for file in file_keys:
        value1 = dict1[f'Data/Expert Consensus dataset/Images/{file}']
        dice1 = value1['dice_endo']
        value2 = dict2[f'Data/Expert Consensus dataset/Images/{file}']
        dice2 = value2['dice_endo']
        dice_dict[file] = (dice1+dice2)/2.0

    return dice_dict

def get_ef_scores_for_method(file_keys, file1, file2):
    dict1 = load_json(file1)
    dict2 = load_json(file2)

    # 'Ejection Fraction gt': eject_frac_gt,
    # 'Ejection Fraction pred': eject_frac_pred,
    # 'EF error': ef_error,
    # 'EDV gt': gt_ED_vol,
    # 'ESV gt': gt_ES_vol,
    # 'EDV pred': pred_ED_vol,
    # 'ESV pred': pred_ES_vol,
    # 'EDV error': ED_vol_error,
    # 'ESV error': ES_vol_error,

    ef_gt_dict = {}
    ef_pred_dict = {}
    ef_err_dict = {}
    edv_gt_dict = {}
    esv_gt_dict = {}
    edv_pred_dict = {}
    esv_pred_dict = {}
    edv_err_dict = {}
    esv_err_dict = {}

    for file in file_keys:
        values1 = dict1[f'{file}']
        values2 = dict2[f'{file}']

        ef_gt_dict[file] = values1['Ejection Fraction gt']
        edv_gt_dict[file] = values1['EDV gt']
        esv_gt_dict[file] = values1['ESV gt']

        val1 = values1['Ejection Fraction pred']
        val2 = values2['Ejection Fraction pred']
        ef_pred_dict[file] = (val1+val2)/2.0

        val1 = values1['EF error']
        val2 = values2['EF error']
        ef_err_dict[file] = (val1 + val2) / 2.0

        val1 = values1['EDV pred']
        val2 = values2['EDV pred']
        edv_pred_dict[file] = (val1 + val2) / 2.0

        val1 = values1['ESV pred']
        val2 = values2['ESV pred']
        esv_pred_dict[file] = (val1 + val2) / 2.0

        val1 = values1['EDV error']
        val2 = values2['EDV error']
        edv_err_dict[file] = (val1 + val2) / 2.0

        val1 = values1['ESV error']
        val2 = values2['ESV error']
        esv_err_dict[file] = (val1 + val2) / 2.0

    return ef_gt_dict, ef_pred_dict, ef_err_dict, edv_gt_dict, esv_gt_dict, edv_pred_dict, esv_pred_dict, edv_err_dict, esv_err_dict

def get_coordinates(csv_file):
    file_and_coords = {}

    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            # line_count = 0 is column names
            # row is of type list
            # row[2] is the file name .png
            # row[3] is the user id
            # row[5] is label and we're interested in 'curve-lv-endo'
            # row[7] is x
            # row[8] is y

            line_count += 1
            if (line_count == 1):
                continue

            file_name = row[2]
            userid = row[3]
            label = row[5]

            if (label.lower() == 'curve-lv-endo'.lower()):

                coord_x_list = row[7]
                coord_x_list = coord_x_list.split()

                coord_y_list = row[8]
                coord_y_list = coord_y_list.split()

                coords_x = [float(x) for x in coord_x_list]
                coords_y = [float(y) for y in coord_y_list]

                if (len(coords_x) > 0 and len(coords_y) > 0):
                    if file_name in file_and_coords:
                        users_dict = file_and_coords[file_name]
                        users_dict[userid] = (coords_x, coords_y)
                    else:
                        users_dict = {}
                        users_dict[userid] = (coords_x, coords_y)
                        file_and_coords[file_name] = users_dict

    return file_and_coords

def reorder_points(coords):

        closest_dist = 999999999
        closest_index = -1
        closest_pt = None

        for j in range(len(coords[0])):        #loop through points
            pt = (coords[0][j], coords[1][j])  # x,y

            dist = math.dist(pt, (0,511))

            if dist < closest_dist:
                closest_dist = dist
                closest_index = j
                closest_pt = pt

        if closest_index > 150: #reverse
            new_order_x = [0 for z in range(len(coords[0]))]
            new_order_y = [0 for z in range(len(coords[0]))]
            for j in range(len(coords[0])):  # loop through points
                x = coords[0][j]
                y = coords[1][j]

                idx = len(coords[0]) -1
                new_order_x[idx-j] = x
                new_order_y[idx-j] = y

            return (new_order_x, new_order_y)
        else:
            return coords

def get_avg_coordinates(file_and_coords):
    files_avg_coords = {}

    # key is the fileid and value is a dictionary keyed by userid its value is that users annotation
    # for the specific file
    for key, value in file_and_coords.items():
        file_name = key
        num_tuples = len(value)  # example 10
        annotations = list(value.values())
        num_coords = len(annotations[0][0])  # example 200 x_coords

        avg_x_coords = [0 for i in range(num_coords)]
        avg_y_coords = [0 for i in range(num_coords)]

        for userid, x_y_tuples in value.items():
            coords_x = x_y_tuples[0]
            coords_y = x_y_tuples[1]
            # print(len(coords_x)) #example 200
            # print(len(coords_y)) #example 200

            for i in range(num_coords):
                avg_x_coords[i] = avg_x_coords[i] + coords_x[i]
                avg_y_coords[i] = avg_y_coords[i] + coords_y[i]

        for i in range(num_coords):
            avg_x_coords[i] = avg_x_coords[i] / num_tuples
            avg_y_coords[i] = avg_y_coords[i] / num_tuples

        files_avg_coords[file_name] = (avg_x_coords, avg_y_coords)

    return files_avg_coords

def get_avg_coordinates_exclude(file_and_coords, userid_to_exclude):
    files_avg_coords = {}

    # key is the fileid and value is a dictionary keyed by userid its value is that users annotation
    # for the specific file
    for key, value in file_and_coords.items():
        file_name = key
        num_tuples = len(value) - 1  # example 10, we subtract 1 for the excluded user
        annotations = list(value.values())
        num_coords = len(annotations[0][0])  # example 200 x_coords

        avg_x_coords = [0 for i in range(num_coords)]
        avg_y_coords = [0 for i in range(num_coords)]

        for userid, x_y_tuples in value.items():
            if userid == userid_to_exclude:
                continue

            coords_x = x_y_tuples[0]
            coords_y = x_y_tuples[1]
            # print(len(coords_x)) #example 200
            # print(len(coords_y)) #example 200

            for i in range(num_coords):
                avg_x_coords[i] = avg_x_coords[i] + coords_x[i]
                avg_y_coords[i] = avg_y_coords[i] + coords_y[i]

        for i in range(num_coords):
            avg_x_coords[i] = avg_x_coords[i] / num_tuples
            avg_y_coords[i] = avg_y_coords[i] / num_tuples

        files_avg_coords[file_name] = (avg_x_coords, avg_y_coords)

    return files_avg_coords

def get_original_files(files_avg_coords, output_folder):
    realistic_timeout_in_seconds = 5
    MAGIQUANT_ADDRESS = "files.magiquant.com"  # "89.39.141.131"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'}

    for key, value in files_avg_coords.items():
        hash = key
        sub_a = hash[:2]
        sub_b = hash[3:5]
        sub_c = hash[5:7]
        file_name_with_path = hash
        location = f"https://{MAGIQUANT_ADDRESS}/scantensus-database-png-flat/{sub_a}/{sub_b}/{sub_c}/{file_name_with_path}"
        request = urllib.request.Request(location, headers=headers)
        output_path = os.path.join(output_folder, file_name_with_path)
        if os.path.exists(output_path):
            continue
        f = open(output_path, 'wb')
        with urllib.request.urlopen(request, timeout=realistic_timeout_in_seconds) as url:
            s = url.read()
            f.write(s)
        f.close()


def compute_rankings():
    save_folder = 'results_expert_vs_ai_ssl'
    if not os.path.exists(save_folder): os.mkdir(save_folder)

    csv_file = r'Data/Expert Consensus dataset/user-labels-all.csv'

    file_and_coords = get_coordinates(csv_file)

    print(len(file_and_coords))

    # reorder points
    file_and_coords_reordered = {}
    for fileid, value in file_and_coords.items():
        users_dict = {}
        for userid, coords in value.items():
            users_dict[userid] = reorder_points(coords)
        file_and_coords_reordered[fileid] = users_dict

    files_avg_coords = get_avg_coordinates(file_and_coords_reordered)

    output_folder = os.path.join(save_folder, 'temp_images_consensus_orig')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    get_original_files(files_avg_coords, output_folder)

    rankings_name, rankings_val = [], []
    for key, value in file_and_coords_reordered.items():  # loop through dataset
        file_name = key
        original_image = cv2.imread(os.path.join(output_folder, file_name), cv2.IMREAD_GRAYSCALE)
        num_coord_sets = len(value)  # example 10 annotations

        points_per_set = []
        masks = []

        for userid, coords in value.items():  # loop through each expert annotation
            coords_list = []
            for j in range(len(coords[0])):  # loop through 200 points
                x_coord = coords[0][j]
                y_coord = coords[1][j]
                coords_list.append((x_coord, y_coord))
            pts = np.array(coords_list, np.int32)
            points_per_set.append(coords_list)

            # create mask
            blank_image = np.zeros((original_image.shape[0], original_image.shape[1]))
            mask = draw_poly_on_image(blank_image, pts, (1, 1, 1), False)
            mask = fill_poly_on_image(mask, pts, (1, 1, 1))
            for i in range(original_image.shape[0]):
                for j in range(original_image.shape[1]):
                    mask[i, j] = np.uint8(mask[i, j])
            masks.append(mask)

        dices = []
        for m in range(len(masks)):
            for n in range(m + 1, len(masks)):
                mask1 = masks[m]
                mask2 = masks[n]
                dice = measurements.compute_Dice_coefficient(mask1, mask2)
                dices.append(dice)

        mean_dice = np.mean(dices)
        std_dice = np.std(dices)
        CV = (std_dice / mean_dice) * 100

        rankings_name.append(file_name)
        rankings_val.append(CV)

    rankings_dict = {}
    rankings_val, rankings_name = zip(*sorted(zip(rankings_val, rankings_name)))
    for i in range(len(rankings_val)):
        name = rankings_name[i]
        ranking = rankings_val[i]
        rankings_dict[name] = ranking

    write_dict_to_json(rankings_dict, save_folder, 'rankings.json')

    return rankings_dict

# def compute_rank():
#     A = ExtractMasks()
#     n = len(A)
#     S = []
#     k = (n*(n-1))/2
#     for i in range(n):
#         for j in range(i + 1, n):
#             mask1 = A[i]
#             mask2 = A[j]
#             dice = measurements.compute_Dice_coefficient(mask1, mask2)
#             S.append(dice)
#
#     assert len(S) == k
#     mean_dice = np.mean(S)
#     std_dice = np.std(S)
#     rank = (std_dice / mean_dice) * 100
#
#     return rank

def get_EF_dataframes(super_vol_results_dict):

    file_column_list1 = []
    new_ef_err_dict = {}
    new_esv_err_dict = {}
    new_edv_err_dict = {}
    new_ef_dict = {}
    new_edv_dict = {}
    new_esv_dict = {}
    cnt=0

    #run once
    for userid, info_dict in super_vol_results_dict.items():
        file_column_list1 = list(info_dict.keys())
        new_ef_err_dict['File'] = list(info_dict.keys())
        new_esv_err_dict['File'] = list(info_dict.keys())
        new_edv_err_dict['File'] = list(info_dict.keys())
        new_ef_dict['File'] = list(info_dict.keys())
        new_edv_dict['File'] = list(info_dict.keys())
        new_esv_dict['File'] = list(info_dict.keys())
        break

    for userid in list(super_vol_results_dict.keys()):
        new_ef_err_dict[userid] = []
        new_esv_err_dict[userid] = []
        new_edv_err_dict[userid] = []
        new_ef_dict[userid] = []
        new_edv_dict[userid] = []
        new_esv_dict[userid] = []

    for userid, info_dict in super_vol_results_dict.items():
        ef_err_list = new_ef_err_dict[userid]
        es_err_list = new_esv_err_dict[userid]
        ed_err_list = new_edv_err_dict[userid]
        ef_list = new_ef_dict[userid]
        esv_list = new_edv_dict[userid]
        edv_list = new_esv_dict[userid]
        for filename, info in info_dict.items():
            ef_error = info['EF error']
            ef_err_list.append(ef_error)
            esv_error = info['ESV error']
            es_err_list.append(esv_error)
            edv_error = info['EDV error']
            ed_err_list.append(edv_error)

            ef = info['Ejection Fraction pred']
            ef_list.append(ef)
            esv = info['ESV pred']
            esv_list.append(esv)
            edv = info['EDV pred']
            edv_list.append(edv)

    ef_gt_dict, ef_pred_dict, ef_err_dict, edv_gt_dict, esv_gt_dict, edv_pred_dict, esv_pred_dict, edv_err_dict, esv_err_dict = get_ef_scores_for_method(
        file_column_list1,
        'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/finetuning_4/Consensus_EF_Vol_results_all_0.json',
        'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/finetuning_4/Consensus_EF_Vol_results_all_1.json')
    new_ef_err_dict['SimCLR4'] = list(ef_err_dict.values())
    new_esv_err_dict['SimCLR4'] = list(esv_err_dict.values())
    new_edv_err_dict['SimCLR4'] = list(edv_err_dict.values())
    new_ef_dict['SimCLR4'] = list(ef_pred_dict.values())
    new_esv_dict['SimCLR4'] = list(esv_pred_dict.values())
    new_edv_dict['SimCLR4'] = list(edv_pred_dict.values())

    ef_gt_dict, ef_pred_dict, ef_err_dict, edv_gt_dict, esv_gt_dict, edv_pred_dict, esv_pred_dict, edv_err_dict, esv_err_dict = get_ef_scores_for_method(
        file_column_list1,
        'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/finetuning_15/Consensus_EF_Vol_results_all_0.json',
        'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/finetuning_15/Consensus_EF_Vol_results_all_1.json')
    new_ef_err_dict['SimCLR15'] = list(ef_err_dict.values())
    new_esv_err_dict['SimCLR15'] = list(esv_err_dict.values())
    new_edv_err_dict['SimCLR15'] = list(edv_err_dict.values())
    new_ef_dict['SimCLR15'] = list(ef_pred_dict.values())
    new_esv_dict['SimCLR15'] = list(esv_pred_dict.values())
    new_edv_dict['SimCLR15'] = list(edv_pred_dict.values())

    ef_gt_dict, ef_pred_dict, ef_err_dict, edv_gt_dict, esv_gt_dict, edv_pred_dict, esv_pred_dict, edv_err_dict, esv_err_dict = get_ef_scores_for_method(
        file_column_list1,
        'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/finetuning_100/Consensus_EF_Vol_results_all_0.json',
        'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/finetuning_100/Consensus_EF_Vol_results_all_1.json')
    new_ef_err_dict['SimCLR100'] = list(ef_err_dict.values())
    new_esv_err_dict['SimCLR100'] = list(esv_err_dict.values())
    new_edv_err_dict['SimCLR100'] = list(edv_err_dict.values())
    new_ef_dict['SimCLR100'] = list(ef_pred_dict.values())
    new_esv_dict['SimCLR100'] = list(esv_pred_dict.values())
    new_edv_dict['SimCLR100'] = list(edv_pred_dict.values())

    new_esv_dict['GT'] = list(esv_gt_dict.values())
    new_ef_dict['GT'] = list(ef_gt_dict.values())
    new_edv_dict['GT'] = list(edv_gt_dict.values())

    df_ef_err = pd.DataFrame(new_ef_err_dict)
    df_esv_err = pd.DataFrame(new_esv_err_dict)
    df_edv_err = pd.DataFrame(new_edv_err_dict)
    df_ef = pd.DataFrame(new_ef_dict)
    df_esv = pd.DataFrame(new_esv_dict)
    df_edv = pd.DataFrame(new_edv_dict)

    return df_ef, df_edv, df_esv, df_ef_err, df_esv_err, df_edv_err

def get_EF_dataframes_without_sslmethods(super_vol_results_dict):

    file_column_list1 = []
    new_ef_err_dict = {}
    new_esv_err_dict = {}
    new_edv_err_dict = {}
    new_ef_dict = {}
    new_edv_dict = {}
    new_esv_dict = {}
    cnt=0

    #run once
    for userid, info_dict in super_vol_results_dict.items():
        file_column_list1 = list(info_dict.keys())
        new_ef_err_dict['File'] = list(info_dict.keys())
        new_esv_err_dict['File'] = list(info_dict.keys())
        new_edv_err_dict['File'] = list(info_dict.keys())
        new_ef_dict['File'] = list(info_dict.keys())
        new_edv_dict['File'] = list(info_dict.keys())
        new_esv_dict['File'] = list(info_dict.keys())
        break

    for userid in list(super_vol_results_dict.keys()):
        new_ef_err_dict[userid] = []
        new_esv_err_dict[userid] = []
        new_edv_err_dict[userid] = []
        new_ef_dict[userid] = []
        new_edv_dict[userid] = []
        new_esv_dict[userid] = []

    ef_gt_dict = {}
    esv_gt_dict = {}
    edv_gt_dict = {}

    for userid, info_dict in super_vol_results_dict.items():
        ef_err_list = new_ef_err_dict[userid]
        es_err_list = new_esv_err_dict[userid]
        ed_err_list = new_edv_err_dict[userid]
        ef_list = new_ef_dict[userid]
        esv_list = new_edv_dict[userid]
        edv_list = new_esv_dict[userid]
        for filename, info in info_dict.items():
            ef_error = info['EF error']
            ef_err_list.append(ef_error)
            esv_error = info['ESV error']
            es_err_list.append(esv_error)
            edv_error = info['EDV error']
            ed_err_list.append(edv_error)

            ef = info['Ejection Fraction pred']
            ef_list.append(ef)
            esv = info['ESV pred']
            esv_list.append(esv)
            edv = info['EDV pred']
            edv_list.append(edv)

            ef_gt_dict[filename] = info['Ejection Fraction gt']
            esv_gt_dict[filename] = info['ESV gt']
            edv_gt_dict[filename] = info['EDV gt']

    new_esv_dict['GT'] = list(esv_gt_dict.values())
    new_ef_dict['GT'] = list(ef_gt_dict.values())
    new_edv_dict['GT'] = list(edv_gt_dict.values())

    df_ef_err = pd.DataFrame(new_ef_err_dict)
    df_esv_err = pd.DataFrame(new_esv_err_dict)
    df_edv_err = pd.DataFrame(new_edv_err_dict)
    df_ef = pd.DataFrame(new_ef_dict)
    df_esv = pd.DataFrame(new_esv_dict)
    df_edv = pd.DataFrame(new_edv_dict)

    return df_ef, df_edv, df_esv, df_ef_err, df_esv_err, df_edv_err



def plot_dice_boxplots_ai_vs_expert(save_path_name, df, userids_dict, userids_int_mappings_dict, userids_colour_dict):
    plt.clf()

    column_means = df.mean(skipna=False)
    sorted_means = column_means.sort_values()
    columns_sorted_dict = {}
    for column, mean_value in sorted_means.items():
        columns_sorted_dict[column] = mean_value

    columns_to_plot = []
    column_labels = []
    colours_list = []

    for column in list(columns_sorted_dict.keys()):
        columns_to_plot.append(column)
        if column in userids_dict:
            column_labels.append(f'Exp {userids_int_mappings_dict[column]}')
            #colours_list.append(userids_colour_dict[column])
            colours_list.append('gray')
        elif 'Sim' in column:
            column_labels.append(f'AI({column[6:]}%)')
            colours_list.append('orange')
        # else:
        #     column_labels.append(f'Rand ({column[6:]}%)')
        #     colours_list.append('purple')

    # Create the boxplot
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)  # Adjust figure size if needed

    # Boxplot
    box = ax.boxplot(
        [df[col].dropna() for col in columns_to_plot],  # Drop NaN values for each column
        patch_artist=True,  # Allow filling the boxes with color
        showmeans=True,  # Show the mean
        meanline=True,  # Use a solid line for the mean
        widths=0.6  # Adjust the box width
    )

    # Colors for each column
    # colors = ['lightblue', 'lightgreen', 'lightcoral']

    # Apply colors to the boxes
    for patch, color in zip(box['boxes'], colours_list):
        patch.set_facecolor(color)

    for median in box['medians']:
        median.set(linewidth=2.5, color='black')

    for mean in box['means']:
        mean.set(linewidth=2.5, color='white', linestyle='dashed')#(0, (2.5, 3)))

    for flier in box['fliers']:
        flier.set(markersize=8)

    # Set x-tick labels to column names
    ax.set_xticks(range(1, len(columns_to_plot) + 1))
    ax.set_xticklabels(column_labels)

    # Add a vertical dashed line to separate first 11 and last 3 boxplots
    #ax.axvline(x=11.5, color='gray', linestyle='dashed', linewidth=1.5)

    # Set y-axis label
    ax.set_ylabel('Dice Score', fontsize=18)
    #ax.tick_params(axis='x', labelrotation=45)
    #ax.set_xlabel('Experts and AI models', fontsize=12)
    # Add a title
    #ax.set_title('Consensus Dataset')#('"Comparison Between Experts and Model"')

    #Set range
    selected_columns = df.iloc[:, 1:]
    # Calculate the mean across all rows and selected columns
    min_value = selected_columns.min().min()
    max_value = selected_columns.max().max()
    ax.set_ylim(float(min_value) - 0.01, max_value + 0.01) #0.97 + 0.01
    yticks = np.arange(float(min_value), float(max_value), 0.06)  # Example: ticks from 0 to 8 with a step of 0.5
    ax.set_yticks(np.round(yticks, 2))
    # yticks = range(float(min_value), 0.95, 0.03)  # Example: Add ticks from 0 to 8 with a step of 1
    # ax.set_yticks(yticks)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    # Show the plot
    plt.savefig(save_path_name, dpi=300)

    return sorted_means

def plot_dice_boxplots_ai_vs_one_expert(save_path_name, df, userids_dict, userids_int_mappings_dict,
                                    userids_colour_dict, specific_userid):
    plt.clf()

    column_means = df.mean(skipna=False)
    sorted_means = column_means.sort_values()
    columns_sorted_dict = {}
    for column, mean_value in sorted_means.items():
        columns_sorted_dict[column] = mean_value

    columns_to_plot = []
    column_labels = []
    colours_list = []

    for column in list(columns_sorted_dict.keys()):

        if column == specific_userid:
            columns_to_plot.append(column)
            column_labels.append(f'Exp {userids_int_mappings_dict[column]}')
            # colours_list.append(userids_colour_dict[column])
            colours_list.append('gray')
        elif 'Sim' in column:
            columns_to_plot.append(column)
            column_labels.append(f'AI ({column[6:]}%)')
            colours_list.append('orange')
        # else:
        #     column_labels.append(f'Rand ({column[6:]}%)')
        #     colours_list.append('purple')

    # Create the boxplot
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)  # Adjust figure size if needed

    # Boxplot
    box = ax.boxplot(
        [df[col].dropna() for col in columns_to_plot],  # Drop NaN values for each column
        patch_artist=True,  # Allow filling the boxes with color
        showmeans=True,  # Show the mean
        meanline=True,  # Use a solid line for the mean
        widths=0.6  # Adjust the box width
    )

    # Colors for each column
    # colors = ['lightblue', 'lightgreen', 'lightcoral']

    # Apply colors to the boxes
    for patch, color in zip(box['boxes'], colours_list):
        patch.set_facecolor(color)

    for median in box['medians']:
        median.set(linewidth=2.5, color='black')

    for mean in box['means']:
        mean.set(linewidth=2.5, color='white', linestyle='dashed')  # (0, (2.5, 3)))

    for flier in box['fliers']:
        flier.set(markersize=8)

    # Set x-tick labels to column names
    ax.set_xticks(range(1, len(columns_to_plot) + 1))
    ax.set_xticklabels(column_labels)

    # Add a vertical dashed line to separate first 11 and last 3 boxplots
    # ax.axvline(x=11.5, color='gray', linestyle='dashed', linewidth=1.5)

    # Set y-axis label
    ax.set_ylabel('Dice Score', fontsize=22)
    # Add a title
    # ax.set_title('"Comparison Between Experts and Model"')

    # Set range
    selected_columns = df.iloc[:, 1:]
    # Calculate the mean across all rows and selected columns
    min_value = selected_columns.min().min()
    max_value = selected_columns.max().max()
    ax.set_ylim(float(min_value) - 0.01, max_value + 0.01)  # 0.97 + 0.01
    yticks = np.arange(float(min_value), float(max_value), 0.06)  # Example: ticks from 0 to 8 with a step of 0.5
    ax.set_yticks(np.round(yticks, 2))
    # yticks = range(float(min_value), 0.97, 0.03)  # Example: Add ticks from 0 to 8 with a step of 1
    # ax.set_yticks(yticks)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=18)

    plt.tight_layout()
    # Show the plot
    plt.savefig(save_path_name, dpi=300)

    return sorted_means

def plot_ef_err_boxplots_expert_vs_ai(df, userids_dict, userids_int_mappings_dict, userids_colour_dict, save_path_name, y_label):
    plt.clf()

    column_means = df.mean(skipna=False)
    sorted_means = column_means.sort_values(ascending=False)
    columns_sorted_dict = {}
    for column, mean_value in sorted_means.items():
        columns_sorted_dict[column] = mean_value

    columns_to_plot = []
    column_labels = []
    colours_list = []

    for column in list(columns_sorted_dict.keys()):
        columns_to_plot.append(column)
        if column in userids_dict:
            column_labels.append(f'Exp {userids_int_mappings_dict[column]}')
            #colours_list.append(userids_colour_dict[column])
            colours_list.append('gray')
        else:
            column_labels.append(f'AI ({column[6:]}%)')
            colours_list.append('orange')

    selected_columns = columns_to_plot
    # Create the boxplot
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)  # Adjust figure size if needed

    # Boxplot
    box = ax.boxplot(
        [df[col].dropna() for col in selected_columns],  # Drop NaN values for each column
        patch_artist=True,  # Allow filling the boxes with color
        showmeans=True,  # Show the mean
        meanline=True,  # Use a solid line for the mean
        widths=0.6  # Adjust the box width
    )

    # Colors for each column
    # colors = ['lightblue', 'lightgreen', 'lightcoral']

    # Apply colors to the boxes
    for patch, color in zip(box['boxes'], colours_list):
        patch.set_facecolor(color)

    for median in box['medians']:
        median.set(linewidth=2.5, color='black')

    for mean in box['means']:
        mean.set(linewidth=2.5, color='white', linestyle='dashed')

    for flier in box['fliers']:
        flier.set(markersize=8)

    # Set x-tick labels to column names
    ax.set_xticks(range(1, len(selected_columns) + 1))
    ax.set_xticklabels(column_labels)

    # Set y-axis label
    ax.set_ylabel(y_label, fontsize=18)
    # Add a title
    # ax.set_title('"Comparison Between Experts and Model"')

    # Set range
    selected_columns = df.iloc[:, 1:]
    # Calculate the mean across all rows and selected columns
    min_value = selected_columns.min().min()
    max_value = selected_columns.max().max()
    ax.set_ylim(float(min_value) - 0.01, max_value + 0.01)  # 0.97 + 0.01
    #yticks = np.arange(float(min_value), float(max_value), 0.06)  # Example: ticks from 0 to 8 with a step of 0.5
    #ax.set_yticks(np.round(yticks, 2))
    # yticks = range(float(min_value), 0.97, 0.03)  # Example: Add ticks from 0 to 8 with a step of 1
    # ax.set_yticks(yticks)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    # Show the plot
    plt.savefig(save_path_name, dpi=300)

    return sorted_means

def plot_ef_err_boxplots_ai_vs_one_expert(df, userids_dict, userids_int_mappings_dict, userids_colour_dict,
                                          save_path_name, y_label, specific_userid):
    plt.clf()

    column_means = df.mean(skipna=False)
    sorted_means = column_means.sort_values(ascending=False)
    columns_sorted_dict = {}
    for column, mean_value in sorted_means.items():
        columns_sorted_dict[column] = mean_value

    columns_to_plot = []
    column_labels = []
    colours_list = []

    for column in list(columns_sorted_dict.keys()):
        if column == specific_userid:
            columns_to_plot.append(column)
            column_labels.append(f'Exp {userids_int_mappings_dict[column]}')
            # colours_list.append(userids_colour_dict[column])
            colours_list.append('gray')
        elif 'Sim' in column:
            columns_to_plot.append(column)
            column_labels.append(f'AI ({column[6:]}%)')
            colours_list.append('orange')

    selected_columns = columns_to_plot
    # Create the boxplot
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)  # Adjust figure size if needed

    # Boxplot
    box = ax.boxplot(
        [df[col].dropna() for col in selected_columns],  # Drop NaN values for each column
        patch_artist=True,  # Allow filling the boxes with color
        showmeans=True,  # Show the mean
        meanline=True,  # Use a solid line for the mean
        widths=0.6  # Adjust the box width
    )

    # Colors for each column
    # colors = ['lightblue', 'lightgreen', 'lightcoral']

    # Apply colors to the boxes
    for patch, color in zip(box['boxes'], colours_list):
        patch.set_facecolor(color)

    for median in box['medians']:
        median.set(linewidth=2.5, color='black')

    for mean in box['means']:
        mean.set(linewidth=2.5, color='white', linestyle='dashed')

    for flier in box['fliers']:
        flier.set(markersize=8)

    # Set x-tick labels to column names
    ax.set_xticks(range(1, len(selected_columns) + 1))
    ax.set_xticklabels(column_labels)

    # Set y-axis label
    ax.set_ylabel(y_label, fontsize=22)
    # Add a title
    # ax.set_title('"Comparison Between Experts and Model"')

    # Set range
    selected_columns = df.iloc[:, 1:]
    # Calculate the mean across all rows and selected columns
    min_value = selected_columns.min().min()
    max_value = selected_columns.max().max()
    ax.set_ylim(float(min_value) - 0.01, max_value + 0.01)  # 0.97 + 0.01
    #yticks = np.arange(float(min_value), float(max_value), 0.06)  # Example: ticks from 0 to 8 with a step of 0.5
    #ax.set_yticks(np.round(yticks, 2))
    # yticks = range(float(min_value), 0.97, 0.03)  # Example: Add ticks from 0 to 8 with a step of 1
    # ax.set_yticks(yticks)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=18)

    plt.tight_layout()
    # Show the plot
    plt.savefig(save_path_name, dpi=300)

    return sorted_means


def plot_ef_bland_altman_expert_vs_ai(df, userids_dict, userids_renamed_dict, userids_colour_dict, save_path_name, y_label):
    plt.clf()

    columns_to_plot = []
    column_labels = []
    colours_list = []

    # Define ground truth and predicted columns
    ground_truth = df["GT"]
    predicted_columns = df.columns[1:-1]  # Exclude the first column and ground truth column

    predicted_columns = list(userids_renamed_dict.keys())

    column_names = list(df.columns)
    predicted_columns.extend(column_names[-4:-1])

    for column in predicted_columns:
        columns_to_plot.append(column)
        if column in userids_dict:
            column_labels.append(f'Exp {userids_renamed_dict[column]}')
            colours_list.append(userids_colour_dict[column])
        else:
            column_labels.append(f'AI ({column[6:]}%)')
            colours_list.append('orange')


    # Create subplots for Bland-Altman plots
    num_plots = len(predicted_columns)
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16), dpi=300)  # Adjust grid size for 14 plots
    axes = axes.flatten()

    for i, col in enumerate(predicted_columns):
        # Predicted values
        predicted = df[col]

        # Calculate mean and difference
        mean = (predicted + ground_truth) / 2
        diff = predicted - ground_truth

        # Calculate mean difference and standard deviation of differences
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)

        # Bland-Altman limits
        upper_limit = mean_diff + 1.96 * std_diff
        lower_limit = mean_diff - 1.96 * std_diff

        # Plot Bland-Altman plot
        ax = axes[i]
        ax.scatter(mean, diff, color="blue", alpha=0.6, label="Data points")
        ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="Mean difference")
        ax.axhline(y=diff.mean() + 1.96 * diff.std(), color="green", linestyle="--", linewidth=1, label="Upper limit")
        ax.axhline(y=diff.mean() - 1.96 * diff.std(), color="green", linestyle="--", linewidth=1, label="Lower limit")

        # Add text with +1.96 and -1.96 std values
        max_mean = mean.max()  # Get the max x-axis value for positioning
        ax.text(max_mean/1.4, upper_limit+0.5, f'+1.96 SD: {upper_limit:.2f}', color='black', fontsize=8)
        ax.text(max_mean/1.4, lower_limit+0.5, f'-1.96 SD: {lower_limit:.2f}', color='black', fontsize=8)

        # Add labels and title
        ax.set_title(f"{column_labels[i]}", fontsize=10)
        ax.set_xlabel("Mean of Ground Truth & Prediction", fontsize=8)
        ax.set_ylabel("Difference", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
        #ax.legend(fontsize=6)

    # Hide unused subplots
    for j in range(len(predicted_columns), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    # Show the plot
    plt.savefig(save_path_name, dpi=300)

def fill_df_mean_per_column(df):
    #df_ef = df_ef.apply(lambda col: col.where(col != 0, col.mean()), axis=0)
    #The above line doesnt work since the first column is a string(filename), do this instead:

    # Exclude the first column (Column1) from the operation
    columns_to_modify = df.columns[1:]

    # Replace 0 with the mean of the column for each column in columns_to_modify
    for col in columns_to_modify:
        mean_value = df[col][df[col] != 0].mean()  # Calculate the mean excluding zeros
        df[col] = df[col].apply(lambda x: mean_value if x == 0 else x)

    return df

def print_df_zero_info(df):
    zero_counts = (df == 0).sum()
    # Display the result
    print("Number of zeros in each column:")
    print(zero_counts)
    print(f'row count: {len(df)}')
    print()

def get_user_colours_dict():
    userids_colour_dict = {'qju05JPYWQf0thHcNjSc6VVYzKb2': 'lightcoral', '9P03GWE1P9XFLEVHa1ckJ3ah1RZ2': 'lightblue',
                           'onzoCZlCxaTS2HNeP75OSgozs2y2': 'green', 'PosPfN1VDhgHfbRQxEKuXFTvtxj1': 'violet',
                           'GblV0kIx4sQF1I94qBnm716Ij3Y2': 'darkkhaki', 'CSZONN4uJ4gMh0RAY2llpjIdZVg2': 'turquoise',
                           'teAZL0rNDyWevenTcbHQTdzCOGi2': 'gold', 'PwBpS83NxccFFfHxNLSIkKYPy3o2': 'crimson',
                           '1IDjWO8M8WYzicp6bPpZ1ydQpbQ2': 'limegreen', 'PU9duvAh4lhxxIoMHarmO6RZIDg1' : 'slateblue',
                           '0Sxv0Eka8sXPzolcbnt1ck082fT2': 'cornflowerblue'}
    return userids_colour_dict

def get_user_int_mappings_dict():

    return {'qju05JPYWQf0thHcNjSc6VVYzKb2': 1, '9P03GWE1P9XFLEVHa1ckJ3ah1RZ2': 2,
           'onzoCZlCxaTS2HNeP75OSgozs2y2': 3, 'PosPfN1VDhgHfbRQxEKuXFTvtxj1': 4,
           'GblV0kIx4sQF1I94qBnm716Ij3Y2': 5, 'CSZONN4uJ4gMh0RAY2llpjIdZVg2': 6,
           'teAZL0rNDyWevenTcbHQTdzCOGi2': 7, 'PwBpS83NxccFFfHxNLSIkKYPy3o2': 8,
           '1IDjWO8M8WYzicp6bPpZ1ydQpbQ2': 9, 'PU9duvAh4lhxxIoMHarmO6RZIDg1' : 10,
           '0Sxv0Eka8sXPzolcbnt1ck082fT2': 11}

def expert_agreement_vs_ai(rankings_file):
    rankings_dict = load_json(rankings_file)

    save_folder = 'results_expert_vs_ai_ssl'
    if not os.path.exists(save_folder): os.mkdir(save_folder)

    csv_file = r'Data/Expert Consensus dataset/user-labels-all.csv'

    file_and_coords = get_coordinates(csv_file)

    print(len(file_and_coords))

    # Hardcode these so they're always the same
    userids_colour_dict = get_user_colours_dict()
    userids_int_mappings_dict = get_user_int_mappings_dict()
    utils.write_dict_to_json(userids_int_mappings_dict, save_folder, 'userids_integer_mappings.json')

    # reorder points
    file_and_coords_reordered = {}
    userids_dict = {}
    for fileid, value in file_and_coords.items():
        users_dict = {}
        for userid, coords in value.items():
            users_dict[userid] = reorder_points(coords)
            if userid not in userids_dict:
                userids_dict[userid] = 0
        file_and_coords_reordered[fileid] = users_dict

    files_avg_coords = get_avg_coordinates(file_and_coords_reordered)

    write_dict_to_json(file_and_coords_reordered, save_folder, 'UnityLV_multiX_annotations_point_order_corrected.json')
    write_dict_to_json(files_avg_coords, save_folder, 'UnityLV_multiX_consensus_curves.json')

    output_folder = os.path.join(save_folder, 'temp_images_consensus_orig')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    get_original_files(files_avg_coords, output_folder)

    if not os.path.exists(os.path.join(save_folder, f'user_dice_scores.csv')):
        # Get masks of each user annotation
        file_user_mask_dict = {}
        consensus_curve_mask_dict = {}
        cnt = 0
        for key, value in file_and_coords_reordered.items():  # loop through dataset
            file_name = key
            original_image = cv2.imread(os.path.join(output_folder, file_name), cv2.IMREAD_GRAYSCALE)
            num_coord_sets = len(value)  # example 10 annotations

            points_per_set = []
            user_masks = {}

            for userid, coords in value.items():  # loop through each expert annotation
                coords_list = []
                for j in range(len(coords[0])):  # loop through 200 points
                    x_coord = coords[0][j]
                    y_coord = coords[1][j]
                    coords_list.append((x_coord, y_coord))
                pts = np.array(coords_list, np.int32)
                points_per_set.append(coords_list)

                # create mask
                blank_image = np.zeros((original_image.shape[0], original_image.shape[1]))
                mask = draw_poly_on_image(blank_image, pts, (1, 1, 1), False)
                mask = fill_poly_on_image(mask, pts, (1, 1, 1))
                for i in range(original_image.shape[0]):
                    for j in range(original_image.shape[1]):
                        mask[i, j] = np.uint8(mask[i, j])
                #mask = np.uint8(mask)
                user_masks[userid] = pad_resize_preserve_aspect_ratio(mask, 512)

            file_user_mask_dict[file_name] = user_masks

            # Get masks for consensus/avg curve:
            consensus_curve_coords = files_avg_coords[file_name]
            coords_list = []
            for j in range(len(consensus_curve_coords[0])):  # loop through 200 points
                x_coord = consensus_curve_coords[0][j]
                y_coord = consensus_curve_coords[1][j]
                coords_list.append((x_coord, y_coord))
            pts = np.array(coords_list, np.int32)
            blank_image = np.zeros((original_image.shape[0], original_image.shape[1]))
            mask = draw_poly_on_image(blank_image, pts, (1, 1, 1), False)
            mask = fill_poly_on_image(mask, pts, (1, 1, 1))
            for i in range(original_image.shape[0]):
                for j in range(original_image.shape[1]):
                    mask[i, j] = np.uint8(mask[i, j])
            #mask = np.uint8(mask)
            consensus_curve_mask_dict[file_name] = pad_resize_preserve_aspect_ratio(mask, 512)

            cnt += 1
            print(cnt)
            # if cnt%2==0: break

    # Take the consensus curves as ground truth
    # Find the dice and EF
    # Compute Dice:
    if not os.path.exists(os.path.join(save_folder, f'user_dice_scores.csv')):
        file_column_list = []
        dice_dict = {'File': file_column_list}
        for key, value in userids_dict.items():
            dice_dict[key] = []
        cnt = 0
        for key, value in file_user_mask_dict.items():
            file_name = key
            file_column_list.append(file_name)
            masks_dict = value
            cosnsensus_mask = consensus_curve_mask_dict[file_name]

            for userid in list(userids_dict.keys()):
                dice_list = dice_dict[userid]
                if userid in masks_dict:
                    mask = masks_dict[userid]
                    dice = measurements.compute_Dice_coefficient(cosnsensus_mask, mask)
                    dice_list.append(dice)
                else:
                    dice_list.append(None)

            # cnt+=1
            # if cnt % 2 == 0: break

        # Also add ssl method dice scores
        method1_dice = get_dice_scores_for_method(list(file_user_mask_dict.keys()),
                                                  'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/finetuning_4/Consensus_dataset_results_all_0.json',
                                                  'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/finetuning_4/Consensus_dataset_results_all_1.json')
        dice_dict['SimCLR4'] = list(method1_dice.values())
        method1_dice = get_dice_scores_for_method(list(file_user_mask_dict.keys()),
                                                  'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/finetuning_15/Consensus_dataset_results_all_0.json',
                                                  'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/finetuning_15/Consensus_dataset_results_all_1.json')
        dice_dict['SimCLR15'] = list(method1_dice.values())
        method1_dice = get_dice_scores_for_method(list(file_user_mask_dict.keys()),
                                                  'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/finetuning_100/Consensus_dataset_results_all_0.json',
                                                  'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/finetuning_100/Consensus_dataset_results_all_1.json')
        dice_dict['SimCLR100'] = list(method1_dice.values())

        # baseline_dice = get_dice_scores_for_method(list(file_user_mask_dict.keys()),
        #                                           'Baseline_444_Final/Baseline_with_4_percent_labelled_data/baseline_model_results_run_0/Consensus_dataset_results_all.json',
        #                                           'Baseline_444_Final/Baseline_with_4_percent_labelled_data/baseline_model_results_run_1/Consensus_dataset_results_all.json')
        # dice_dict['Rand  4'] = list(baseline_dice.values())

        df = pd.DataFrame(dice_dict)
        df.to_csv(os.path.join(save_folder, f'user_dice_scores.csv'), index=False)
    else:
        df = pd.read_csv(os.path.join(save_folder, f'user_dice_scores.csv'))

    nan_counts = df.isna().sum()
    selected_columns = df.iloc[:, 1:12]
    # Calculate the mean across all rows and selected columns
    mean_value = selected_columns.mean().mean()
    df = df.fillna(mean_value)
    # df = df.dropna()

    sorted_means = plot_dice_boxplots_ai_vs_expert(os.path.join(save_folder, f'plots_boxplot_dice.png'), df,
                                    userids_dict, userids_int_mappings_dict,
                                    userids_colour_dict)
    utils.write_dict_to_json(sorted_means.to_dict(), save_folder, f'plots_boxplot_dice.json')

    # T100
    t100_df = df[df['File'].isin(list(rankings_dict.keys())[0:100])]
    sorted_means = plot_dice_boxplots_ai_vs_expert(os.path.join(save_folder, f'plots_boxplot_dice_t100.png'), t100_df,
                                    userids_dict, userids_int_mappings_dict,
                                    userids_colour_dict)
    utils.write_dict_to_json(sorted_means.to_dict(), save_folder, f'plots_boxplot_dice_t100.json')

    # Compute EF:
    if not os.path.exists(os.path.join(save_folder, f'EF_Vol_results_all_users.json')):
        consensus_dataset_path = 'Data/Expert Consensus dataset'
        # First retrieve ground truths:
        ratios_dict = get_consensus_ratios()
        # Calculate and save consensus ground truth volumes:
        vols_gt_dict = calculate_consensus_ground_truth_volumes(consensus_dataset_path)
        vols_ml_gt_dict = convert_consensus_vols_to_ml(vols_gt_dict, ratios_dict)

        file_column_list = []
        super_vols_dict = {}
        for key, value in userids_dict.items():
            super_vols_dict[key] = {}

        if not os.path.exists(os.path.join(save_folder, 'user_disks_volumes_all_users.json')):
            for key, value in file_user_mask_dict.items():
                file_name = key
                masks_dict = value
                file_column_list.append(file_name)

                for userid in list(userids_dict.keys()):
                    vols_dict = super_vols_dict[userid]
                    if userid in masks_dict:
                        mask = masks_dict[userid]
                        (vol, poly_points, minmaxline,
                         midpointline, segments) = get_mask_volume_quick(mask, K=20, is_binary_image=True)

                        if poly_points is not None:
                            poly_points = poly_points.tolist()

                        vols_dict[file_name] = (vol, poly_points, minmaxline,
                                                midpointline, segments)
                    else:
                        vols_dict[file_name] = (0, [], [],
                                                [], [])

            write_dict_to_json(super_vols_dict,
                               save_folder,
                               f'user_disks_volumes_all_users.json')
        else:
            super_vols_dict = load_json(os.path.join(save_folder, f'user_disks_volumes_all_users.json'))

        super_vol_results_dict = {}
        for userid, vols_dict in super_vols_dict.items():
            vols_ml_dict = convert_consensus_vols_to_ml(vols_dict, ratios_dict)
            (info_dict,
             info_dict_list,
             avg_ef_error,
             avg_ED_vol_error,
             avg_ES_vol_error,
             count_no_ef) = calc_ejection_fractions_and_errors(vols_ml_gt_dict, vols_ml_dict)
            super_vol_results_dict[userid] = info_dict
            df = pd.DataFrame(info_dict_list)
            df.to_csv(os.path.join(save_folder, f'{userid}_EF_Vol_results_all.csv'), index=False)

        write_dict_to_json(super_vol_results_dict,
                           save_folder,
                           f'EF_Vol_results_all_users.json')
    else:
        super_vol_results_dict = load_json(os.path.join(save_folder, f'EF_Vol_results_all_users.json'))

    df_ef, df_edv, df_esv, df_ef_err, df_esv_err, df_edv_err = get_EF_dataframes(super_vol_results_dict)
    # nan_counts1 = df_ef.isna().sum()

    # fill a pandas DataFrame with the mean of each column wherever a value in the column is 0
    df_ef = fill_df_mean_per_column(df_ef)
    df_edv = fill_df_mean_per_column(df_edv)
    df_esv = fill_df_mean_per_column(df_esv)
    df_ef_err = fill_df_mean_per_column(df_ef_err)
    df_esv_err = fill_df_mean_per_column(df_esv_err)
    df_edv_err = fill_df_mean_per_column(df_edv_err)
    # Either fill the mean as above, or remove the row completely:
    # df_ef = df_ef[(df_ef != 0).all(axis=1)]
    # df_edv = df_edv[(df_edv != 0).all(axis=1)]
    # df_esv = df_esv[(df_esv != 0).all(axis=1)]
    # df_ef_err = df_ef_err[(df_ef_err != 0).all(axis=1)]
    # df_esv_err = df_esv_err[(df_esv_err != 0).all(axis=1)]
    # df_edv_err = df_edv_err[(df_edv_err != 0).all(axis=1)]

    print_df_zero_info(df_ef)
    print_df_zero_info(df_edv)
    print_df_zero_info(df_esv)
    print_df_zero_info(df_ef_err)
    print_df_zero_info(df_esv_err)
    print_df_zero_info(df_edv_err)

    sorted_means = plot_ef_err_boxplots_expert_vs_ai(df_ef_err, userids_dict, userids_int_mappings_dict, userids_colour_dict,
                                      os.path.join(save_folder, f'plots_boxplot_ef_err.png'), 'EF Error')
    utils.write_dict_to_json(sorted_means.to_dict(), save_folder, f'plots_boxplot_ef_err.json')
    sorted_means = plot_ef_err_boxplots_expert_vs_ai(df_edv_err, userids_dict, userids_int_mappings_dict, userids_colour_dict,
                                      os.path.join(save_folder, f'plots_boxplot_edv_err.png'), 'EDV Error (ml)')
    utils.write_dict_to_json(sorted_means.to_dict(), save_folder, f'plots_boxplot_edv_err.json')
    sorted_means = plot_ef_err_boxplots_expert_vs_ai(df_esv_err, userids_dict, userids_int_mappings_dict, userids_colour_dict,
                                      os.path.join(save_folder, f'plots_boxplot_esv_err.png'), 'ESV Error (ml)')
    utils.write_dict_to_json(sorted_means.to_dict(), save_folder, f'plots_boxplot_esv_err.json')

    plot_ef_bland_altman_expert_vs_ai(df_ef, userids_dict, userids_int_mappings_dict, userids_colour_dict,
                                      os.path.join(save_folder, f'plots_bland_altman_ef.png'), 'EF')
    plot_ef_bland_altman_expert_vs_ai(df_edv, userids_dict, userids_int_mappings_dict, userids_colour_dict,
                                      os.path.join(save_folder, f'plots_bland_altman_edv.png'), 'EDV (ml)')
    plot_ef_bland_altman_expert_vs_ai(df_esv, userids_dict, userids_int_mappings_dict, userids_colour_dict,
                                      os.path.join(save_folder, f'plots_bland_altman_esv.png'), 'ESV (ml)')

def get_avg_info_between_runs(vols_ml_gt_dict, ratios_dict, file_list, method1_masks_dict, method2_masks_dict):
    vols_dict1 = {}
    for file_name in file_list:
        mask = method1_masks_dict[file_name]
        (vol, poly_points, minmaxline,
         midpointline, segments) = get_mask_volume_quick(mask, K=20, is_binary_image=True)
        if poly_points is not None:
            poly_points = poly_points.tolist()
        vols_dict1[file_name] = (vol, poly_points, minmaxline,
                                midpointline, segments)

    vols_dict2 = {}
    for file_name in file_list:
        mask = method2_masks_dict[file_name]
        (vol, poly_points, minmaxline,
         midpointline, segments) = get_mask_volume_quick(mask, K=20, is_binary_image=True)
        if poly_points is not None:
            poly_points = poly_points.tolist()
        vols_dict2[file_name] = (vol, poly_points, minmaxline,
                                midpointline, segments)

    vols_ml_dict1 = convert_consensus_vols_to_ml(vols_dict1, ratios_dict)
    (info_dict1,
     info_dict_list1,
     avg_ef_error1,
     avg_ED_vol_error1,
     avg_ES_vol_error1,
     count_no_ef1) = calc_ejection_fractions_and_errors(vols_ml_gt_dict, vols_ml_dict1)

    vols_ml_dict2 = convert_consensus_vols_to_ml(vols_dict2, ratios_dict)
    (info_dict2,
     info_dict_list2,
     avg_ef_error2,
     avg_ED_vol_error2,
     avg_ES_vol_error2,
     count_no_ef2) = calc_ejection_fractions_and_errors(vols_ml_gt_dict, vols_ml_dict2)

    # avg info_dict1 and info_dict1 and create a new info_dict
    new_info_dict = {}
    for key, value1 in info_dict1.items():
        value2 = info_dict2[key]

        new_values_dict = {}
        if value1["Ejection Fraction gt"] ==0 or value2["Ejection Fraction gt"]==0:
            new_values_dict["Ejection Fraction gt"] = 0
        else:
            new_values_dict["Ejection Fraction gt"] = (value1["Ejection Fraction gt"] + value2["Ejection Fraction gt"])/2.0

        if value1["Ejection Fraction pred"] == 0 or value2["Ejection Fraction pred"] == 0:
            new_values_dict["Ejection Fraction pred"] = 0
        else:
            new_values_dict["Ejection Fraction pred"] = (value1["Ejection Fraction pred"] + value2["Ejection Fraction pred"])/2.0

        if value1["EF error"] == 0 or value2["EF error"] == 0:
            new_values_dict["EF error"] = 0
        else:
            new_values_dict["EF error"] = (value1["EF error"] + value2["EF error"])/2.0

        if value1["EDV gt"] == 0 or value2["EDV gt"] == 0:
            new_values_dict["EDV gt"] = 0
        else:
            new_values_dict["EDV gt"] = (value1["EDV gt"] + value2["EDV gt"])/2.0

        if value1["ESV gt"] == 0 or value2["ESV gt"] == 0:
            new_values_dict["ESV gt"] = 0
        else:
            new_values_dict["ESV gt"] = (value1["ESV gt"] + value2["ESV gt"])/2.0

        if value1["EDV pred"] == 0 or value2["EDV pred"] == 0:
            new_values_dict["EDV pred"] = 0
        else:
            new_values_dict["EDV pred"] = (value1["EDV pred"] + value2["EDV pred"])/2.0

        if value1["ESV pred"] == 0 or value2["ESV pred"] == 0:
            new_values_dict["ESV pred"] = 0
        else:
            new_values_dict["ESV pred"] = (value1["ESV pred"] + value2["ESV pred"])/2.0

        if value1["EDV error"] == 0 or value2["EDV error"] == 0:
            new_values_dict["EDV error"] = 0
        else:
            new_values_dict["EDV error"] = (value1["EDV error"] + value2["EDV error"])/2.0

        if value1["ESV error"] == 0 or value2["ESV error"] == 0:
            new_values_dict["ESV error"] = 0
        else:
            new_values_dict["ESV error"] = (value1["ESV error"] + value2["ESV error"])/2.0

        new_values_dict["ED frame"] = value1["ED frame"]
        new_values_dict["ES frame"] = value1["ES frame"]


        new_info_dict[key] = new_values_dict

    return new_info_dict

def one_expert_agreement_vs_ai(rankings_file, userid_index):
    rankings_dict = load_json(rankings_file)

    # Hardcode these so they're always the same
    userids_colour_dict = get_user_colours_dict()
    userids_int_mappings_dict = get_user_int_mappings_dict()
    userid_for_comparison = list(userids_int_mappings_dict.keys())[userid_index]

    save_folder = f'results_one_expert_vs_ai_ssl_{userids_int_mappings_dict[userid_for_comparison]}'
    if not os.path.exists(save_folder): os.mkdir(save_folder)

    csv_file = r'Data/Expert Consensus dataset/user-labels-all.csv'

    file_and_coords = get_coordinates(csv_file)

    print(len(file_and_coords))

    utils.write_dict_to_json(userids_int_mappings_dict, save_folder, 'userids_integer_mappings.json')

    #reorder points
    file_and_coords_reordered = {}
    userids_dict = {}
    for fileid, value in file_and_coords.items():
        users_dict = {}
        for userid, coords in value.items():
            users_dict[userid] = reorder_points(coords)
            if userid not in userids_dict:
                userids_dict[userid] = 0
        file_and_coords_reordered[fileid] = users_dict

    files_avg_coords = get_avg_coordinates_exclude(file_and_coords_reordered, userid_for_comparison)

    output_folder = os.path.join(save_folder, 'temp_images_consensus_orig')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    get_original_files(files_avg_coords, output_folder)

    if not os.path.exists(os.path.join(save_folder, f'user_dice_scores.csv')):
        #Get masks of each user annotation
        file_user_mask_dict = {}
        consensus_curve_mask_dict = {}
        cnt=0
        for key, value in file_and_coords_reordered.items():  # loop through dataset
            file_name = key
            original_image = cv2.imread(os.path.join(output_folder, file_name), cv2.IMREAD_GRAYSCALE)
            num_coord_sets = len(value)  # example 10 annotations

            points_per_set = []
            user_masks = {}

            for userid, coords in value.items():  # loop through each expert annotation
                coords_list = []
                for j in range(len(coords[0])):  # loop through 200 points
                    x_coord = coords[0][j]
                    y_coord = coords[1][j]
                    coords_list.append((x_coord, y_coord))
                pts = np.array(coords_list, np.int32)
                points_per_set.append(coords_list)

                # create mask
                blank_image = np.zeros((original_image.shape[0], original_image.shape[1]))
                mask = draw_poly_on_image(blank_image, pts, (1, 1, 1), False)
                mask = fill_poly_on_image(mask, pts, (1, 1, 1))
                for i in range(original_image.shape[0]):
                    for j in range(original_image.shape[1]):
                        mask[i, j] = np.uint8(mask[i, j])
                user_masks[userid] = pad_resize_preserve_aspect_ratio(mask, 512)

            file_user_mask_dict[file_name] = user_masks

            # Get masks for consensus/avg curve:
            consensus_curve_coords = files_avg_coords[file_name]
            coords_list = []
            for j in range(len(consensus_curve_coords[0])):  # loop through 200 points
                x_coord = consensus_curve_coords[0][j]
                y_coord = consensus_curve_coords[1][j]
                coords_list.append((x_coord, y_coord))
            pts = np.array(coords_list, np.int32)
            blank_image = np.zeros((original_image.shape[0], original_image.shape[1]))
            mask = draw_poly_on_image(blank_image, pts, (1, 1, 1), False)
            mask = fill_poly_on_image(mask, pts, (1, 1, 1))
            for i in range(original_image.shape[0]):
                for j in range(original_image.shape[1]):
                    mask[i, j] = np.uint8(mask[i, j])
            consensus_curve_mask_dict[file_name] = pad_resize_preserve_aspect_ratio(mask, 512)

            cnt+=1
            print(cnt)
            #if cnt%2==0: break

    # Take the consensus curves as ground truth
    # Find the dice and EF
    # Compute Dice:
    if not os.path.exists(os.path.join(save_folder, f'user_dice_scores.csv')):

        # Since we excluded 1 expert, that means we have a consensus curve without that expert's contribution,
        # we need to recompute the dice scores for the ssl models also
        simclr4_run0_masks_dict = get_raw_masks_for_method(
            'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/finetuning_4/Consensus_dataset_results_all_0_raw_model_outputs.json')
        simclr4_run1_masks_dict = get_raw_masks_for_method(
            'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/finetuning_4/Consensus_dataset_results_all_1_raw_model_outputs.json')
        simclr15_run0_masks_dict = get_raw_masks_for_method(
            'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/finetuning_15/Consensus_dataset_results_all_0_raw_model_outputs.json')
        simclr15_run1_masks_dict = get_raw_masks_for_method(
            'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/finetuning_15/Consensus_dataset_results_all_1_raw_model_outputs.json')
        simclr100_run0_masks_dict = get_raw_masks_for_method(
            'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/finetuning_100/Consensus_dataset_results_all_0_raw_model_outputs.json')
        simclr100_run1_masks_dict = get_raw_masks_for_method(
            'run_SimCLR_Unity_unet_bce_True_batch20/Exp_with_1900_percent_unlabelled_data/finetuning_100/Consensus_dataset_results_all_1_raw_model_outputs.json')

        file_column_list = []
        dice_dict = {'File':file_column_list}
        for key, value in userids_dict.items():
            dice_dict[key] = []
        cnt=0
        for key, value in file_user_mask_dict.items():
            file_name = key
            file_column_list.append(file_name)
            masks_dict = value
            consensus_mask = consensus_curve_mask_dict[file_name]

            for userid in list(userids_dict.keys()):
                dice_list = dice_dict[userid]
                if userid in masks_dict:
                    mask = masks_dict[userid]
                    dice = measurements.compute_Dice_coefficient(consensus_mask, mask)
                    dice_list.append(dice)
                else:
                    dice_list.append(None)

            #cnt+=1
            #if cnt % 2 == 0: break

        #recompute dice for ssl models
        dice_ssl_list1 = []
        dice_ssl_list2 = []
        dice_ssl_list3 = []
        for key, value in file_user_mask_dict.items():
            consensus_mask = consensus_curve_mask_dict[key]
            simclr4_run0_mask = simclr4_run0_masks_dict[key]
            simclr4_run1_mask = simclr4_run1_masks_dict[key]
            dice1 = (measurements.compute_Dice_coefficient(consensus_mask, simclr4_run0_mask) +
                    measurements.compute_Dice_coefficient(consensus_mask, simclr4_run1_mask)) /2.0
            dice_ssl_list1.append(dice1)
            simclr15_run0_mask = simclr15_run0_masks_dict[key]
            simclr15_run1_mask = simclr15_run1_masks_dict[key]
            dice2 = (measurements.compute_Dice_coefficient(consensus_mask, simclr15_run0_mask) +
                     measurements.compute_Dice_coefficient(consensus_mask, simclr15_run1_mask)) / 2.0
            dice_ssl_list2.append(dice2)
            simclr100_run0_mask = simclr100_run0_masks_dict[key]
            simclr100_run1_mask = simclr100_run1_masks_dict[key]
            dice3 = (measurements.compute_Dice_coefficient(consensus_mask, simclr100_run0_mask) +
                     measurements.compute_Dice_coefficient(consensus_mask, simclr100_run1_mask)) / 2.0
            dice_ssl_list3.append(dice3)

        # Also add ssl method dice scores
        dice_dict['SimCLR4'] = dice_ssl_list1
        dice_dict['SimCLR15'] = dice_ssl_list2
        dice_dict['SimCLR100'] = dice_ssl_list3

        df = pd.DataFrame(dice_dict)
        df.to_csv(os.path.join(save_folder, f'user_dice_scores.csv'), index=False)
    else:
        df = pd.read_csv(os.path.join(save_folder, f'user_dice_scores.csv'))

    nan_counts = df.isna().sum()
    selected_columns = df.iloc[:, 1:12]
    # Calculate the mean across all rows and selected columns
    mean_value = selected_columns.mean().mean()
    df = df.fillna(mean_value)
    #df = df.dropna()

    sorted_means = plot_dice_boxplots_ai_vs_one_expert(os.path.join(save_folder, f'plots_boxplot_dice.png'), df,
                                    userids_dict, userids_int_mappings_dict,
                                    userids_colour_dict, userid_for_comparison)
    utils.write_dict_to_json(sorted_means.to_dict(), save_folder, f'plots_boxplot_dice.json')

    #T100
    t100_df = df[df['File'].isin(list(rankings_dict.keys())[0:100])]
    sorted_means = plot_dice_boxplots_ai_vs_one_expert(os.path.join(save_folder, f'plots_boxplot_dice_t100.png'), t100_df,
                                    userids_dict, userids_int_mappings_dict,
                                    userids_colour_dict, userid_for_comparison)
    utils.write_dict_to_json(sorted_means.to_dict(), save_folder, f'plots_boxplot_dice_t100.json')

    # Compute EF:
    if not os.path.exists(os.path.join(save_folder,f'EF_Vol_results_all_users.json')):
        consensus_dataset_path = 'Data/Expert Consensus dataset'
        # First retrieve ground truths:
        ratios_dict = get_consensus_ratios()
        # Calculate and save consensus ground truth volumes:
        vols_gt_dict = {}
        for file, mask in consensus_curve_mask_dict.items():
            (vol, poly_points, minmaxline,
             midpointline, segments) = get_mask_volume_quick(mask, K=20, is_binary_image=True)
            if poly_points is not None:
                poly_points = poly_points.tolist()
            vols_gt_dict[file] = [vol, poly_points, minmaxline, midpointline, segments]

        vols_ml_gt_dict = convert_consensus_vols_to_ml(vols_gt_dict, ratios_dict)

        file_column_list = []
        super_vols_dict = {}
        for key, value in userids_dict.items():
            super_vols_dict[key] = {}

        if not os.path.exists(os.path.join(save_folder, 'user_disks_volumes_all_users.json')):
            for key, value in file_user_mask_dict.items():
                file_name = key
                masks_dict = value
                file_column_list.append(file_name)

                for userid in list(userids_dict.keys()):
                    vols_dict = super_vols_dict[userid]
                    if userid in masks_dict:
                        mask = masks_dict[userid]
                        (vol, poly_points, minmaxline,
                         midpointline, segments) = get_mask_volume_quick(mask, K=20, is_binary_image=True)

                        if poly_points is not None:
                            poly_points = poly_points.tolist()

                        vols_dict[file_name] = (vol, poly_points, minmaxline,
                                                             midpointline, segments)
                    else:
                        vols_dict[file_name] = (0, [], [],
                                                [], [])

            write_dict_to_json(super_vols_dict,
                               save_folder,
                               f'user_disks_volumes_all_users.json')
        else:
            super_vols_dict = load_json(os.path.join(save_folder, f'user_disks_volumes_all_users.json'))

        super_vol_results_dict = {}
        for userid, vols_dict in super_vols_dict.items():
            vols_ml_dict = convert_consensus_vols_to_ml(vols_dict, ratios_dict)
            (info_dict,
             info_dict_list,
             avg_ef_error,
             avg_ED_vol_error,
             avg_ES_vol_error,
             count_no_ef) = calc_ejection_fractions_and_errors(vols_ml_gt_dict, vols_ml_dict)
            super_vol_results_dict[userid] = info_dict
            df = pd.DataFrame(info_dict_list)
            df.to_csv(os.path.join(save_folder, f'{userid}_EF_Vol_results_all.csv'), index=False)

        ############################################
        # Since we excluded one expert from the average curve, we have to recompute ef for SSL methods also
        # Do the same as above but for the SSL methods
        super_vol_results_dict['SimCLR4'] = get_avg_info_between_runs(vols_ml_gt_dict, ratios_dict,
                                                               list(file_user_mask_dict.keys()),
                                                               simclr4_run0_masks_dict, simclr4_run1_masks_dict)
        super_vol_results_dict['SimCLR15'] = get_avg_info_between_runs(vols_ml_gt_dict, ratios_dict,
                                                               list(file_user_mask_dict.keys()),
                                                               simclr15_run0_masks_dict, simclr15_run1_masks_dict)
        super_vol_results_dict['SimCLR100'] = get_avg_info_between_runs(vols_ml_gt_dict, ratios_dict,
                                                               list(file_user_mask_dict.keys()),
                                                               simclr100_run0_masks_dict, simclr100_run1_masks_dict)

        ############################################

        write_dict_to_json(super_vol_results_dict,
                           save_folder,
                           f'EF_Vol_results_all_users.json')
    else:
        super_vol_results_dict = load_json(os.path.join(save_folder, f'EF_Vol_results_all_users.json'))

    df_ef, df_edv, df_esv, df_ef_err, df_esv_err, df_edv_err = get_EF_dataframes_without_sslmethods(super_vol_results_dict)
    # nan_counts1 = df_ef.isna().sum()

    #fill a pandas DataFrame with the mean of each column wherever a value in the column is 0
    df_ef = fill_df_mean_per_column(df_ef)
    df_edv = fill_df_mean_per_column(df_edv)
    df_esv = fill_df_mean_per_column(df_esv)
    df_ef_err = fill_df_mean_per_column(df_ef_err)
    df_esv_err = fill_df_mean_per_column(df_esv_err)
    df_edv_err = fill_df_mean_per_column(df_edv_err)
    #Either fill the mean as above, or remove the row completely:
    # df_ef = df_ef[(df_ef != 0).all(axis=1)]
    # df_edv = df_edv[(df_edv != 0).all(axis=1)]
    # df_esv = df_esv[(df_esv != 0).all(axis=1)]
    # df_ef_err = df_ef_err[(df_ef_err != 0).all(axis=1)]
    # df_esv_err = df_esv_err[(df_esv_err != 0).all(axis=1)]
    # df_edv_err = df_edv_err[(df_edv_err != 0).all(axis=1)]

    print_df_zero_info(df_ef)
    print_df_zero_info(df_edv)
    print_df_zero_info(df_esv)
    print_df_zero_info(df_ef_err)
    print_df_zero_info(df_esv_err)
    print_df_zero_info(df_edv_err)

    sorted_means = plot_ef_err_boxplots_ai_vs_one_expert(df_ef_err, userids_dict, userids_int_mappings_dict, userids_colour_dict,
                                      os.path.join(save_folder, f'plots_boxplot_ef_err.png'), 'EF Error', userid_for_comparison)
    utils.write_dict_to_json(sorted_means.to_dict(), save_folder, f'plots_boxplot_ef_err.json')
    sorted_means = plot_ef_err_boxplots_ai_vs_one_expert(df_edv_err, userids_dict, userids_int_mappings_dict, userids_colour_dict,
                                      os.path.join(save_folder, f'plots_boxplot_edv_err.png'), 'EDV Error (ml)', userid_for_comparison)
    utils.write_dict_to_json(sorted_means.to_dict(), save_folder, f'plots_boxplot_edv_err.json')
    sorted_means = plot_ef_err_boxplots_ai_vs_one_expert(df_esv_err, userids_dict, userids_int_mappings_dict, userids_colour_dict,
                                      os.path.join(save_folder, f'plots_boxplot_esv_err.png'), 'ESV Error (ml)', userid_for_comparison)
    utils.write_dict_to_json(sorted_means.to_dict(), save_folder, f'plots_boxplot_esv_err.json')

    plot_ef_bland_altman_expert_vs_ai(df_ef, userids_dict, userids_int_mappings_dict, userids_colour_dict,
                                      os.path.join(save_folder, f'plots_bland_altman_ef.png'), 'EF')
    plot_ef_bland_altman_expert_vs_ai(df_edv, userids_dict, userids_int_mappings_dict, userids_colour_dict,
                                      os.path.join(save_folder, f'plots_bland_altman_edv.png'), 'EDV (ml)')
    plot_ef_bland_altman_expert_vs_ai(df_esv, userids_dict, userids_int_mappings_dict, userids_colour_dict,
                                      os.path.join(save_folder, f'plots_bland_altman_esv.png'), 'ESV (ml)')


def evaluate_ssl(main_folder, consensus_dataset_folder, name):
    compute_ejection_fraction_consensus_set(main_folder, consensus_dataset_folder)
    write_avg_results_per_encoder_percentage(main_folder, name)
    write_comparisons()

def main():
    consensus_dataset_folder = 'Data/Expert Consensus dataset'

    evaluate_ssl('run_BTwin_unet_bce_True_batch20', consensus_dataset_folder, 'Pretext-BTwin')
    evaluate_ssl('run_Patch_rand_unity_patch_unet_bce_False_batch16', consensus_dataset_folder, 'Region-Based')
    evaluate_ssl('run_Patch_rand_unity_horizontal_unet_bce_False_batch16', consensus_dataset_folder, 'Strip-Based')
    evaluate_ssl('run_ssl_rotation_unet_bce_False_batch16', consensus_dataset_folder, 'Pretext-Rotation')
    evaluate_ssl('run_SimCLR_Unity_unet_bce_True_batch20', consensus_dataset_folder, 'Pretext-SimCLR')
    evaluate_ssl('run_Split2_unet_bce_False_batch16', consensus_dataset_folder, 'Pretext-Split')

    grouped_box_plot_figure_enc_vs_dec()
    grouped_box_plot_figure_batch_size_simclr()
    grouped_box_plot_figure_unlabelled()
    grouped_box_plot_figure_methods_compare()

    #Uncomment to create the rankings.json
    #rankings_dict = compute_rankings()
    expert_agreement_vs_ai('results_expert_vs_ai_ssl/rankings.json')
    one_expert_agreement_vs_ai('results_expert_vs_ai_ssl/rankings.json', 0)
    one_expert_agreement_vs_ai('results_expert_vs_ai_ssl/rankings.json', 1)
    one_expert_agreement_vs_ai('results_expert_vs_ai_ssl/rankings.json', 2)
    one_expert_agreement_vs_ai('results_expert_vs_ai_ssl/rankings.json', 3)
    one_expert_agreement_vs_ai('results_expert_vs_ai_ssl/rankings.json', 4)
    one_expert_agreement_vs_ai('results_expert_vs_ai_ssl/rankings.json', 5)
    one_expert_agreement_vs_ai('results_expert_vs_ai_ssl/rankings.json', 6)
    one_expert_agreement_vs_ai('results_expert_vs_ai_ssl/rankings.json', 7)
    one_expert_agreement_vs_ai('results_expert_vs_ai_ssl/rankings.json', 8)
    one_expert_agreement_vs_ai('results_expert_vs_ai_ssl/rankings.json', 9)
    one_expert_agreement_vs_ai('results_expert_vs_ai_ssl/rankings.json', 10)

    find_low_ef_error_with_bad_dice()

    print('Complete!')


if __name__ == "__main__":
    main()