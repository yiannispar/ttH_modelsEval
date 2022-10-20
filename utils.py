# utils.py
#
# author: I. Paraskevas <ioannis.paraskevas@cern.ch>
# created: May, 2022

import os
import glob
import itertools 
import numpy as np

def check_if_output_dir_exists(output_dir):
    if output_dir == "":
        return
    isExist = os.path.exists(output_dir)
    if not isExist:
        print("output dir", output_dir, "not found. Creating one for you...")
        os.makedirs(output_dir)

def get_pred_values(bdt_cl0,bdt_cl1,bdt_cl2,bdt_cl3,bdt_cl4,bdt_cl5):

    assert len(bdt_cl0) == len(bdt_cl1) == len(bdt_cl2) == len(bdt_cl3) == len(bdt_cl4) == len(bdt_cl5) 

    pred_values = list()

    for (cl0, cl1, cl2, cl3, cl4, cl5) in zip(bdt_cl0, bdt_cl1, bdt_cl2, bdt_cl3, bdt_cl4, bdt_cl5):
        class_list =[cl0,cl1,cl2,cl3,cl4,cl5]
        max_index = class_list.index(max(class_list)) #pos of max value in list
        pred_values.append(max_index)
        
    return np.array(pred_values)

def print_plots_dir(output_dir):
    working_dir = os.getcwd()
    if output_dir:
        print("All plots saved at", working_dir + "/" + output_dir)
    else:
        print("All plots saved at", working_dir)

# remove PN events with genEventClassifier = 2 (not included in PN training)
def fix_truth_and_scores(truth_values, scores_ttHcc, scores_ttHbb, scores_ttLF, scores_ttcc, scores_ttbb):

    list_truth_values = list()
    list_scores_ttHcc = list()
    list_scores_ttHbb = list()
    list_scores_ttLF = list()
    list_scores_ttcc = list()
    list_scores_ttbb = list()
    
    for (truth_value, score_ttHcc, score_ttHbb, score_ttLF, score_ttcc, score_ttbb) in zip(truth_values, scores_ttHcc, scores_ttHbb, scores_ttLF, scores_ttcc, scores_ttbb):
        if truth_value != 2: #2 is not included in PN training
            list_truth_values.append(truth_value)
            list_scores_ttHcc.append(score_ttHcc)
            list_scores_ttHbb.append(score_ttHbb)
            list_scores_ttLF.append(score_ttLF)
            list_scores_ttcc.append(score_ttcc)
            list_scores_ttbb.append(score_ttbb)

    return list_truth_values, list_scores_ttHcc, list_scores_ttHbb, list_scores_ttLF, list_scores_ttcc, list_scores_ttbb

# int to str true labels for PN
def int_to_str_labels_pn(truth_values):

    labels = list()
    
    for value in truth_values:
        if value == 0:
            labels.append("ttHcc")
        elif value == 1:
            labels.append("ttHbb")
        elif value == 3:
            labels.append("tt+lf")
        elif value >= 4 and value <= 6:
            labels.append("tt+cf")
        elif value >= 7 and value <= 9:
            labels.append("tt+bf")
        else:
            print(value)

    return np.array(labels)

def get_pred_str_labels_pn(scores_ttHcc,scores_ttHbb,scores_ttLF,scores_ttcc,scores_ttbb):

    assert len(scores_ttHcc) == len(scores_ttHbb) == len(scores_ttLF) == len(scores_ttcc) == len(scores_ttbb)

    pred_values = list()
    class_list_string = ["ttHcc","ttHbb","tt+lf","tt+cf","tt+bf"]
        
    for (score_ttHcc,score_ttHbb,score_ttLF,score_ttcc,score_ttbb) in zip(scores_ttHcc,scores_ttHbb,scores_ttLF,scores_ttcc,scores_ttbb):
        class_list =[score_ttHcc,score_ttHbb,score_ttLF,score_ttcc,score_ttbb]
        max_index = class_list.index(max(class_list)) #pos of max value in list
        pred_values.append(class_list_string[max_index])

    return np.array(pred_values)

# int to str labels for XGBoost
def int_to_str_labels(values):

    labels = list()
    
    for value in values:
        if value == 0:
            labels.append("ttHcc")
        elif value == 1:
            labels.append("ttHbb")
        elif value == 2:
            labels.append("ttHtautau")
        elif value == 3:
            labels.append("tt+lf")
        elif value == 4:
            labels.append("tt+cf")
        elif value == 5:
            labels.append("tt+bf")
        else:
            print("label not recognised")

    return np.array(labels)


