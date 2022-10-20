# draw.py
#
# author: I. Paraskevas <ioannis.paraskevas@cern.ch>
# created: May, 2022

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import utils
import json
import uproot
import numpy as np
import itertools
import pandas as pd
import glob
import sys

colours = ["r","g","b","magenta","orange","cyan"]

def make_plots(output_dir,config):

    option = json.loads(config.get("plots", "option"))
    if option == "xgb":
        print("Making plots for XGBoost model")
        make_plots_xgb(output_dir,config)
    elif option == "pn":
        print("Making plots for ParticleNet model")
        make_plots_pn(output_dir,config)
    elif option == "both":
        print("Making plots for both models")
        make_plots_xgb(output_dir,config)
        make_plots_pn(output_dir,config)
    else:
        print("Option not recognised. Please choose between xgb,pn,both")

# XGB plots
def make_plots_xgb(output_dir,config):

    plot_cm_xgb(output_dir,config)
    plot_roc_xgb(output_dir,config)

# PN plots
def make_plots_pn(output_dir,config):

    plot_cm_pn(output_dir,config)
    plot_roc_pn(output_dir,config)

def plot_cm_xgb(output_dir,config):

    tree_name = json.loads(config.get("root", "tree_name_xgb"))
    root_files_dir = json.loads(config.get("root", "dir_to_look_for_files_xgb"))

    root_files_found = glob.glob(root_files_dir + "*.root")
    if not root_files_found:
        print("no root files found")
        sys.exit(0)

    print("plotting confusion matrix from files:")
    [print(i) for i in root_files_found]

    all_data = uproot.concatenate(root_files_dir + "*.root:" + tree_name, library="np")

    true_int_labels = all_data["genEventClassifier"]
    bdt_cl0 = all_data["bdtScore_cl0"]
    bdt_cl1 = all_data["bdtScore_cl1"]
    bdt_cl2 = all_data["bdtScore_cl2"]
    bdt_cl3 = all_data["bdtScore_cl3"]
    bdt_cl4 = all_data["bdtScore_cl4"]
    bdt_cl5 = all_data["bdtScore_cl5"]

    pred_int_labels = utils.get_pred_values(bdt_cl0, bdt_cl1, bdt_cl2, bdt_cl3, bdt_cl4, bdt_cl5)

    # convert integers to strings
    true_str_labels = utils.int_to_str_labels(true_int_labels)
    pred_str_labels = utils.int_to_str_labels(pred_int_labels)
    
    cm = confusion_matrix(true_str_labels, pred_str_labels, normalize='true', labels=["ttHcc","ttHbb","ttHtautau","tt+lf","tt+cf","tt+bf"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = ["ttHcc","ttHbb","ttHtautau","tt+lf","tt+cf","tt+bf"])
    disp.plot()
    plt.title("XGBoost Model")
    plt.savefig(output_dir + "CM_XGB.png",dpi=200)

# make ROC curves for XGB
def plot_roc_xgb(output_dir,config):

    tree_name = json.loads(config.get("root", "tree_name_xgb"))
    root_files_dir = json.loads(config.get("root", "dir_to_look_for_files_xgb"))

    root_files_found = glob.glob(root_files_dir + "*.root")
    if not root_files_found:
        print("no root files found")
        sys.exit(0)

    print("plotting ROC curve from files:")
    [print(i) for i in root_files_found]
    
    all_data = uproot.concatenate(root_files_dir + "*.root:" + tree_name, library="np")

    true_int_labels = all_data["genEventClassifier"]
    bdt_cl0 = all_data["bdtScore_cl0"]
    bdt_cl1 = all_data["bdtScore_cl1"]
    bdt_cl2 = all_data["bdtScore_cl2"]
    bdt_cl3 = all_data["bdtScore_cl3"]
    bdt_cl4 = all_data["bdtScore_cl4"]
    bdt_cl5 = all_data["bdtScore_cl5"]

    # convert integers to strings
    true_str_labels = utils.int_to_str_labels(true_int_labels)

    plt.figure()
    plt.title("XGBoost Model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    fpr, tpr, thresholds = roc_curve(true_str_labels,bdt_cl0,pos_label="ttHcc")
    roc_auc = auc(fpr, tpr)
    label_ = "Sig: ttHcc, Bkg: Rest, AUC = " + str(round(roc_auc,2))
    plt.plot(fpr,tpr,color=colours[0],label=label_)
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.legend(loc="lower right")
    plt.savefig(output_dir + "ROC_XGB.png", dpi=200)

# make CM for PN
def plot_cm_pn(output_dir,config):

    tree_name = json.loads(config.get("root", "tree_name_pn"))
    root_files_dir = json.loads(config.get("root", "dir_to_look_for_files_pn"))

    root_files_found = glob.glob(root_files_dir + "*.root")
    if not root_files_found:
        print("no root files found")
        sys.exit(0)

    print("plotting confusion matrix from files:")
    [print(i) for i in root_files_found]
    
    all_data = uproot.concatenate(root_files_dir + "*.root:" + tree_name, library="np")
    true_int_labels = all_data["genEventClassifier"]
    scores_ttHcc = all_data["score_ttHcc"]
    scores_ttHbb = all_data["score_ttHbb"]
    scores_ttLF = all_data["score_ttLF"]
    scores_ttcc = all_data["score_ttcc"]
    scores_ttbb = all_data["score_ttbb"]

    # remove events with genEventClassifier = 2 (ttHtautau not included in PN training)
    true_int_labels, scores_ttHcc, scores_ttHbb, scores_ttLF, scores_ttcc, scores_ttbb = utils.fix_truth_and_scores(true_int_labels, scores_ttHcc, scores_ttHbb, scores_ttLF, scores_ttcc, scores_ttbb)

    # convert integer to string labels
    true_str_labels = utils.int_to_str_labels_pn(true_int_labels)
    pred_str_labels = utils.get_pred_str_labels_pn(scores_ttHcc,scores_ttHbb,scores_ttLF,scores_ttcc,scores_ttbb)

    cm = confusion_matrix(true_str_labels, pred_str_labels, normalize='true', labels=["ttHcc","ttHbb","tt+lf","tt+cf","tt+bf"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = ["ttHcc","ttHbb","tt+lf","tt+cf","tt+bf"])
    disp.plot()
    plt.title("ParticleNet Model")
    plt.savefig(output_dir + "CM_PN.png",dpi=200)

# make ROC curves for PN
def plot_roc_pn(output_dir,config):

    tree_name = json.loads(config.get("root", "tree_name_pn"))
    root_files_dir = json.loads(config.get("root", "dir_to_look_for_files_pn"))

    root_files_found = glob.glob(root_files_dir + "*.root")
    if not root_files_found:
        print("no root files found")
        sys.exit(0)

    print("plotting ROC curve from files:")
    [print(i) for i in root_files_found]

    all_data = uproot.concatenate(root_files_dir + "*.root:" + tree_name, library="np")
    true_int_labels = all_data["genEventClassifier"]
    scores_ttHcc = all_data["score_ttHcc"]
    scores_ttHbb = all_data["score_ttHbb"]
    scores_ttLF = all_data["score_ttLF"]
    scores_ttcc = all_data["score_ttcc"]
    scores_ttbb = all_data["score_ttbb"]

    # remove events with genEventClassifier = 2 (ttHtautau not included in PN training)
    true_int_labels, scores_ttHcc, scores_ttHbb, scores_ttLF, scores_ttcc, scores_ttbb = utils.fix_truth_and_scores(true_int_labels, scores_ttHcc, scores_ttHbb, scores_ttLF, scores_ttcc, scores_ttbb)

    # convert integer to string labels
    true_str_labels = utils.int_to_str_labels_pn(true_int_labels)

    plt.figure()
    plt.title("ParticleNet Model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    fpr, tpr, thresholds = roc_curve(true_str_labels,scores_ttHcc,pos_label="ttHcc")
    roc_auc = auc(fpr, tpr)
    label_ = "Sig: ttHcc, Bkg: Rest, AUC = " + str(round(roc_auc,2))
    plt.plot(fpr,tpr,color=colours[0],label=label_)
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.legend(loc="lower right")
    plt.savefig(output_dir + "ROC_PN.png", dpi=200)
