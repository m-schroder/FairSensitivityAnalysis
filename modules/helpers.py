import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns

import os, random


#-------------------------------------------------------------General helper functions--------------------------------------------------------

def seed(seed):

    pl.seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

seed(1)


#------------------------------------------------------------- Plotting --------------------------------------------------

def plot_bounds_over_gamma(gammas, accuracy, oracle_fairness, fairness_ub, fairness_lb, path=None):

    data_plot = pd.DataFrame({"gammas": gammas, "accuracy": accuracy, "oracle_fairness_DE": oracle_fairness[0,:], "oracle_fairness_IE": oracle_fairness[1,:], "oracle_fairness_SE": oracle_fairness[2,:]})
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=data_plot, x=gammas, y=accuracy, label="Accuracy", color="darkred", linewidth=3)

    colors = sns.color_palette("crest", 3)

    legend_labels = ["DE", "IE", "SE"]
    for i in range(3):
        sns.lineplot(data=data_plot, x="gammas", y=("oracle_fairness_" + legend_labels[i]), label=legend_labels[i], color=colors[i], linewidth=2)
        plt.fill_between(gammas, fairness_ub[i,:], fairness_lb[i,:], alpha=0.5, color=colors[i])

    plt.xlabel("Sensitivity parameter")
    plt.ylabel("Effect")
    plt.legend()

    if path != None:
        plt.savefig(path)
    plt.show()



def plot_bounds_over_confounding(conf_level, bounds, data_fairness, path=None):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(40, 8))

    sns.set(style="whitegrid", font_scale=1.5)
    
    palette = sns.color_palette("pastel")   
    colors = [palette[7], palette[4], palette[1]]

    legend_labels = ["Standard", "Fair naive", "Fair robust (ours)"]
    classifier_labels = ["StandardClf", "FairClf_naive", "FairClf"]
    
    #ax1 -> DE
    sns.lineplot(x=conf_level, y=data_fairness["DE"], label="Oracle unfairness", color="darkred", linewidth=3, ax = ax1)
    ax1.plot(conf_level, np.array([0,0,0,0]), label="Optimal fairness", linestyle='dashed', color = "grey", linewidth=3)
    for i in range(3):
        for param in bounds["Sensitivity parameter"].unique():
            ax1.fill_between(conf_level, bounds.loc[(bounds["Model"]==classifier_labels[i]) & (bounds["Sensitivity parameter"] == param)]["DE_ub"].values, bounds.loc[(bounds["Model"]==classifier_labels[i]) & (bounds["Sensitivity parameter"] == param)]["DE_lb"].values, alpha=0.6, label=legend_labels[i] + ", $\Gamma_M$ = "+ str(param), color=colors[i]) 
    ax1.set_xlabel("Confounding level \u03A6",fontsize="30")
    ax1.set_ylabel("Direct effect",fontsize="30")
    ax1.legend(fontsize="20")

    #ax2 -> IE
    sns.lineplot(x=conf_level, y=data_fairness["IE"], color="darkred", linewidth=3, ax = ax2)
    ax2.plot(conf_level, np.array([0,0,0,0]), label="Optimal fairness", linestyle='dashed', color = "grey", linewidth=3)
    for i in range(3):
        for param in bounds["Sensitivity parameter"].unique():
            ax2.fill_between(conf_level, bounds.loc[(bounds["Model"]==classifier_labels[i]) & (bounds["Sensitivity parameter"] == param)]["IE_ub"].values, bounds.loc[(bounds["Model"]==classifier_labels[i]) & (bounds["Sensitivity parameter"] == param)]["IE_lb"].values, alpha=0.6, label=legend_labels[i] + ", $\Gamma_M$ = "+ str(param), color=colors[i]) 
    ax2.set_xlabel("Confounding level \u03A6",fontsize="30")
    ax2.set_ylabel("Indirect effect",fontsize="30")

    #ax3 -> SE
    sns.lineplot(x=conf_level, y=data_fairness["SE"], color="darkred", linewidth=3, ax = ax3)
    ax3.plot(conf_level, np.array([0,0,0,0]), label="Optimal fairness", linestyle='dashed', color = "grey", linewidth=3)
    for i in range(3):
        for param in bounds["Sensitivity parameter"]:
            ax3.fill_between(conf_level, bounds.loc[(bounds["Model"]==classifier_labels[i]) & (bounds["Sensitivity parameter"] == param )]["SE_ub"].values, bounds.loc[(bounds["Model"]==classifier_labels[i]) & (bounds["Sensitivity parameter"] == param)]["SE_lb"].values, alpha=0.6, label=legend_labels[i] + ", $\Gamma_M$ = "+ str(param), color=colors[i]) 
    ax3.set_xlabel("Confounding level \u03A6",fontsize="30")
    ax3.set_ylabel("Spurious effect",fontsize="30")

    if path != None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_bounds_over_confounding_continuous(conf_level, bounds, path=None):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(40, 8))
    
    sns.set(style="whitegrid", font_scale=1.5)
    
    palette = sns.color_palette("pastel")   
    colors = [palette[7], palette[4], palette[1]]

    legend_labels = ["Standard", "Fair naive", "Fair robust (ours)"]
    classifier_labels = ["StandardClf", "FairClf_naive", "FairClf"]
    
    #ax1 -> DE
    ax1.plot(conf_level, np.array([0,0,0,0]), label="Optimal fairness", linestyle='dashed', color = "grey", linewidth=3)
    for i in range(3):
        for param in bounds["Sensitivity parameter"].unique():
            ax1.fill_between(conf_level, np.array(bounds.loc[(bounds["Model"]==classifier_labels[i]) & (bounds["Sensitivity parameter"] == param)]["DE_ub"].values, dtype  = float), np.array(bounds.loc[(bounds["Model"]==classifier_labels[i]) & (bounds["Sensitivity parameter"] == param)]["DE_lb"].values, dtype=float), alpha=0.7, label=legend_labels[i] + ", $\Gamma_M$ = "+ str(param), color=colors[i])        
    ax1.set_xlabel("Confounding level \u03A6",fontsize="30")
    ax1.set_ylabel("Direct effect",fontsize="30")

    ax1.legend(fontsize="20")

    #ax2 -> IE
    ax2.plot(conf_level, np.array([0,0,0,0]), label="Optimal fairness", linestyle='dashed', color = "grey", linewidth=3)
    for i in range(3):
        for param in bounds["Sensitivity parameter"].unique():
            ax2.fill_between(conf_level, np.array(bounds.loc[(bounds["Model"]==classifier_labels[i]) & (bounds["Sensitivity parameter"] == param)]["IE_ub"].values, dtype=float), np.array(bounds.loc[(bounds["Model"]==classifier_labels[i]) & (bounds["Sensitivity parameter"] == param)]["IE_lb"].values, dtype=float), alpha=0.7, label=legend_labels[i] + ", $\Gamma_M$ = "+ str(param), color=colors[i])    

    ax2.set_xlabel("Confounding level \u03A6",fontsize="30")
    ax2.set_ylabel("Indirect effect",fontsize="30")


    #ax3 -> SE
    ax3.plot(conf_level, np.array([0,0,0,0]), label="Optimal fairness", linestyle='dashed', color = "grey", linewidth=3)
    for i in range(3):
        for param in bounds["Sensitivity parameter"]:
            ax3.fill_between(conf_level, np.array(bounds.loc[(bounds["Model"]==classifier_labels[i]) & (bounds["Sensitivity parameter"] == param )]["SE_ub"].values, dtype=float), np.array(bounds.loc[(bounds["Model"]==classifier_labels[i]) & (bounds["Sensitivity parameter"] == param)]["SE_lb"].values, dtype=float), alpha=0.7, label=legend_labels[i] + ", $\Gamma_M$ = "+ str(param), color=colors[i])    
    ax3.set_xlabel("Confounding level \u03A6",fontsize="30")
    ax3.set_ylabel("Spurious effect",fontsize="30")


    if path != None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.show()




def plot_bounds_data(conf_level, oracle, naive, bounds, tv, path=None):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(40, 8))
    ax2.tick_params('y', labelleft=True)
    ax3.tick_params('y', labelleft=True)

    sns.set(style="whitegrid", font_scale=1.5)

    palette = sns.color_palette("pastel")
    colors = [palette[3],palette[6], palette[4], palette[8], palette[2], palette[0]]

    sens_params = bounds["Sensitivity parameter"].unique()
    
    #ax1 -> DE
    sns.lineplot(x=conf_level, y=oracle["DE"].values, label="Oracle unfairness", color="darkred", linewidth=3, ax = ax1)
    sns.lineplot(x=conf_level, y=tv["TV"].values, label="Total variation",color="grey", linewidth=3, ax = ax1)
    sns.lineplot(x=conf_level, y= naive["DE"].values, label="Naive fairness",color="darkblue", linewidth=3, ax = ax1)   
    ax1.plot(conf_level, np.array([0,0,0,0]), linestyle='dashed', color = "grey")
    i=0
    for param in sens_params:
        ax1.fill_between(conf_level, bounds.loc[bounds["Sensitivity parameter"] == param , "DE_ub"], bounds.loc[bounds["Sensitivity parameter"] == param, "DE_lb"], alpha=0.4, label= "$\Gamma_M$ = " + str(param), color=colors[i])    
        i = i+1   
    ax1.set_xlabel("Confounding level \u03A6",fontsize="30")
    ax1.set_ylabel("Direct effect",fontsize="30")
    ax1.legend(fontsize="20")

    #ax2 -> IE
    sns.lineplot(x=conf_level, y=oracle["IE"].values, color="darkred", linewidth=3, ax = ax2)
    sns.lineplot(x=conf_level, y=tv["TV"].values,color="grey", linewidth=3, ax = ax2)
    sns.lineplot(x=conf_level, y= naive["IE"].values,color="darkblue", linewidth=3, ax = ax2)  
    ax2.plot(conf_level, np.array([0,0,0,0]), linestyle='dashed', color = "grey")
    i=0
    for param in sens_params:
        ax2.fill_between(conf_level, bounds.loc[bounds["Sensitivity parameter"] == param, "IE_ub"], bounds.loc[bounds["Sensitivity parameter"] == param, "IE_lb"], alpha=0.4, label= "$\Gamma_M$ = " + str(param), color=colors[i])    
        i = i+1  
    ax2.set_xlabel("Confounding level \u03A6",fontsize="30")
    ax2.set_ylabel("Indirect effect",fontsize="30")

    #ax3 -> SE
    sns.lineplot(x=conf_level, y=oracle["SE"].values, color="darkred", linewidth=3, ax = ax3)
    sns.lineplot(x=conf_level, y=tv["TV"].values,color="grey", linewidth=3, ax = ax3)
    sns.lineplot(x=conf_level, y= naive["SE"].values,color="darkblue", linewidth=3, ax = ax3)  
    ax3.plot(conf_level, np.array([0,0,0,0]), linestyle='dashed', color = "grey")
    i=0
    for param in sens_params:
        ax3.fill_between(conf_level, bounds.loc[bounds["Sensitivity parameter"] == param]["SE_ub"], bounds.loc[bounds["Sensitivity parameter"] == param]["SE_lb"], alpha=0.4, label= "$\Gamma_M$ = " + str(param), color=colors[i])    
        i = i+1     
    ax3.set_xlabel("Confounding level \u03A6",fontsize="30")
    ax3.set_ylabel("Spurious effect",fontsize="30")

    if path != None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.show()



def plot_performance_over_sensitivity_param(df, path=None):

    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(6,4))
    sns.lineplot(data=df, x="sensitivity parameter", y="mean", label="Test error", color="darkblue", linewidth=2)
    plt.fill_between(df["sensitivity parameter"], df["std_minus"], df["std_plus"], alpha=0.5)

    plt.xlim([1, 4]) 
    plt.ylim([-20, 20]) 
    plt.xlabel("Sensitivity parameter "r'$\Gamma$' " (training)")
    plt.ylabel(r'$\Delta$'" MSE in %")
    plt.legend()
    plt.tight_layout()

    if path != None:
        plt.savefig(path)
    plt.show()



def plot_sentence_distribution(pred_standard, pred_fair, path=None):
    seed(1)
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 6), sharey=True, sharex=True)

    sns.histplot(data=pred_standard, x='Sentence length', kde=True, hue = "Race", linewidth=2, ax=ax1, fill=True, alpha = 0.6, palette="pastel")
    sns.histplot(data=pred_fair, x='Sentence length', kde=True, hue = "Race", linewidth=2, ax=ax2, fill=True, alpha = 0.6, palette="pastel")

    fig.legend(fontsize="20")
    
    if path != None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.show()