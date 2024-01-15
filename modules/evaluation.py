import numpy as np
import pandas as pd
import torch
import pickle

from modules.predictors_modules import StandardClf, FairClf, FairClf_naive, Density_estimator
from modules import functions

from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser

from modules.helpers import seed

seed(1)



#------------------------------------------------Fairness calculation-------------------------------------------

def calculate_bounds(label_col, sensitive_attributes, sensitivity_parameter_list, checkpoints, experiment_list):

    bounds = pd.DataFrame(columns = ["Sensitivity parameter", "Model", "Experiment", "DE_ub", "DE_lb", "IE_ub", "IE_lb", "SE_ub","SE_lb"])
    oracle = pd.DataFrame(columns = ["Experiment", "DE", "IE", "SE"])

    for experiment in experiment_list:

        data_location = "../data/simulator/"+ experiment + "_full_dataframe"
        with open(data_location, "rb") as input:
            oracle_data = pickle.load(input)

        full_data = oracle_data.copy()
        full_data = full_data.drop(["USE", "UIE", "UDE"],axis=1)

        test_data = full_data.iloc[int(0.8*full_data.shape[0]):,:]

        checkpointpath_density = "../models/density_estimator_" + experiment + "/density_estimator_" + experiment + "_checkpoints.ckpt"
        density_estimator = Density_estimator.load_from_checkpoint(checkpointpath_density)

        
        for sensitivity_parameter in sensitivity_parameter_list:
        
            # StandardClf
            standard = StandardClf.load_from_checkpoint("../models/"  + "StandardClf_"+ experiment + "/StandardClf_"+ experiment + "_checkpoints.ckpt")
            ctf_effects_standard = functions.Ctf_effects_binary(network=standard, full_data = test_data, task = "binary", sensitivity_param=sensitivity_parameter, density_estimator=density_estimator)

            y_ind = 1.0
            ub_DE_y, lb_DE_y = ctf_effects_standard.DE_binary(sensitive_attributes[0], sensitive_attributes[1], y_ind)
            ub_IE_y, lb_IE_y = ctf_effects_standard.IE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)
            ub_SE_y, lb_SE_y = ctf_effects_standard.SE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)

            bounds_standard_df = pd.DataFrame(columns = ["Sensitivity parameter", "Model", "Experiment", "DE_ub", "DE_lb", "IE_ub", "IE_lb", "SE_ub","SE_lb"])
            bounds_standard_df = pd.concat([bounds_standard_df, pd.DataFrame({"Sensitivity parameter": sensitivity_parameter, "Model": "StandardClf", "Experiment": experiment, "DE_ub": [ub_DE_y], "DE_lb": [lb_DE_y], "IE_ub": [ub_IE_y], "IE_lb": [lb_IE_y], "SE_ub": [ub_SE_y],"SE_lb": [lb_SE_y]})])
            bounds = pd.concat([bounds, bounds_standard_df], ignore_index= True)


            # FairClf_naive
            fair_naive = FairClf_naive.load_from_checkpoint("../models/"  + "FairClf_naive_"+ experiment + "/FairClf_naive_"+ experiment + checkpoints)
            ctf_effects_naive = functions.Ctf_effects_binary(network=fair_naive, full_data = test_data, task = "binary", sensitivity_param=sensitivity_parameter, density_estimator=density_estimator)
    
            y_ind = 1.0
            ub_DE_y, lb_DE_y = ctf_effects_naive.DE_binary(sensitive_attributes[0], sensitive_attributes[1], y_ind)
            ub_IE_y, lb_IE_y = ctf_effects_naive.IE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)
            ub_SE_y, lb_SE_y = ctf_effects_naive.SE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)

            bounds_naive_df = pd.DataFrame(columns = ["Sensitivity parameter", "Model", "Experiment", "DE_ub", "DE_lb", "IE_ub", "IE_lb", "SE_ub","SE_lb"])
            bounds_naive_df = pd.concat([bounds_naive_df, pd.DataFrame({"Sensitivity parameter": sensitivity_parameter, "Model": "FairClf_naive", "Experiment": experiment, "DE_ub": [ub_DE_y], "DE_lb": [lb_DE_y], "IE_ub": [ub_IE_y], "IE_lb": [lb_IE_y], "SE_ub": [ub_SE_y],"SE_lb": [lb_SE_y]})])       
            bounds = pd.concat([bounds, bounds_naive_df], ignore_index= True)


            # FairClf
            fair = FairClf.load_from_checkpoint("../models/"  + "FairClf_"+ experiment + "/FairClf_"+ experiment + checkpoints)
            ctf_effects_fair = functions.Ctf_effects_binary(network=fair, full_data = test_data, task = "binary", sensitivity_param=sensitivity_parameter, density_estimator=density_estimator)

            y_ind = 1.0
            ub_DE_y, lb_DE_y = ctf_effects_fair.DE_binary(sensitive_attributes[0], sensitive_attributes[1], y_ind)
            ub_IE_y, lb_IE_y = ctf_effects_fair.IE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)
            ub_SE_y, lb_SE_y = ctf_effects_fair.SE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)

            bounds_fair_df = pd.DataFrame(columns = ["Sensitivity parameter", "Model", "Experiment", "DE_ub", "DE_lb", "IE_ub", "IE_lb", "SE_ub","SE_lb"])
            bounds_fair_df = pd.concat([bounds_fair_df, pd.DataFrame({"Sensitivity parameter": sensitivity_parameter, "Model": "FairClf", "Experiment": experiment, "DE_ub": [ub_DE_y], "DE_lb": [lb_DE_y], "IE_ub": [ub_IE_y], "IE_lb": [lb_IE_y], "SE_ub": [ub_SE_y],"SE_lb": [lb_SE_y]})])       
            bounds = pd.concat([bounds, bounds_fair_df], ignore_index= True)

        # Oracle 
        nodes, edges = graph_file_parser("../data/simulator/"+ experiment +".yml")
        g = Graph(nodes=nodes, edges=edges)

        oracle_data = g.sample(100000)[["A", "Z", "M", "Y"]]
        oracle_data["M"] = np.round(oracle_data["M"])
        oracle_data["Y"] = np.round(oracle_data["Y"])    
        do_s0 = g.do(size=100000, interventions={'A': sensitive_attributes[0]})
        do_s0["M"] = np.round(do_s0["M"])
        do_s0["Y"] = np.round(do_s0["Y"])   
        do_s1 = g.do(size=100000, interventions={'A': sensitive_attributes[1]})
        do_s1["M"] = np.round(do_s1["M"])
        do_s1["Y"] = np.round(do_s1["Y"])   
        do_s0_m0 = g.do(size=100000, interventions={'A': sensitive_attributes[0], 'M': 0.0})
        do_s0_m0["M"] = np.round(do_s0_m0["M"])
        do_s0_m0["Y"] = np.round(do_s0_m0["Y"])   
        do_s1_m0 = g.do(size=100000, interventions={'A': sensitive_attributes[1], 'M': 0.0})
        do_s1_m0["M"] = np.round(do_s1_m0["M"])
        do_s1_m0["Y"] = np.round(do_s1_m0["Y"])   
        do_s0_m1 = g.do(size=100000, interventions={'A': sensitive_attributes[0], 'M': 1.0})
        do_s0_m1["M"] = np.round(do_s0_m1["M"])
        do_s0_m1["Y"] = np.round(do_s0_m1["Y"])   
        do_s1_m1 = g.do(size=100000, interventions={'A': sensitive_attributes[1], 'M': 1.0})
        do_s1_m1["M"] = np.round(do_s1_m1["M"])
        do_s1_m1["Y"] = np.round(do_s1_m1["Y"])   

        p_y_s0 = oracle_data.loc[oracle_data["A"] == sensitive_attributes[0], "Y"].value_counts(normalize=True)[sorted(oracle_data["Y"].unique())]
        p_y_s0 = p_y_s0.reset_index()["proportion"]
        p_y_do_s0_given_s1 = do_s0.loc[oracle_data["A"] == sensitive_attributes[1], "Y"].value_counts(normalize=True)[sorted(oracle_data["Y"].unique())]
        p_y_do_s0_given_s1= p_y_do_s0_given_s1.reset_index()["proportion"]
        p_y_s1 = oracle_data.loc[oracle_data["A"] == sensitive_attributes[1], "Y"].value_counts(normalize=True)[sorted(oracle_data["Y"].unique())]
        p_y_s1 = p_y_s1.reset_index()["proportion"]
        p_y_do_s1_given_s0 = do_s1.loc[oracle_data["A"] == sensitive_attributes[0], "Y"].value_counts(normalize=True)[sorted(oracle_data["Y"].unique())]
        p_y_do_s1_given_s0 = p_y_do_s1_given_s0.reset_index()["proportion"]

        m_do_s0_given_s1 = do_s0.loc[oracle_data["A"] == sensitive_attributes[1]].reset_index()
        y_do_s0_m0 = do_s0_m0[oracle_data["A"] == sensitive_attributes[1]].reset_index()
        y_do_s0_m0 = y_do_s0_m0[y_do_s0_m0["M"]==m_do_s0_given_s1["M"]].reset_index()
        y_do_s0_m1 = do_s0_m1[oracle_data["A"] == sensitive_attributes[1]].reset_index()
        y_do_s0_m1 = y_do_s0_m1[y_do_s0_m1["M"]==m_do_s0_given_s1["M"]].reset_index()

        m_do_s0_given_s0 = do_s0.loc[oracle_data["A"] == sensitive_attributes[0]].reset_index()
        y_do_s1_m0 = do_s1_m0[oracle_data["A"] == sensitive_attributes[0]].reset_index()
        y_do_s1_m0 = y_do_s1_m0[y_do_s1_m0["M"]==m_do_s0_given_s0["M"]].reset_index()
        y_do_s1_m1 = do_s1_m1[oracle_data["A"] == sensitive_attributes[0]].reset_index()
        y_do_s1_m1 = y_do_s1_m1[y_do_s1_m1["M"]==m_do_s0_given_s0["M"]].reset_index()
        p_y_do_s1_do_m_do_s0_given_s0 = pd.concat([y_do_s1_m0, y_do_s1_m1], axis=0)
        p_y_do_s1_do_m_do_s0_given_s0 = p_y_do_s1_do_m_do_s0_given_s0["Y"].value_counts(normalize=True)[sorted(p_y_do_s1_do_m_do_s0_given_s0["Y"].unique())]
        p_y_do_s1_do_m_do_s0_given_s0 = p_y_do_s1_do_m_do_s0_given_s0.reset_index()["proportion"]  
  
        de = (p_y_do_s1_do_m_do_s0_given_s0 - p_y_s0)[1]
        ie = (p_y_do_s1_do_m_do_s0_given_s0 - p_y_do_s1_given_s0)[1]
        se = (p_y_do_s1_given_s0 - p_y_s1)[1]

        fairness_oracle = pd.DataFrame({"Experiment": experiment, "DE": [de],"IE": [ie],"SE": [se]})

        oracle = pd.concat([oracle, fairness_oracle], ignore_index= True)

    return bounds, oracle


def calculate_bounds_continuous(label_col, sensitive_attributes, sensitivity_parameter_list, checkpoints, experiment_list):

    bounds = pd.DataFrame(columns = ["Sensitivity parameter", "Model", "Experiment", "DE_ub", "DE_lb", "IE_ub", "IE_lb", "SE_ub","SE_lb"])
    oracle = pd.DataFrame(columns = ["Experiment", "DE", "IE", "SE"])

    for experiment in experiment_list:

        data_location = "../data/simulator/"+ experiment + "_full_dataframe"
        with open(data_location, "rb") as input:
            oracle_data = pickle.load(input)

        full_data = oracle_data.copy()
        full_data = full_data.drop(["USE", "UIE", "UDE"],axis=1)

        test_data = full_data.iloc[int(0.8*full_data.shape[0]):,:]

        checkpointpath_density = "../models/density_estimator_" + experiment + "/density_estimator_" + experiment + "_checkpoints.ckpt"
        density_estimator = Density_estimator.load_from_checkpoint(checkpointpath_density)
        checkpointpath_propensity = "../models/density_estimator_A_" + experiment + "/density_estimator_A_" + experiment + "_checkpoints.ckpt"
        propensity_model = Density_estimator.load_from_checkpoint(checkpointpath_propensity)

        
        for sensitivity_parameter in sensitivity_parameter_list:
        
            # StandardClf
            standard = StandardClf.load_from_checkpoint("../models/"  + "StandardClf_"+ experiment + "/StandardClf_"+ experiment + "_checkpoints.ckpt")
            ctf_effects_standard = functions.Ctf_effects_binary(network=standard, full_data = test_data, task = "continuous", sensitivity_param=sensitivity_parameter, density_estimator=density_estimator, propensity_model=propensity_model)

            y_ind = 1.0
            ub_DE_y, lb_DE_y = ctf_effects_standard.DE_binary(sensitive_attributes[0], sensitive_attributes[1], y_ind)
            ub_IE_y, lb_IE_y = ctf_effects_standard.IE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)
            ub_SE_y, lb_SE_y = ctf_effects_standard.SE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)

            bounds_standard_df = pd.DataFrame(columns = ["Sensitivity parameter", "Model", "Experiment", "DE_ub", "DE_lb", "IE_ub", "IE_lb", "SE_ub","SE_lb"])
            bounds_standard_df = pd.concat([bounds_standard_df, pd.DataFrame({"Sensitivity parameter": sensitivity_parameter, "Model": "StandardClf", "Experiment": experiment, "DE_ub": [ub_DE_y], "DE_lb": [lb_DE_y], "IE_ub": [ub_IE_y], "IE_lb": [lb_IE_y], "SE_ub": [ub_SE_y],"SE_lb": [lb_SE_y]})])
            bounds = pd.concat([bounds, bounds_standard_df], ignore_index= True)


            # FairClf_naive
            fair_naive = FairClf_naive.load_from_checkpoint("../models/"  + "FairClf_naive_lambda"+ experiment + "/FairClf_naive_lambda"+ experiment + checkpoints)
            ctf_effects_naive = functions.Ctf_effects_binary(network=fair_naive, full_data = test_data, task = "continuous", sensitivity_param=sensitivity_parameter, density_estimator=density_estimator, propensity_model=propensity_model)
    
            y_ind = 1.0
            ub_DE_y, lb_DE_y = ctf_effects_naive.DE_binary(sensitive_attributes[0], sensitive_attributes[1], y_ind)
            ub_IE_y, lb_IE_y = ctf_effects_naive.IE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)
            ub_SE_y, lb_SE_y = ctf_effects_naive.SE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)

            bounds_naive_df = pd.DataFrame(columns = ["Sensitivity parameter", "Model", "Experiment", "DE_ub", "DE_lb", "IE_ub", "IE_lb", "SE_ub","SE_lb"])
            bounds_naive_df = pd.concat([bounds_naive_df, pd.DataFrame({"Sensitivity parameter": sensitivity_parameter, "Model": "FairClf_naive", "Experiment": experiment, "DE_ub": [ub_DE_y], "DE_lb": [lb_DE_y], "IE_ub": [ub_IE_y], "IE_lb": [lb_IE_y], "SE_ub": [ub_SE_y],"SE_lb": [lb_SE_y]})])       
            bounds = pd.concat([bounds, bounds_naive_df], ignore_index= True)


            # FairClf
            fair = FairClf.load_from_checkpoint("../models/"  + "FairClf_lambda_"+ experiment + "/FairClf_lambda_"+ experiment + checkpoints)
            ctf_effects_fair = functions.Ctf_effects_binary(network=fair, full_data = test_data, task = "continuous", sensitivity_param=sensitivity_parameter, density_estimator=density_estimator, propensity_model=propensity_model)

            y_ind = 1.0
            ub_DE_y, lb_DE_y = ctf_effects_fair.DE_binary(sensitive_attributes[0], sensitive_attributes[1], y_ind)
            ub_IE_y, lb_IE_y = ctf_effects_fair.IE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)
            ub_SE_y, lb_SE_y = ctf_effects_fair.SE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)

            bounds_fair_df = pd.DataFrame(columns = ["Sensitivity parameter", "Model", "Experiment", "DE_ub", "DE_lb", "IE_ub", "IE_lb", "SE_ub","SE_lb"])
            bounds_fair_df = pd.concat([bounds_fair_df, pd.DataFrame({"Sensitivity parameter": sensitivity_parameter, "Model": "FairClf", "Experiment": experiment, "DE_ub": [ub_DE_y], "DE_lb": [lb_DE_y], "IE_ub": [ub_IE_y], "IE_lb": [lb_IE_y], "SE_ub": [ub_SE_y],"SE_lb": [lb_SE_y]})])       
            bounds = pd.concat([bounds, bounds_fair_df], ignore_index= True)

    return bounds, oracle



def calculate_bounds_multiclass(label_col, sensitive_attributes, sensitivity_parameter_list, checkpoints, experiment_list):

    bounds = pd.DataFrame(columns = ["Sensitivity parameter", "Model", "Experiment", "DE_ub", "DE_lb", "IE_ub", "IE_lb", "SE_ub","SE_lb"])
    oracle = pd.DataFrame(columns = ["Experiment", "DE", "IE", "SE"])

    for experiment in experiment_list:

        data_location = "../data/simulator/"+ experiment + "_full_dataframe"
        with open(data_location, "rb") as input:
            oracle_data = pickle.load(input)

        full_data = oracle_data.copy()
        full_data = full_data.drop(["USE", "UIE", "UDE"],axis=1)

        test_data = full_data.iloc[int(0.8*full_data.shape[0]):,:]

        checkpointpath_density = "../models/density_estimator_" + experiment + "/density_estimator_" + experiment + "_checkpoints.ckpt"
        density_estimator = Density_estimator.load_from_checkpoint(checkpointpath_density)

        
        for sensitivity_parameter in sensitivity_parameter_list:
        
            # StandardClf
            standard = StandardClf.load_from_checkpoint("../models/"  + "StandardClf_"+ experiment + "/StandardClf_"+ experiment + "_checkpoints.ckpt")
            ctf_effects_standard = functions.Ctf_effects_binary(network=standard, full_data = test_data, task = "multiclass", sensitivity_param=sensitivity_parameter, density_estimator=density_estimator)
            
            bounds_standard = np.empty((0,6))
            for y_ind in np.unique(full_data["Y"]):
                ub_DE_y, lb_DE_y = ctf_effects_standard.DE_binary(sensitive_attributes[0], sensitive_attributes[1], y_ind)
                ub_IE_y, lb_IE_y = ctf_effects_standard.IE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)
                ub_SE_y, lb_SE_y = ctf_effects_standard.SE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)
                bounds_standard = np.concatenate((bounds_standard, np.expand_dims([ub_DE_y, lb_DE_y, ub_IE_y, lb_IE_y, ub_SE_y, lb_SE_y], axis = 0)), 0)  
            bounds_standard = bounds_standard * np.repeat(np.expand_dims(ctf_effects_standard.rel_freq_y.sort_values(by = "Y")["proportion"], axis =1), 6, axis = 1)
            bounds_standard = np.sum(bounds_standard, axis = 0)
            
            bounds_standard_df = pd.DataFrame(columns = ["Sensitivity parameter", "Model", "Experiment", "DE_ub", "DE_lb", "IE_ub", "IE_lb", "SE_ub","SE_lb"])
            bounds_standard_df = pd.concat([bounds_standard_df, pd.DataFrame({"Sensitivity parameter": sensitivity_parameter, "Model": "StandardClf", "Experiment": experiment, "DE_ub": [bounds_standard[0]], "DE_lb": [bounds_standard[1]], "IE_ub": [bounds_standard[2]], "IE_lb": [bounds_standard[3]], "SE_ub": [bounds_standard[4]],"SE_lb": [bounds_standard[5]]})])
            bounds = pd.concat([bounds, bounds_standard_df], ignore_index= True)


            # FairClf_naive
            fair_naive = FairClf_naive.load_from_checkpoint("../models/"  + "FairClf_naive_"+ experiment + "/FairClf_naive_"+ experiment + checkpoints)
            ctf_effects_naive = functions.Ctf_effects_binary(network=fair_naive, full_data = test_data, task = "multiclass", sensitivity_param=sensitivity_parameter, density_estimator=density_estimator)
    
            bounds_naive = np.empty((0,6))
            for y_ind in np.unique(full_data["Y"]):
                ub_DE_y, lb_DE_y = ctf_effects_naive.DE_binary(sensitive_attributes[0], sensitive_attributes[1], y_ind)
                ub_IE_y, lb_IE_y = ctf_effects_naive.IE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)
                ub_SE_y, lb_SE_y = ctf_effects_naive.SE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)
                bounds_naive = np.concatenate((bounds_naive, np.expand_dims([ub_DE_y, lb_DE_y, ub_IE_y, lb_IE_y, ub_SE_y, lb_SE_y], axis = 0)), 0)  
            bounds_naive = bounds_naive * np.repeat(np.expand_dims(ctf_effects_naive.rel_freq_y.sort_values(by = "Y")["proportion"].values, axis =1), 6, axis = 1)
            bounds_naive = np.sum(bounds_naive, axis = 0)

            bounds_naive_df = pd.DataFrame(columns = ["Sensitivity parameter", "Model", "Experiment", "DE_ub", "DE_lb", "IE_ub", "IE_lb", "SE_ub","SE_lb"])
            bounds_naive_df = pd.concat([bounds_naive_df, pd.DataFrame({"Sensitivity parameter": sensitivity_parameter, "Model": "FairClf_naive", "Experiment": experiment, "DE_ub": [bounds_naive[0]], "DE_lb": [bounds_naive[1]], "IE_ub": [bounds_naive[2]], "IE_lb": [bounds_naive[3]], "SE_ub": [bounds_naive[4]],"SE_lb": [bounds_naive[5]]})])       
            bounds = pd.concat([bounds, bounds_naive_df], ignore_index= True)


            # FairClf
            fair = FairClf.load_from_checkpoint("../models/"  + "FairClf_"+ experiment + "/FairClf_"+ experiment + checkpoints)
            ctf_effects_fair = functions.Ctf_effects_binary(network=fair, full_data = test_data, task = "multiclass", sensitivity_param=sensitivity_parameter, density_estimator=density_estimator)

            bounds_fair = np.empty((0,6))
            for y_ind in np.unique(full_data["Y"]):
                ub_DE_y, lb_DE_y = ctf_effects_fair.DE_binary(sensitive_attributes[0], sensitive_attributes[1], y_ind)
                ub_IE_y, lb_IE_y = ctf_effects_fair.IE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)
                ub_SE_y, lb_SE_y = ctf_effects_fair.SE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)
                bounds_fair = np.concatenate((bounds_fair, np.expand_dims([ub_DE_y, lb_DE_y, ub_IE_y, lb_IE_y, ub_SE_y, lb_SE_y], axis = 0)), 0)  
            bounds_fair = bounds_fair * np.repeat(np.expand_dims(ctf_effects_fair.rel_freq_y.sort_values(by = "Y")["proportion"].values, axis =1), 6, axis = 1)
            bounds_fair = np.sum(bounds_fair, axis = 0)

            bounds_fair_df = pd.DataFrame(columns = ["Sensitivity parameter", "Model", "Experiment", "DE_ub", "DE_lb", "IE_ub", "IE_lb", "SE_ub","SE_lb"])
            bounds_fair_df = pd.concat([bounds_fair_df, pd.DataFrame({"Sensitivity parameter": sensitivity_parameter, "Model": "FairClf", "Experiment": experiment, "DE_ub": [bounds_fair[0]], "DE_lb": [bounds_fair[1]], "IE_ub": [bounds_fair[2]], "IE_lb": [bounds_fair[3]], "SE_ub": [bounds_fair[4]],"SE_lb": [bounds_fair[5]]})])       
            bounds = pd.concat([bounds, bounds_fair_df], ignore_index= True)

        # Oracle 
        nodes, edges = graph_file_parser("../data/simulator/"+ experiment +".yml")
        g = Graph(nodes=nodes, edges=edges)

        oracle_data = g.sample(100000)[["A", "Z1", "Z2","Z3","Z4","M", "Y"]]
        oracle_data["M"] = np.where(oracle_data["M"] < 5, oracle_data["M"], 5)
        oracle_data["Y"] = np.where(oracle_data["Y"] < 6, oracle_data["Y"], 6)    
        rel_freq_y = oracle_data["Y"].value_counts(normalize=True)[sorted(oracle_data["Y"].unique())]
        do_s0 = g.do(size=100000, interventions={'A': sensitive_attributes[0]})
        do_s0["M"] = np.where(do_s0["M"] < 5, do_s0["M"], 5)
        do_s0["Y"] = np.where(do_s0["Y"] < 6, do_s0["Y"], 6)  
        do_s1 = g.do(size=100000, interventions={'A': sensitive_attributes[1]})
        do_s1["M"] = np.where(do_s1["M"] < 5, do_s1["M"], 5)
        do_s1["Y"] = np.where(do_s1["Y"] < 6, do_s1["Y"], 6)  
        do_s0_m0 = g.do(size=100000, interventions={'A': sensitive_attributes[0], 'M': 0.0})
        do_s0_m0["M"] = np.where(do_s0_m0["M"] < 5, do_s0_m0["M"], 5)
        do_s0_m0["Y"] = np.where(do_s0_m0["Y"] < 6, do_s0_m0["Y"], 6)  
        do_s1_m0 = g.do(size=100000, interventions={'A': sensitive_attributes[1], 'M': 0.0})
        do_s1_m0["M"] = np.where(do_s1_m0["M"] < 5, do_s1_m0["M"], 5)
        do_s1_m0["Y"] = np.where(do_s1_m0["Y"] < 6, do_s1_m0["Y"], 6)  
        do_s0_m1 = g.do(size=100000, interventions={'A': sensitive_attributes[0], 'M': 1.0})
        do_s0_m1["M"] = np.where(do_s0_m1["M"] < 5, do_s0_m1["M"], 5)
        do_s0_m1["Y"] = np.where(do_s0_m1["Y"] < 6, do_s0_m1["Y"], 6)  
        do_s1_m1 = g.do(size=100000, interventions={'A': sensitive_attributes[1], 'M': 1.0})
        do_s1_m1["M"] = np.where(do_s1_m1["M"] < 5, do_s1_m1["M"], 5)
        do_s1_m1["Y"] = np.where(do_s1_m1["Y"] < 6, do_s1_m1["Y"], 6)  
        do_s0_m2 = g.do(size=100000, interventions={'A': sensitive_attributes[0], 'M': 2.0})
        do_s0_m2["M"] = np.where(do_s0_m2["M"] < 5, do_s0_m2["M"], 5)
        do_s0_m2["Y"] = np.where(do_s0_m2["Y"] < 6, do_s0_m2["Y"], 6)  
        do_s1_m2 = g.do(size=100000, interventions={'A': sensitive_attributes[1], 'M': 2.0})
        do_s1_m2["M"] = np.where(do_s1_m2["M"] < 5, do_s1_m2["M"], 5)
        do_s1_m2["Y"] = np.where(do_s1_m2["Y"] < 6, do_s1_m2["Y"], 6)  
        do_s0_m3 = g.do(size=100000, interventions={'A': sensitive_attributes[0], 'M': 3.0})
        do_s0_m3["M"] = np.where(do_s0_m3["M"] < 5, do_s0_m3["M"], 5)
        do_s0_m3["Y"] = np.where(do_s0_m3["Y"] < 6, do_s0_m3["Y"], 6)  
        do_s1_m3 = g.do(size=100000, interventions={'A': sensitive_attributes[1], 'M': 3.0})
        do_s1_m3["M"] = np.where(do_s1_m3["M"] < 5, do_s1_m3["M"], 5)
        do_s1_m3["Y"] = np.where(do_s1_m3["Y"] < 6, do_s1_m3["Y"], 6) 
        do_s0_m4 = g.do(size=100000, interventions={'A': sensitive_attributes[0], 'M': 4.0})
        do_s0_m4["M"] = np.where(do_s0_m4["M"] < 5, do_s0_m4["M"], 5)
        do_s0_m4["Y"] = np.where(do_s0_m4["Y"] < 6, do_s0_m4["Y"], 6)  
        do_s1_m4 = g.do(size=100000, interventions={'A': sensitive_attributes[1], 'M': 4.0})
        do_s1_m4["M"] = np.where(do_s1_m4["M"] < 5, do_s1_m4["M"], 5)
        do_s1_m4["Y"] = np.where(do_s1_m4["Y"] < 6, do_s1_m4["Y"], 6) 
        do_s0_m5 = g.do(size=100000, interventions={'A': sensitive_attributes[0], 'M': 5.0})
        do_s0_m5["M"] = np.where(do_s0_m5["M"] < 5, do_s0_m5["M"], 5)
        do_s0_m5["Y"] = np.where(do_s0_m5["Y"] < 6, do_s0_m5["Y"], 6)  
        do_s1_m5 = g.do(size=100000, interventions={'A': sensitive_attributes[1], 'M': 5.0})
        do_s1_m5["M"] = np.where(do_s1_m5["M"] < 5, do_s1_m5["M"], 5)
        do_s1_m5["Y"] = np.where(do_s1_m5["Y"] < 6, do_s1_m5["Y"], 6) 

        p_y_s0 = oracle_data.loc[oracle_data["A"] == sensitive_attributes[0], "Y"].sort_values().value_counts(normalize=True)
        p_y_s0 = p_y_s0.reset_index()["proportion"]
        p_y_do_s0_given_s1 = do_s0.loc[oracle_data["A"] == sensitive_attributes[1], "Y"].sort_values().value_counts(normalize=True)
        p_y_do_s0_given_s1= p_y_do_s0_given_s1.reset_index()["proportion"]
        p_y_s1 = oracle_data.loc[oracle_data["A"] == sensitive_attributes[1], "Y"].sort_values().value_counts(normalize=True)
        p_y_s1 = p_y_s1.reset_index()["proportion"]
        p_y_do_s1_given_s0 = do_s1.loc[oracle_data["A"] == sensitive_attributes[0], "Y"].sort_values().value_counts(normalize=True)
        p_y_do_s1_given_s0 = p_y_do_s1_given_s0.reset_index()["proportion"]

        m_do_s0_given_s1 = do_s0.loc[oracle_data["A"] == sensitive_attributes[1]].reset_index()
        y_do_s0_m0 = do_s0_m0[oracle_data["A"] == sensitive_attributes[1]].reset_index()
        y_do_s0_m0 = y_do_s0_m0[y_do_s0_m0["M"]==m_do_s0_given_s1["M"]].reset_index()
        y_do_s0_m1 = do_s0_m1[oracle_data["A"] == sensitive_attributes[1]].reset_index()
        y_do_s0_m1 = y_do_s0_m1[y_do_s0_m1["M"]==m_do_s0_given_s1["M"]].reset_index()
        y_do_s0_m2 = do_s0_m2[oracle_data["A"] == sensitive_attributes[1]].reset_index()
        y_do_s0_m2 = y_do_s0_m2[y_do_s0_m2["M"]==m_do_s0_given_s1["M"]].reset_index()
        y_do_s0_m3 = do_s0_m3[oracle_data["A"] == sensitive_attributes[1]].reset_index()
        y_do_s0_m3 = y_do_s0_m3[y_do_s0_m3["M"]==m_do_s0_given_s1["M"]].reset_index()
        y_do_s0_m4 = do_s0_m4[oracle_data["A"] == sensitive_attributes[1]].reset_index()
        y_do_s0_m4 = y_do_s0_m4[y_do_s0_m4["M"]==m_do_s0_given_s1["M"]].reset_index()
        y_do_s0_m5 = do_s0_m5[oracle_data["A"] == sensitive_attributes[1]].reset_index()
        y_do_s0_m5 = y_do_s0_m5[y_do_s0_m5["M"]==m_do_s0_given_s1["M"]].reset_index()

        m_do_s0_given_s0 = do_s0.loc[oracle_data["A"] == sensitive_attributes[0]].reset_index()
        y_do_s1_m0 = do_s1_m0[oracle_data["A"] == sensitive_attributes[0]].reset_index()
        y_do_s1_m0 = y_do_s1_m0[y_do_s1_m0["M"]==m_do_s0_given_s0["M"]].reset_index()
        y_do_s1_m1 = do_s1_m1[oracle_data["A"] == sensitive_attributes[0]].reset_index()
        y_do_s1_m1 = y_do_s1_m1[y_do_s1_m1["M"]==m_do_s0_given_s0["M"]].reset_index()
        y_do_s1_m2= do_s1_m2[oracle_data["A"] == sensitive_attributes[0]].reset_index()
        y_do_s1_m2 = y_do_s1_m2[y_do_s1_m2["M"]==m_do_s0_given_s0["M"]].reset_index()
        y_do_s1_m3 = do_s1_m3[oracle_data["A"] == sensitive_attributes[0]].reset_index()
        y_do_s1_m3 = y_do_s1_m3[y_do_s1_m3["M"]==m_do_s0_given_s0["M"]].reset_index()
        y_do_s1_m4 = do_s1_m4[oracle_data["A"] == sensitive_attributes[0]].reset_index()
        y_do_s1_m4 = y_do_s1_m4[y_do_s1_m4["M"]==m_do_s0_given_s0["M"]].reset_index()
        y_do_s1_m5 = do_s1_m5[oracle_data["A"] == sensitive_attributes[0]].reset_index()
        y_do_s1_m5 = y_do_s1_m5[y_do_s1_m5["M"]==m_do_s0_given_s0["M"]].reset_index()

        p_y_do_s1_do_m_do_s0_given_s0 = pd.concat([y_do_s1_m0, y_do_s1_m1, y_do_s1_m2, y_do_s1_m3, y_do_s1_m4, y_do_s1_m5], axis=0)
        p_y_do_s1_do_m_do_s0_given_s0 = p_y_do_s1_do_m_do_s0_given_s0["Y"].sort_values().value_counts(normalize=True)
        p_y_do_s1_do_m_do_s0_given_s0 = p_y_do_s1_do_m_do_s0_given_s0.reset_index()["proportion"]  

        de = (p_y_do_s1_do_m_do_s0_given_s0 - p_y_s0)[1]
        ie = (p_y_do_s1_do_m_do_s0_given_s0 - p_y_do_s1_given_s0)[1]
        se = (p_y_do_s1_given_s0 - p_y_s1)[1]

        fairness_oracle = pd.DataFrame({"Experiment": experiment, "DE": [de],"IE": [ie],"SE": [se]})

        oracle = pd.concat([oracle, fairness_oracle], ignore_index= True)

    return bounds, oracle





def caculate_data_fairness(sensitive_attributes, sensitivity_parameter, experiment_list, density_y = False):

    bounds = pd.DataFrame(columns = ["Sensitivity parameter","Experiment", "DE_ub", "DE_lb", "IE_ub", "IE_lb", "SE_ub","SE_lb"])
    oracle = pd.DataFrame(columns = ["Experiment", "DE", "IE", "SE"])
    naive = pd.DataFrame(columns = ["Experiment", "DE", "IE", "SE"])
    tv = pd.DataFrame(columns = ["Experiment", "TV"])
    label_col = "Y"

    for experiment in experiment_list:

        data_location = "../data/simulator/"+ experiment + "_full_dataframe"
        with open(data_location, "rb") as input:
            oracle_data = pickle.load(input)

        full_data = oracle_data.copy()
        full_data = full_data.drop(["USE", "UIE", "UDE"],axis=1)
        node_names = full_data.drop(["Y"],axis=1).columns.values
        n_classes = num_classes=len(np.unique(full_data[label_col]))
        test_data = full_data.iloc[int(0.8*full_data.shape[0]):,:]
        test_data_oracle = oracle_data.iloc[int(0.8*oracle_data.shape[0]):,:]
        covariates = test_data.iloc[:, test_data.columns != label_col].values
        covariates = torch.tensor(covariates, requires_grad=False).float()
        y = torch.tensor(test_data[label_col].values, requires_grad=False).float().unsqueeze(dim=1)

        checkpointpath_density = "../models/density_estimator_" + experiment + "/density_estimator_" + experiment + "_checkpoints.ckpt"
        density_estimator = Density_estimator.load_from_checkpoint(checkpointpath_density)

        total_var = (oracle_data.loc[oracle_data["A"]==sensitive_attributes[1], "Y"].value_counts(normalize=True)[sorted(oracle_data["Y"].unique())] - oracle_data.loc[oracle_data["A"]==sensitive_attributes[0], "Y"].value_counts(normalize=True)[sorted(oracle_data["Y"].unique())])[1]
        TV_df = pd.DataFrame({"Experiment": experiment, "TV": [total_var]})
        tv = pd.concat([tv, TV_df], ignore_index= True)

        # Oracle

        nodes, edges = graph_file_parser("../data/simulator/"+ experiment +".yml")
        g = Graph(nodes=nodes, edges=edges)

        oracle_data = g.sample(100000)[["A", "Z", "M", "Y"]]
        oracle_data["M"] = np.round(oracle_data["M"])
        oracle_data["Y"] = np.round(oracle_data["Y"])    
        rel_freq_y = oracle_data["Y"].value_counts(normalize=True)[sorted(oracle_data["Y"].unique())]
        do_s0 = g.do(size=100000, interventions={'A': sensitive_attributes[0]})
        do_s0["M"] = np.round(do_s0["M"])
        do_s0["Y"] = np.round(do_s0["Y"])   
        do_s1 = g.do(size=100000, interventions={'A': sensitive_attributes[1]})
        do_s1["M"] = np.round(do_s1["M"])
        do_s1["Y"] = np.round(do_s1["Y"])   
        do_s0_m0 = g.do(size=100000, interventions={'A': sensitive_attributes[0], 'M': 0.0})
        do_s0_m0["M"] = np.round(do_s0_m0["M"])
        do_s0_m0["Y"] = np.round(do_s0_m0["Y"])   
        do_s1_m0 = g.do(size=100000, interventions={'A': sensitive_attributes[1], 'M': 0.0})
        do_s1_m0["M"] = np.round(do_s1_m0["M"])
        do_s1_m0["Y"] = np.round(do_s1_m0["Y"])   
        do_s0_m1 = g.do(size=100000, interventions={'A': sensitive_attributes[0], 'M': 1.0})
        do_s0_m1["M"] = np.round(do_s0_m1["M"])
        do_s0_m1["Y"] = np.round(do_s0_m1["Y"])   
        do_s1_m1 = g.do(size=100000, interventions={'A': sensitive_attributes[1], 'M': 1.0})
        do_s1_m1["M"] = np.round(do_s1_m1["M"])
        do_s1_m1["Y"] = np.round(do_s1_m1["Y"])   

        p_y_s0 = oracle_data.loc[oracle_data["A"] == sensitive_attributes[0], "Y"].value_counts(normalize=True)[sorted(oracle_data["Y"].unique())]
        p_y_s0 = p_y_s0.reset_index()["proportion"]
        p_y_do_s0_given_s1 = do_s0.loc[oracle_data["A"] == sensitive_attributes[1], "Y"].value_counts(normalize=True)[sorted(oracle_data["Y"].unique())]
        p_y_do_s0_given_s1= p_y_do_s0_given_s1.reset_index()["proportion"]
        p_y_s1 = oracle_data.loc[oracle_data["A"] == sensitive_attributes[1], "Y"].value_counts(normalize=True)[sorted(oracle_data["Y"].unique())]
        p_y_s1 = p_y_s1.reset_index()["proportion"]
        p_y_do_s1_given_s0 = do_s1.loc[oracle_data["A"] == sensitive_attributes[0], "Y"].value_counts(normalize=True)[sorted(oracle_data["Y"].unique())]
        p_y_do_s1_given_s0 = p_y_do_s1_given_s0.reset_index()["proportion"]

        m_do_s0_given_s1 = do_s0.loc[oracle_data["A"] == sensitive_attributes[1]].reset_index()
        y_do_s0_m0 = do_s0_m0[oracle_data["A"] == sensitive_attributes[1]].reset_index()
        y_do_s0_m0 = y_do_s0_m0[y_do_s0_m0["M"]==m_do_s0_given_s1["M"]].reset_index()
        y_do_s0_m1 = do_s0_m1[oracle_data["A"] == sensitive_attributes[1]].reset_index()
        y_do_s0_m1 = y_do_s0_m1[y_do_s0_m1["M"]==m_do_s0_given_s1["M"]].reset_index()

        m_do_s0_given_s0 = do_s0.loc[oracle_data["A"] == sensitive_attributes[0]].reset_index()
        y_do_s1_m0 = do_s1_m0[oracle_data["A"] == sensitive_attributes[0]].reset_index()
        y_do_s1_m0 = y_do_s1_m0[y_do_s1_m0["M"]==m_do_s0_given_s0["M"]].reset_index()
        y_do_s1_m1 = do_s1_m1[oracle_data["A"] == sensitive_attributes[0]].reset_index()
        y_do_s1_m1 = y_do_s1_m1[y_do_s1_m1["M"]==m_do_s0_given_s0["M"]].reset_index()
        p_y_do_s1_do_m_do_s0_given_s0 = pd.concat([y_do_s1_m0, y_do_s1_m1], axis=0)
        p_y_do_s1_do_m_do_s0_given_s0 = p_y_do_s1_do_m_do_s0_given_s0["Y"].value_counts(normalize=True)[sorted(p_y_do_s1_do_m_do_s0_given_s0["Y"].unique())]
        p_y_do_s1_do_m_do_s0_given_s0 = p_y_do_s1_do_m_do_s0_given_s0.reset_index()["proportion"]  

        de = (p_y_do_s1_do_m_do_s0_given_s0 - p_y_s0)[1]
        ie = (p_y_do_s1_do_m_do_s0_given_s0 - p_y_do_s1_given_s0)[1]
        se = (p_y_do_s1_given_s0 - p_y_s1)[1]

        fairness_oracle = pd.DataFrame({"Experiment": experiment, "DE": [de],"IE": [ie],"SE": [se]})

        oracle = pd.concat([oracle, fairness_oracle], ignore_index= True)

        # StandardClf -> density estimator for P(y |m,z,a)
        standard = StandardClf.load_from_checkpoint("../models/" + "StandardClf_"+ experiment + "/StandardClf_"+ experiment + "_checkpoints.ckpt")

        ctf_effects_naive = functions.Ctf_effects_binary(network=standard, full_data = test_data, task = "binary", sensitivity_param=None, density_estimator=density_estimator)
        de, ie, se = ctf_effects_naive.unbounded_effects(sensitive_attributes[0], sensitive_attributes[1], 1.0)
        fairness_naive = pd.DataFrame({"Experiment": experiment, "DE": [de],"IE": [ie],"SE": [se]})
        naive = pd.concat([naive, fairness_naive], ignore_index= True)
        
        for param in sensitivity_parameter:
            ctf_effects_standard = functions.Ctf_effects_binary(network=standard, full_data = test_data, task = "binary", sensitivity_param=param, density_estimator=density_estimator)
            y_ind = 1.0
            ub_DE_y, lb_DE_y = ctf_effects_standard.DE_binary(sensitive_attributes[0], sensitive_attributes[1], y_ind)
            ub_IE_y, lb_IE_y = ctf_effects_standard.IE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)
            ub_SE_y, lb_SE_y = ctf_effects_standard.SE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)

            bounds_standard_df = pd.DataFrame(columns = ["Sensitivity parameter", "Experiment", "DE_ub", "DE_lb", "IE_ub", "IE_lb", "SE_ub","SE_lb"])
            bounds_standard_df = pd.concat([bounds_standard_df, pd.DataFrame({"Sensitivity parameter": param, "Experiment": experiment, "DE_ub": [ub_DE_y], "DE_lb": [lb_DE_y], "IE_ub": [ub_IE_y], "IE_lb": [lb_IE_y], "SE_ub": [ub_SE_y],"SE_lb": [lb_SE_y]})])

            bounds = pd.concat([bounds, bounds_standard_df], ignore_index= True)

    return bounds, oracle, naive, tv


def caculate_data_fairness_continuous(sensitive_attributes, sensitivity_parameter, experiment_list, density_y = False):

    bounds = pd.DataFrame(columns = ["Sensitivity parameter","Experiment", "DE_ub", "DE_lb", "IE_ub", "IE_lb", "SE_ub","SE_lb"])
    oracle = pd.DataFrame(columns = ["Experiment", "DE", "IE", "SE"])
    naive = pd.DataFrame(columns = ["Experiment", "DE", "IE", "SE"])
    label_col = "Y"

    for experiment in experiment_list:

        data_location = "../data/simulator/"+ experiment + "_full_dataframe"
        with open(data_location, "rb") as input:
            oracle_data = pickle.load(input)

        full_data = oracle_data.copy()
        full_data = full_data.drop(["USE", "UIE", "UDE"],axis=1)
        node_names = full_data.drop(["Y"],axis=1).columns.values
        n_classes = num_classes=len(np.unique(full_data[label_col]))
        test_data = full_data.iloc[int(0.8*full_data.shape[0]):,:]
        test_data_oracle = oracle_data.iloc[int(0.8*oracle_data.shape[0]):,:]
        covariates = test_data.iloc[:, test_data.columns != label_col].values
        covariates = torch.tensor(covariates, requires_grad=False).float()
        y = torch.tensor(test_data[label_col].values, requires_grad=False).float().unsqueeze(dim=1)

        checkpointpath_density = "../models/density_estimator_" + experiment + "/density_estimator_" + experiment + "_checkpoints.ckpt"
        checkpointpath_prop = "../models/density_estimator_A_" + experiment + "/density_estimator_A_" + experiment + "_checkpoints.ckpt"
        density_estimator = Density_estimator.load_from_checkpoint(checkpointpath_density)
        propensity_model = Density_estimator.load_from_checkpoint(checkpointpath_prop)

        # StandardClf -> density estimator for P(y |m,z,a)
        standard = StandardClf.load_from_checkpoint("../models/" + "StandardClf_"+ experiment + "/StandardClf_"+ experiment + "_checkpoints.ckpt")

        ctf_effects_naive = functions.Ctf_effects_binary(network=standard, full_data = test_data, task = "binary", sensitivity_param=None, density_estimator=density_estimator,)
        de, ie, se = ctf_effects_naive.unbounded_effects(sensitive_attributes[0], sensitive_attributes[1], 1.0)
        fairness_naive = pd.DataFrame({"Experiment": experiment, "DE": [de],"IE": [ie],"SE": [se]})
        naive = pd.concat([naive, fairness_naive], ignore_index= True)
        
        for param in sensitivity_parameter:
            ctf_effects_standard = functions.Ctf_effects_binary(network=standard, full_data = test_data, task = "binary", sensitivity_param=param, density_estimator=density_estimator)
            y_ind = 1.0
            ub_DE_y, lb_DE_y = ctf_effects_standard.DE_binary(sensitive_attributes[0], sensitive_attributes[1], y_ind)
            ub_IE_y, lb_IE_y = ctf_effects_standard.IE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)
            ub_SE_y, lb_SE_y = ctf_effects_standard.SE_binary(sensitive_attributes[1], sensitive_attributes[0], y_ind)

            bounds_standard_df = pd.DataFrame(columns = ["Sensitivity parameter", "Experiment", "DE_ub", "DE_lb", "IE_ub", "IE_lb", "SE_ub","SE_lb"])
            bounds_standard_df = pd.concat([bounds_standard_df, pd.DataFrame({"Sensitivity parameter": param, "Experiment": experiment, "DE_ub": [ub_DE_y], "DE_lb": [lb_DE_y], "IE_ub": [ub_IE_y], "IE_lb": [lb_IE_y], "SE_ub": [ub_SE_y],"SE_lb": [lb_SE_y]})])

            bounds = pd.concat([bounds, bounds_standard_df], ignore_index= True)

    return bounds, oracle, naive



def caculate_data_fairness_multiclass(sensitive_attributes, sensitivity_parameter, experiment_list):

    bounds = pd.DataFrame(columns = ["Sensitivity parameter","Experiment", "DE_ub", "DE_lb", "IE_ub", "IE_lb", "SE_ub","SE_lb"])
    oracle = pd.DataFrame(columns = ["Experiment", "DE", "IE", "SE"])
    naive = pd.DataFrame(columns = ["Experiment", "DE", "IE", "SE"])
    tv = pd.DataFrame(columns = ["Experiment", "TV"])
    label_col = "Y"

    for experiment in experiment_list:

        data_location = "../data/simulator/"+ experiment + "_full_dataframe"
        with open(data_location, "rb") as input:
            oracle_data = pickle.load(input)

        full_data = oracle_data.copy()
        full_data = full_data.drop(["USE", "UIE", "UDE"],axis=1)
        node_names = full_data.drop(["Y"],axis=1).columns.values
        n_classes = num_classes=len(np.unique(full_data[label_col]))
        test_data = full_data.iloc[int(0.8*full_data.shape[0]):,:]
        covariates = test_data.iloc[:, test_data.columns != label_col].values
        covariates = torch.tensor(covariates, requires_grad=False).float()
        y = torch.tensor(test_data[label_col].values, requires_grad=False).float().unsqueeze(dim=1)

        checkpointpath_density = "../models/density_estimator_" + experiment + "/density_estimator_" + experiment + "_checkpoints.ckpt"
        density_estimator = Density_estimator.load_from_checkpoint(checkpointpath_density)

        total_var = (oracle_data.loc[oracle_data["A"]==sensitive_attributes[1], "Y"].value_counts(normalize=True)[sorted(oracle_data["Y"].unique())] - oracle_data.loc[oracle_data["A"]==sensitive_attributes[0], "Y"].value_counts(normalize=True)[sorted(oracle_data["Y"].unique())])[1]
        TV_df = pd.DataFrame({"Experiment": experiment, "TV": [total_var]})
        tv = pd.concat([tv, TV_df], ignore_index= True)

        # Oracle
        nodes, edges = graph_file_parser("../data/simulator/"+ experiment +".yml")
        g = Graph(nodes=nodes, edges=edges)

        oracle_data = g.sample(100000)[["A", "Z1", "Z2", "Z3", "Z4", "M", "Y"]]
        oracle_data["M"] = np.where(oracle_data["M"] < 5, oracle_data["M"], 5)
        oracle_data["Y"] = np.where(oracle_data["Y"] < 6, oracle_data["Y"], 6) 
        rel_freq_y = oracle_data["Y"].sort_values().value_counts(normalize=True)
        do_s0 = g.do(size=100000, interventions={'A': sensitive_attributes[0]})
        do_s0["M"] = np.where(do_s0["M"] < 5, do_s0["M"], 5)
        do_s0["Y"] = np.where(do_s0["Y"] < 6, do_s0["Y"], 6)  
        do_s1 = g.do(size=100000, interventions={'A': sensitive_attributes[1]})
        do_s1["M"] = np.where(do_s1["M"] < 5, do_s1["M"], 5)
        do_s1["Y"] = np.where(do_s1["Y"] < 6, do_s1["Y"], 6)  
        do_s0_m0 = g.do(size=100000, interventions={'A': sensitive_attributes[0], 'M': 0.0})
        do_s0_m0["M"] = np.where(do_s0_m0["M"] < 5, do_s0_m0["M"], 5)
        do_s0_m0["Y"] = np.where(do_s0_m0["Y"] < 6, do_s0_m0["Y"], 6)  
        do_s1_m0 = g.do(size=100000, interventions={'A': sensitive_attributes[1], 'M': 0.0})
        do_s1_m0["M"] = np.where(do_s1_m0["M"] < 5, do_s1_m0["M"], 5)
        do_s1_m0["Y"] = np.where(do_s1_m0["Y"] < 6, do_s1_m0["Y"], 6)  
        do_s0_m1 = g.do(size=100000, interventions={'A': sensitive_attributes[0], 'M': 1.0})
        do_s0_m1["M"] = np.where(do_s0_m1["M"] < 5, do_s0_m1["M"], 5)
        do_s0_m1["Y"] = np.where(do_s0_m1["Y"] < 6, do_s0_m1["Y"], 6)  
        do_s1_m1 = g.do(size=100000, interventions={'A': sensitive_attributes[1], 'M': 1.0})
        do_s1_m1["M"] = np.where(do_s1_m1["M"] < 5, do_s1_m1["M"], 5)
        do_s1_m1["Y"] = np.where(do_s1_m1["Y"] < 6, do_s1_m1["Y"], 6)  
        do_s0_m2 = g.do(size=100000, interventions={'A': sensitive_attributes[0], 'M': 2.0})
        do_s0_m2["M"] = np.where(do_s0_m2["M"] < 5, do_s0_m2["M"], 5)
        do_s0_m2["Y"] = np.where(do_s0_m2["Y"] < 6, do_s0_m2["Y"], 6)  
        do_s1_m2 = g.do(size=100000, interventions={'A': sensitive_attributes[1], 'M': 2.0})
        do_s1_m2["M"] = np.where(do_s1_m2["M"] < 5, do_s1_m2["M"], 5)
        do_s1_m2["Y"] = np.where(do_s1_m2["Y"] < 6, do_s1_m2["Y"], 6)  
        do_s0_m3 = g.do(size=100000, interventions={'A': sensitive_attributes[0], 'M': 3.0})
        do_s0_m3["M"] = np.where(do_s0_m3["M"] < 5, do_s0_m3["M"], 5)
        do_s0_m3["Y"] = np.where(do_s0_m3["Y"] < 6, do_s0_m3["Y"], 6)  
        do_s1_m3 = g.do(size=100000, interventions={'A': sensitive_attributes[1], 'M': 3.0})
        do_s1_m3["M"] = np.where(do_s1_m3["M"] < 5, do_s1_m3["M"], 5)
        do_s1_m3["Y"] = np.where(do_s1_m3["Y"] < 6, do_s1_m3["Y"], 6) 
        do_s0_m4 = g.do(size=100000, interventions={'A': sensitive_attributes[0], 'M': 4.0})
        do_s0_m4["M"] = np.where(do_s0_m4["M"] < 5, do_s0_m4["M"], 5)
        do_s0_m4["Y"] = np.where(do_s0_m4["Y"] < 6, do_s0_m4["Y"], 6)  
        do_s1_m4 = g.do(size=100000, interventions={'A': sensitive_attributes[1], 'M': 4.0})
        do_s1_m4["M"] = np.where(do_s1_m4["M"] < 5, do_s1_m4["M"], 5)
        do_s1_m4["Y"] = np.where(do_s1_m4["Y"] < 6, do_s1_m4["Y"], 6) 
        do_s0_m5 = g.do(size=100000, interventions={'A': sensitive_attributes[0], 'M': 5.0})
        do_s0_m5["M"] = np.where(do_s0_m5["M"] < 5, do_s0_m5["M"], 5)
        do_s0_m5["Y"] = np.where(do_s0_m5["Y"] < 6, do_s0_m5["Y"], 6)  
        do_s1_m5 = g.do(size=100000, interventions={'A': sensitive_attributes[1], 'M': 5.0})
        do_s1_m5["M"] = np.where(do_s1_m5["M"] < 5, do_s1_m5["M"], 5)
        do_s1_m5["Y"] = np.where(do_s1_m5["Y"] < 6, do_s1_m5["Y"], 6) 

        p_y_s0 = oracle_data.loc[oracle_data["A"] == sensitive_attributes[0], "Y"].sort_values().value_counts(normalize=True)
        p_y_do_s0_given_s1 = do_s0.loc[oracle_data["A"] == sensitive_attributes[1], "Y"].sort_values().value_counts(normalize=True)
        p_y_do_s0_given_s1= p_y_do_s0_given_s1.reset_index()["proportion"]
        p_y_s1 = oracle_data.loc[oracle_data["A"] == sensitive_attributes[1], "Y"].sort_values().value_counts(normalize=True)
        p_y_s1 = p_y_s1.reset_index()["proportion"]
        p_y_do_s1_given_s0 = do_s1.loc[oracle_data["A"] == sensitive_attributes[0], "Y"].sort_values().value_counts(normalize=True)
        p_y_do_s1_given_s0 = p_y_do_s1_given_s0.reset_index()["proportion"]

        m_do_s0_given_s1 = do_s0.loc[oracle_data["A"] == sensitive_attributes[1]].reset_index()
        y_do_s0_m0 = do_s0_m0[oracle_data["A"] == sensitive_attributes[1]].reset_index()
        y_do_s0_m0 = y_do_s0_m0[y_do_s0_m0["M"]==m_do_s0_given_s1["M"]].reset_index()
        y_do_s0_m1 = do_s0_m1[oracle_data["A"] == sensitive_attributes[1]].reset_index()
        y_do_s0_m1 = y_do_s0_m1[y_do_s0_m1["M"]==m_do_s0_given_s1["M"]].reset_index()
        y_do_s0_m2 = do_s0_m2[oracle_data["A"] == sensitive_attributes[1]].reset_index()
        y_do_s0_m2 = y_do_s0_m2[y_do_s0_m2["M"]==m_do_s0_given_s1["M"]].reset_index()
        y_do_s0_m3 = do_s0_m3[oracle_data["A"] == sensitive_attributes[1]].reset_index()
        y_do_s0_m3 = y_do_s0_m3[y_do_s0_m3["M"]==m_do_s0_given_s1["M"]].reset_index()
        y_do_s0_m4 = do_s0_m4[oracle_data["A"] == sensitive_attributes[1]].reset_index()
        y_do_s0_m4 = y_do_s0_m4[y_do_s0_m4["M"]==m_do_s0_given_s1["M"]].reset_index()
        y_do_s0_m5 = do_s0_m5[oracle_data["A"] == sensitive_attributes[1]].reset_index()
        y_do_s0_m5 = y_do_s0_m5[y_do_s0_m5["M"]==m_do_s0_given_s1["M"]].reset_index()

        m_do_s0_given_s0 = do_s0.loc[oracle_data["A"] == sensitive_attributes[0]].reset_index()
        y_do_s1_m0 = do_s1_m0[oracle_data["A"] == sensitive_attributes[0]].reset_index()
        y_do_s1_m0 = y_do_s1_m0[y_do_s1_m0["M"]==m_do_s0_given_s0["M"]].reset_index()
        y_do_s1_m1 = do_s1_m1[oracle_data["A"] == sensitive_attributes[0]].reset_index()
        y_do_s1_m1 = y_do_s1_m1[y_do_s1_m1["M"]==m_do_s0_given_s0["M"]].reset_index()
        y_do_s1_m2= do_s1_m2[oracle_data["A"] == sensitive_attributes[0]].reset_index()
        y_do_s1_m2 = y_do_s1_m2[y_do_s1_m2["M"]==m_do_s0_given_s0["M"]].reset_index()
        y_do_s1_m3 = do_s1_m3[oracle_data["A"] == sensitive_attributes[0]].reset_index()
        y_do_s1_m3 = y_do_s1_m3[y_do_s1_m3["M"]==m_do_s0_given_s0["M"]].reset_index()
        y_do_s1_m4 = do_s1_m4[oracle_data["A"] == sensitive_attributes[0]].reset_index()
        y_do_s1_m4 = y_do_s1_m4[y_do_s1_m4["M"]==m_do_s0_given_s0["M"]].reset_index()
        y_do_s1_m5 = do_s1_m5[oracle_data["A"] == sensitive_attributes[0]].reset_index()
        y_do_s1_m5 = y_do_s1_m5[y_do_s1_m5["M"]==m_do_s0_given_s0["M"]].reset_index()

        p_y_do_s1_do_m_do_s0_given_s0 = pd.concat([y_do_s1_m0, y_do_s1_m1, y_do_s1_m2, y_do_s1_m3, y_do_s1_m4, y_do_s1_m5], axis=0)
        p_y_do_s1_do_m_do_s0_given_s0 = p_y_do_s1_do_m_do_s0_given_s0["Y"].sort_values().value_counts(normalize=True)
        p_y_do_s1_do_m_do_s0_given_s0 = p_y_do_s1_do_m_do_s0_given_s0.reset_index()["proportion"]  

        de = (p_y_do_s1_do_m_do_s0_given_s0 - p_y_s0)[1]
        ie = (p_y_do_s1_do_m_do_s0_given_s0 - p_y_do_s1_given_s0)[1]
        se = (p_y_do_s1_given_s0 - p_y_s1)[1]

        fairness_oracle = pd.DataFrame({"Experiment": experiment, "DE": [de],"IE": [ie],"SE": [se]})

        oracle = pd.concat([oracle, fairness_oracle], ignore_index= True)

        # Empirical
        standard = StandardClf.load_from_checkpoint("../models/" + "StandardClf_"+ experiment + "/StandardClf_"+ experiment + "_checkpoints.ckpt")

        ctf_effects_naive = functions.Ctf_effects_binary(network=standard, full_data = test_data, task = "multiclass", sensitivity_param=None, density_estimator=density_estimator)
        de, ie, se = ctf_effects_naive.unbounded_effects(sensitive_attributes[0], sensitive_attributes[1], 1.0)
        fairness_naive = pd.DataFrame({"Experiment": experiment, "DE": [de],"IE": [ie],"SE": [se]})
        naive = pd.concat([naive, fairness_naive], ignore_index= True)
        
        for param in sensitivity_parameter:
            ctf_effects_standard = functions.Ctf_effects_binary(network=standard, full_data = test_data, task = "multiclass", sensitivity_param=param, density_estimator=density_estimator)#, density_y = density_y)

            bounds_standard = np.empty(shape = (0,6))
            for y_ind in sorted(np.unique(oracle_data["Y"])):
                ub_DE_y, lb_DE_y = ctf_effects_standard.DE_binary(sensitive_attributes[0], sensitive_attributes[1], float(y_ind))
                ub_IE_y, lb_IE_y = ctf_effects_standard.IE_binary(sensitive_attributes[0], sensitive_attributes[1], float(y_ind))
                ub_SE_y, lb_SE_y = ctf_effects_standard.SE_binary(sensitive_attributes[0], sensitive_attributes[1], float(y_ind))
                bounds_standard = np.append(bounds_standard, np.expand_dims(np.array([ub_DE_y, lb_DE_y, ub_IE_y, lb_IE_y, ub_SE_y, lb_SE_y]),0), axis = 0)  
            bounds_standard = bounds_standard * np.tile(np.expand_dims(rel_freq_y.values, axis = 1), (1,6))
            bounds_standard = np.sum(bounds_standard.squeeze(), axis = 0)
            
            bounds_standard_df = pd.DataFrame(columns = ["Sensitivity parameter", "Experiment", "DE_ub", "DE_lb", "IE_ub", "IE_lb", "SE_ub","SE_lb"])
            bounds_standard_df = pd.concat([bounds_standard_df, pd.DataFrame({"Sensitivity parameter": param, "Experiment": experiment, "DE_ub": [bounds_standard[0]], "DE_lb": [bounds_standard[1]], 
            "IE_ub": [bounds_standard[2]], "IE_lb": [bounds_standard[3]], "SE_ub": [bounds_standard[4]],"SE_lb": [bounds_standard[5]]})])

            bounds = pd.concat([bounds, bounds_standard_df], ignore_index= True)

    return bounds, oracle, naive, tv