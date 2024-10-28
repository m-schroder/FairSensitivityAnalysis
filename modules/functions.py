import numpy as np
import pandas as pd
import torch
from itertools import product

from modules import helpers

helpers.seed(1)


#---------------------------------------- Bounds functions -------------------------------------


class Bounds_binary:
    def __init__(self, network, full_data, density_estimator, sensitivity_param, task, propensity_model=None):
        if network == None:
            self.network = None
        else:
            self.network = network.eval()
        self.full_data = full_data
        self.data = full_data
        self.task = task
        self.propensity_model = propensity_model
        if density_estimator == None:
            self.density_estimator = None
        else:
            self.density_estimator = density_estimator.eval()
        if self.task == "multiclass":
            self.rel_freq_z = self.data.value_counts(normalize=True, subset = ["Z1", "Z2", "Z3", "Z4"])
            self.rel_freq_z = self.rel_freq_z.reset_index().sort_values(by=["Z1", "Z2", "Z3", "Z4"])
            self.rel_freq_a_z = self.data.value_counts(normalize=True, subset = ["A", "Z1", "Z2", "Z3", "Z4"])
            self.rel_freq_a_z = self.rel_freq_a_z.reset_index().sort_values(by=["Z1", "Z2", "Z3", "Z4"]) 
            self.order = pd.DataFrame(list(product(self.data["Z1"].unique(), self.data["Z2"].unique(), self.data["Z3"].unique(), self.data["Z4"].unique())), columns=(["Z1", "Z2", "Z3", "Z4"]))
        if self.task == "binary":        
            self.rel_freq_z = self.data["Z"].value_counts(normalize=True)[sorted(self.data["Z"].unique())]
            self.rel_freq_z =  self.rel_freq_z.reset_index()
            self.rel_freq_a_z = self.data[["A","Z"]].value_counts(normalize=True)[sorted(self.data["Z"].unique())]
            self.rel_freq_a_z = self.rel_freq_a_z.reset_index().sort_values(by="Z") 
            self.order = self.data["Z"].drop_duplicates().sort_values().reset_index().drop("index", axis = 1)       
        else:
            pass
        self.covariates = self.data.columns
        self.sensitivity_param = sensitivity_param

    def build_score(self,a, data):
        if self.task == "multiclass":
            prop_table_a  = self.rel_freq_a_z.loc[self.rel_freq_a_z["A"] == a].sort_values(by=["Z1", "Z2", "Z3", "Z4"])
            prop_table_a = prop_table_a.merge(self.order, on=["Z1", "Z2", "Z3", "Z4"], how = "outer").fillna(0).sort_values(by=["Z1", "Z2", "Z3", "Z4"])
            prop_a = prop_table_a["proportion"].values/self.rel_freq_z["proportion"].values
        if self.task == "binary":  
            prop_table_a  = self.rel_freq_a_z.loc[self.rel_freq_a_z["A"] == a].sort_values(by="Z")
            prop_table_a = prop_table_a.merge(self.order, on="Z").fillna(0).sort_values(by="Z")
            prop_a = prop_table_a["proportion"].values/self.rel_freq_z["proportion"].values
        else:
            prop_table_a = None
            logits, prop_a = self.propensity_model(data.float()) 
            prop_a = prop_a.detach().numpy().transpose()[0]

        s_plus = 1/((1-1/self.sensitivity_param)*prop_a + 1/self.sensitivity_param)
        s_minus = 1/((1-self.sensitivity_param)*prop_a + self.sensitivity_param)
        c_plus = np.float(self.sensitivity_param/(1+self.sensitivity_param))
        c_minus = np.float(1/(1+self.sensitivity_param))

        return s_plus, s_minus, c_plus, c_minus, prop_table_a 


    def bounds_single(self, a, y):
        """
        Calculate bounds of the form \sum_z Q(z,a_i,S)*P(Z=z),
        where Q(z,a_i,S) = \sum_m P(Y=y| M=m, Z=z, do(A=a_i))*P(M=m|Z=z, do(A=a_i)) 
        """
        temp_df = self.full_data.loc[self.full_data["A"]==a].drop_duplicates()

        prob_m = pd.DataFrame({"M":[], "prob": []})
        with torch.no_grad():
            density_logits, density_estimates_all = self.density_estimator(torch.tensor(temp_df.drop("M", axis = 1).values, requires_grad=False).float())
            density_estimates_all = density_estimates_all.detach().numpy()
        f_m_in = self.order.copy()
        f_m_in["A"] = np.repeat(a, self.order.shape[0])
        f_m_in["M"] = np.repeat(np.nan, self.order.shape[0])
        f_m_in = f_m_in[self.covariates]   

        logits, f_m_z_array = self.density_estimator(torch.tensor(f_m_in.drop("M", axis = 1).values, requires_grad=False).float())
        F_m = np.cumsum(f_m_z_array.detach().numpy(), axis = 1).transpose()[0]

        ind = 0
        Q_plus = np.zeros(shape=np.shape(self.order)[0])
        Q_minus = np.zeros(shape=np.shape(self.order)[0])#

        for m in sorted(np.unique(self.data["M"])): 
            if self.task == "multiclass":
                y_maz = self.data.loc[self.data["M"] == m,["M", "A", "Z1", "Z2", "Z3", "Z4"]].drop_duplicates().copy()
                y_maz = y_maz.loc[y_maz["A"] == a]
                y_maz= y_maz.merge(self.order.drop(["M", "A"], axis = 1), on =["Z1", "Z2", "Z3", "Z4"], how = "outer").sort_values(by=["Z1", "Z2", "Z3", "Z4"]) 
                y_maz = y_maz[self.covariates]
                y_maz["A"] = y_maz["A"].fillna(a)
                y_maz["M"] = y_maz["M"].fillna(m)
                p_y_maz = self.network(torch.tensor(y_maz.values).float())[:, int(y)]  
                p_y_maz = p_y_maz.detach().numpy()
                with torch.no_grad():
                    logits, p_m_az = self.density_estimator(torch.tensor(y_maz.drop(["M"], axis = 1).values).float())
                    p_m_az = p_m_az.detach().numpy()[:, int(m)]
            else:
                y_maz = self.data.loc[self.data["M"] == m,["M", "A", "Z"]].drop_duplicates().copy()
                y_maz = y_maz.loc[y_maz["A"] == a]
                y_maz = y_maz[self.covariates]
                y_maz = y_maz.merge(self.order, on ="Z", how = "outer").sort_values(by="Z")
                logits, p_y_maz = self.network(torch.tensor(y_maz.values).float())
                p_y_maz = p_y_maz.detach().numpy().transpose()[0]
                if y == 0.0:
                    p_y_maz = torch.ones(p_y_maz.size()) - p_y_maz
                    y = 1.0

                density_logits, p_m_az = self.density_estimator(torch.tensor(y_maz.drop(["M"], axis = 1).values).float())
                p_m_az = p_m_az.detach().numpy().transpose()[0]   
                f_m = p_m_az.copy()              
                if m == 0:
                    p_m_az = np.ones(len(p_m_az)) - p_m_az
                p_m_az = np.nan_to_num(p_m_az)

            s_m_plus, s_m_minus, c_m_plus, c_m_minus, prop_table_a = self.build_score(a, torch.tensor(y_maz.drop(["M", "A"], axis = 1).values)) 

            if ind == 0:
                p_plus_m_az = (F_m[ind]<c_m_plus) * (s_m_plus * p_m_az) + (F_m[ind]>=c_m_plus) * (s_m_plus * c_m_plus + s_m_minus * (F_m[ind] - c_m_plus))
                p_minus_m_az = (F_m[ind]<c_m_minus) * (s_m_minus * p_m_az) + (F_m[ind]>=c_m_minus) * (s_m_minus * c_m_minus + s_m_plus * (F_m[ind] - c_m_minus))

            elif ind > 0:

                p_plus_m_az = (F_m[ind]<c_m_plus) * (s_m_plus * p_m_az) + (F_m[ind-1]>c_m_plus)* (s_m_minus * p_m_az) + (F_m[ind-1]<=c_m_plus) * (s_m_plus * (c_m_plus - F_m[ind -1]) + s_m_minus * (F_m[ind] - c_m_plus))               
                p_minus_m_az = (F_m[ind]<c_m_minus) * (s_m_minus * p_m_az) + (F_m[ind-1]>c_m_minus) * (s_m_plus * p_m_az) + (F_m[ind]>=c_m_minus) * (F_m[ind-1]<=c_m_minus) * (s_m_minus * (c_m_minus - F_m[ind -1]) + s_m_plus * (F_m[ind] - c_m_minus))
                
            ind = ind + 1
            if self.task == "multiclass":
                Q_plus = Q_plus + (p_plus_m_az * p_y_maz * (np.argmax(p_y_maz)==int(y))* self.rel_freq_z["proportion"].values)
                Q_minus = Q_minus + (p_minus_m_az * p_y_maz* (np.argmax(p_y_maz)==int(y))* self.rel_freq_z["proportion"].values)
            else:  
                Q_plus = Q_plus + (p_plus_m_az * p_y_maz * (np.round(p_y_maz)==y)* self.rel_freq_z["proportion"].values)
                Q_minus = Q_minus + (p_minus_m_az * p_y_maz* (np.round(p_y_maz)==y)* self.rel_freq_z["proportion"].values)

        ub = np.sum(Q_plus)
        lb = np.sum(Q_minus)

        return ub, lb
    


    def bounds_single_continuous(self, a, y):
        """
        Calculate bounds of the form \sum_z Q(z,a_i,S)*P(Z=z),
        where Q(z,a_i,S) = \sum_m P(Y=y| M=m, Z=z, do(A=a_i))*P(M=m|Z=z, do(A=a_i)) 
        """
        
        temp_df = self.full_data.loc[self.full_data["A"]==a].drop_duplicates()

        prob_m = pd.DataFrame({"M":[], "prob": []})
        with torch.no_grad():
            density_logits, density_estimates_all = self.density_estimator(torch.tensor(temp_df.drop("M", axis = 1).values, requires_grad=False).float())
            density_estimates_all = density_estimates_all.detach().numpy()

        ind = 0
        Q_plus = np.zeros(1)
        Q_minus = np.zeros(1)

        for m in sorted(np.unique(self.data["M"])): 
            y_maz = self.data.loc[self.data["M"] == m].copy()
            y_maz = y_maz.loc[y_maz["A"] == a]
            y_maz = y_maz[self.covariates]
            logits, p_y_maz = self.network(torch.tensor(y_maz.values).float())
            p_y_maz = p_y_maz.detach().numpy().transpose()[0]
            if y == 0.0:
                p_y_maz = torch.ones(p_y_maz.size()) - p_y_maz
                y = 1.0

            density_logits, p_m_az = self.density_estimator(torch.tensor(y_maz.drop(["M"], axis = 1).values).float())
            p_m_az = p_m_az.detach().numpy().transpose()[0]   
            f_m = p_m_az.copy()              
            if m == 0:
                p_m_az = np.ones(len(p_m_az)) - p_m_az
            p_m_az = np.nan_to_num(p_m_az)

            s_m_plus, s_m_minus, c_m_plus, c_m_minus, prop_table_a = self.build_score(a, torch.tensor(y_maz.drop(["M", "A"], axis = 1).values)) 
            F_m = np.concatenate((np.expand_dims((np.ones(len(f_m)) - f_m), axis=0), np.expand_dims(np.ones(len(f_m)), axis=0)), axis = 0)

            if ind == 0:
                p_plus_m_az = (F_m[ind]<c_m_plus) * (s_m_plus * p_m_az) + (F_m[ind]>=c_m_plus) * (s_m_plus * c_m_plus + s_m_minus * (F_m[ind] - c_m_plus))
                p_minus_m_az = (F_m[ind]<c_m_minus) * (s_m_minus * p_m_az) + (F_m[ind]>=c_m_minus) * (s_m_minus * c_m_minus + s_m_plus * (F_m[ind] - c_m_minus))
            elif ind > 0:
                p_plus_m_az = (F_m[ind]<c_m_plus) * (s_m_plus * p_m_az) + (F_m[ind-1]>c_m_plus)* (s_m_minus * p_m_az) + (F_m[ind-1]<=c_m_plus) * (s_m_plus * (c_m_plus - F_m[ind -1]) + s_m_minus * (F_m[ind] - c_m_plus))               
                p_minus_m_az = (F_m[ind]<c_m_minus) * (s_m_minus * p_m_az) + (F_m[ind-1]>c_m_minus) * (s_m_plus * p_m_az) + (F_m[ind]>=c_m_minus) * (F_m[ind-1]<=c_m_minus) * (s_m_minus * (c_m_minus - F_m[ind -1]) + s_m_plus * (F_m[ind] - c_m_minus))
                
            ind = ind + 1

            Q_plus = Q_plus + np.mean(p_plus_m_az * p_y_maz * (np.round(p_y_maz)==y))
            Q_minus = Q_minus + np.mean(p_minus_m_az * p_y_maz* (np.round(p_y_maz)==y))

        return Q_plus, Q_minus


    def bounds_double(self, a_i, a_j, y):
        """
        Calculate bounds of the form \sum_z Q(z,(a_i, a_j),S)*P(Z=z),
        where Q(z,(a_i, a_j),S) = \sum_m P(Y=y| M=m, Z=z, do(A=a_i))*P(M=m|Z=z, do(A=a_j)) 
        """

        temp_df = self.data.loc[self.data["A"]==a_j].drop_duplicates()

        prob_m = pd.DataFrame({"M":[], "prob": []})
        with torch.no_grad():
            density_logits, density_estimates_all = self.density_estimator(torch.tensor(temp_df.drop("M", axis = 1).values).float())
            density_estimates_all = density_estimates_all.detach().numpy()

        f_m_in = self.order.copy()
        f_m_in["A"] = np.repeat(a_j, self.order.shape[0])
        f_m_in["M"] = np.repeat(np.nan, self.order.shape[0])
        f_m_in = f_m_in[self.covariates]   

        logits, f_m_z_array = self.density_estimator(torch.tensor(f_m_in.drop("M", axis = 1).values, requires_grad=False).float())
        F_m = np.cumsum(f_m_z_array.detach().numpy(), axis = 1).transpose()[0]
        
        ind = 0
        Q_plus = np.zeros(shape=np.shape(self.order)[0])
        Q_minus = np.zeros(shape=np.shape(self.order)[0])

        for m in sorted(np.unique(self.data["M"])): 

            if self.task == "multiclass":
                maz = self.data.loc[self.data["M"] == m,["M", "A", "Z1", "Z2", "Z3", "Z4"]].drop_duplicates().copy()
                y_maz = maz.loc[maz["A"] == a_i]
                y_maz= y_maz.merge(self.order.drop(["M", "A"], axis = 1), on =["Z1", "Z2", "Z3", "Z4"], how = "outer").sort_values(by=["Z1", "Z2", "Z3", "Z4"]) 
                y_maz = y_maz[self.covariates]
                y_maz["A"] = y_maz["A"].fillna(a_i)
                y_maz["M"] = y_maz["M"].fillna(m)

                m_za_j = maz.loc[maz["A"] == a_j]
                m_za_j = m_za_j.merge(self.order.drop(["M", "A"], axis = 1), on =["Z1", "Z2", "Z3", "Z4"], how = "outer").sort_values(by=["Z1", "Z2", "Z3", "Z4"])
                m_za_j = m_za_j[self.covariates]
                m_za_j["A"] = m_za_j["A"].fillna(a_j)
                m_za_j["M"] = m_za_j["M"].fillna(m)

                p_y_maz = self.network(torch.tensor(y_maz.values).float())[:, int(y)]
                p_y_maz = p_y_maz.detach().numpy()
                with torch.no_grad():
                    logits, p_m_az = self.density_estimator(torch.tensor(m_za_j.drop(["M"], axis = 1).values, requires_grad=False).float())
                    p_m_az = p_m_az.detach().numpy()[:, int(m)] 
            
            else:
                maz = self.data.loc[self.data["M"] == m,["M", "A", "Z"]].drop_duplicates().copy()
                y_maz =  maz.loc[maz["A"] == a_i]
                y_maz = y_maz[self.covariates]
                y_maz = y_maz.merge(self.order, on ="Z", how = "outer").sort_values(by="Z")

                m_za_j = maz.loc[maz["A"] == a_j]
                m_za_j = m_za_j[self.covariates]
                m_za_j = y_maz
                m_za_j["A"] = np.repeat(a_j, len(m_za_j["A"])) 

                logits, p_y_maz = self.network(torch.tensor(y_maz.drop_duplicates().values).float())
                p_y_maz = p_y_maz.detach().numpy().transpose()[0]
                if y == 0.0:
                    p_y_maz = np.ones(p_y_maz.size()) - p_y_maz
                    y = 1.0
            
                logits, p_m_az = self.density_estimator(torch.tensor(m_za_j.drop(["M"], axis = 1).values).float())
                p_m_az = p_m_az.detach().numpy().transpose()[0]
                f_m = p_m_az.copy()
                if m == 0:
                    p_m_az = np.ones(len(p_m_az)) - p_m_az
                p_m_az = np.nan_to_num(p_m_az)

            s_m_plus, s_m_minus, c_m_plus, c_m_minus, prop_table_a = self.build_score(a_j, torch.tensor(m_za_j.drop(["M", "A"], axis = 1).values)) 

            if ind == 0:
                p_plus_m_az = (F_m[ind]<c_m_plus) * (1/s_m_plus * p_m_az) + (F_m[ind]>=c_m_plus) * (1/s_m_plus * c_m_plus + 1/s_m_minus * (F_m[ind] - c_m_plus))
                p_minus_m_az = (F_m[ind]<c_m_minus) * (1/s_m_minus * p_m_az) + (F_m[ind]>=c_m_minus) * (1/s_m_minus * c_m_minus + 1/s_m_plus * (F_m[ind] - c_m_minus))
            else:
                p_plus_m_az = (F_m[ind]<c_m_plus) * (1/s_m_plus * p_m_az) + (F_m[ind-1]>c_m_plus)* (1/s_m_minus * p_m_az) + (F_m[ind]>=c_m_plus) * (F_m[ind-1]<=c_m_plus) * (1/s_m_plus * (c_m_plus - F_m[ind -1]) + 1/s_m_minus * (F_m[ind] - c_m_plus))
                p_minus_m_az = (F_m[ind]<c_m_minus) * (1/s_m_minus * p_m_az) + (F_m[ind-1]>c_m_minus)* (1/s_m_plus * p_m_az) + (F_m[ind]>=c_m_minus) * (F_m[ind-1]<=c_m_minus) * (1/s_m_minus * (c_m_minus - F_m[ind -1]) + 1/s_m_plus * (F_m[ind] - c_m_minus))
            
            if self.task == "multiclass":
                Q_plus = Q_plus + (p_plus_m_az * p_y_maz * (np.argmax(p_y_maz)==(y)))
                Q_minus = Q_minus + (p_minus_m_az * p_y_maz* (np.argmax(p_y_maz)==(y)))
            else:    
                Q_plus = Q_plus + (p_plus_m_az * p_y_maz * (np.round(p_y_maz)==y))
                Q_minus = Q_minus + (p_minus_m_az * p_y_maz* (np.round(p_y_maz)==y))

            ind = ind + 1

        ub = np.sum(Q_plus * self.rel_freq_z["proportion"].values)
        lb = np.sum(Q_minus * self.rel_freq_z["proportion"].values)

        return ub, lb


    def bounds_double_continuous(self, a_i, a_j, y):
        """
        Calculate bounds of the form \sum_z Q(z,(a_i, a_j),S)*P(Z=z),
        where Q(z,(a_i, a_j),S) = \sum_m P(Y=y| M=m, Z=z, do(A=a_i))*P(M=m|Z=z, do(A=a_j)) 
        """

        temp_df = self.data.loc[self.data["A"]==a_j].drop_duplicates()

        prob_m = pd.DataFrame({"M":[], "prob": []})
        with torch.no_grad():
            density_logits, density_estimates_all = self.density_estimator(torch.tensor(temp_df.drop("M", axis = 1).values).float())
            density_estimates_all = density_estimates_all.detach().numpy()
        
        ind = 0
        Q_plus = np.zeros(1)
        Q_minus = np.zeros(1)

        for m in sorted(np.unique(self.data["M"])): 

            maz = self.data.loc[self.data["M"] == m].copy()
            y_maz =  maz.loc[maz["A"] == a_i]
            y_maz = y_maz[self.covariates]
            m_za_j = y_maz
            m_za_j["A"] = np.repeat(a_j, len(m_za_j["A"])) 

            logits, p_y_maz = self.network(torch.tensor(y_maz.drop_duplicates().values).float())
            p_y_maz = p_y_maz.detach().numpy().transpose()[0]
            if y == 0.0:
                p_y_maz = np.ones(p_y_maz.size()) - p_y_maz
                y = 1.0

            logits, p_m_az = self.density_estimator(torch.tensor(m_za_j.drop(["M"], axis = 1).values).float())
            p_m_az = p_m_az.detach().numpy().transpose()[0]
            f_m = p_m_az.copy()
            if m == 0:
                p_m_az = np.ones(len(p_m_az)) - p_m_az
                p_m_az = np.nan_to_num(p_m_az)

            s_m_plus, s_m_minus, c_m_plus, c_m_minus, prop_table_a = self.build_score(a_j, torch.tensor(m_za_j.drop(["M", "A"], axis = 1).values)) 
            F_m = np.concatenate((np.expand_dims((np.ones(len(f_m)) - f_m), axis=0), np.expand_dims(np.ones(len(f_m)), axis=0)), axis = 0)      

            if ind == 0:
                p_plus_m_az = (F_m[ind]<c_m_plus) * (1/s_m_plus * p_m_az) + (F_m[ind]>=c_m_plus) * (1/s_m_plus * c_m_plus + 1/s_m_minus * (F_m[ind] - c_m_plus))
                p_minus_m_az = (F_m[ind]<c_m_minus) * (1/s_m_minus * p_m_az) + (F_m[ind]>=c_m_minus) * (1/s_m_minus * c_m_minus + 1/s_m_plus * (F_m[ind] - c_m_minus))
            
            else:
                p_plus_m_az = (F_m[ind]<c_m_plus) * (1/s_m_plus * p_m_az) + (F_m[ind-1]>c_m_plus)* (1/s_m_minus * p_m_az) + (F_m[ind]>=c_m_plus) * (F_m[ind-1]<=c_m_plus) * (1/s_m_plus * (c_m_plus - F_m[ind -1]) + 1/s_m_minus * (F_m[ind] - c_m_plus))
                p_minus_m_az = (F_m[ind]<c_m_minus) * (1/s_m_minus * p_m_az) + (F_m[ind-1]>c_m_minus)* (1/s_m_plus * p_m_az) + (F_m[ind]>=c_m_minus) * (F_m[ind-1]<=c_m_minus) * (1/s_m_minus * (c_m_minus - F_m[ind -1]) + 1/s_m_plus * (F_m[ind] - c_m_minus))
             
            Q_plus = Q_plus + np.mean(p_plus_m_az * p_y_maz * (np.round(p_y_maz)==y))
            Q_minus = Q_minus + np.mean(p_minus_m_az * p_y_maz* (np.round(p_y_maz)==y))

            ind = ind + 1

        return Q_plus, Q_minus



class Ctf_effects_binary:
    def __init__(self, network, full_data, task, sensitivity_param=None, density_estimator=None, propensity_model = None):
        if network == None:
            self.network = None
        else:
            self.network = network.eval()
        self.full_data = full_data
        self.task = task
        self.propensity_model = propensity_model
        if density_estimator==None:
            self.density_estimator = None
        else:
            self.density_estimator = density_estimator.eval()
        self.sensitivity_param = sensitivity_param
        self.column_order = full_data.drop("Y", axis = 1).columns   
        if task == "multiclass":
            self.bounds = Bounds_binary(network=self.network, full_data = self.full_data.drop("Y", axis = 1), density_estimator=self.density_estimator, sensitivity_param=self.sensitivity_param, task = "multiclass", propensity_model=propensity_model)
            self.covariates = pd.merge(full_data["M"].drop_duplicates(), self.full_data[["Z1", "Z2", "Z3", "Z4"]].drop_duplicates(), how = "cross")
            self.rel_freq_z = self.full_data.value_counts(normalize=True, subset = ["Z1", "Z2", "Z3", "Z4"])
            self.rel_freq_z = self.rel_freq_z.reset_index().sort_values(by=["Z1", "Z2", "Z3", "Z4"])
            self.order = self.full_data.sort_values(by=["Z1", "Z2", "Z3", "Z4"])[["Z1", "Z2", "Z3", "Z4"]].drop_duplicates()
            self.rel_freq_y = self.full_data["Y"].value_counts(normalize=True)
            self.rel_freq_y = self.rel_freq_y.reset_index()
        elif task == "binary":
            self.bounds = Bounds_binary(network=self.network, full_data = self.full_data.drop("Y", axis = 1), density_estimator=self.density_estimator, sensitivity_param=self.sensitivity_param, task = "binary", propensity_model=propensity_model)
            self.rel_freq_z = self.full_data["Z"].value_counts(normalize=True)[sorted(self.full_data["Z"].unique())]
            self.rel_freq_z = self.rel_freq_z.reset_index()
            self.rel_freq_y = self.full_data["Y"].value_counts(normalize=True)[sorted(self.full_data["Y"].unique())]
            self.rel_freq_y = self.rel_freq_y.reset_index()     
            self.covariates = pd.merge(full_data["M"].drop_duplicates(), self.full_data["Z"].drop_duplicates(), how = "cross")
        else:
            self.bounds = Bounds_binary(network=self.network, full_data = self.full_data.drop("Y", axis = 1), density_estimator=self.density_estimator, sensitivity_param=self.sensitivity_param, task = "continuous", propensity_model=propensity_model)
            self.rel_freq_y = self.full_data["Y"].value_counts(normalize=True)[sorted(self.full_data["Y"].unique())]
            self.rel_freq_y = self.rel_freq_y.reset_index()     


    def DE_binary(self, a_i, a_j, y):
        """
        Function to calculate bounds for  DE_{a_i, a_j}(y|a_i) for binary A

        input_data = dataframe of prediction input (does not include column "label")
        density_estimate = dataframe with columns M,A,Z and M_hat, where M_hat contains the class probabilites for M given A,Z (from density estimator function)
        """

        if self.task == "binary":   
            ub_double, lb_double = self.bounds.bounds_double(a_i = a_j, a_j=a_i, y=y) 

            data_ai = self.full_data.drop("Y", axis = 1).loc[self.full_data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', 'Z']).sort_values(by=['M', 'Z']).drop_duplicates()[self.column_order]
            data_aj = self.full_data.drop("Y", axis = 1).loc[self.full_data["A"] == a_j].merge(self.covariates, how="outer", on= ['M', 'Z']).sort_values(by=['M', 'Z']).drop_duplicates()[self.column_order]
            data_aiaj = data_ai.copy()
            data_aiaj["A"] = np.repeat(a_j, len(data_aiaj["A"]))
            
            p_ai = np.sum(self.full_data["A"] == a_i)/self.full_data.shape[0]
            p_aj = np.sum(self.full_data["A"] == a_j)/self.full_data.shape[0]
            p_z = data_ai.merge(self.rel_freq_z, how="outer", on= ['Z']).drop_duplicates().sort_values(by=['M', 'Z'])["proportion"]
            
            with torch.no_grad():
                logits, m_hat = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
                m_hat = m_hat.detach().numpy().transpose()[0]
                data_ai["M_hat"] = m_hat
                data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
                p_m_ai = data_ai["M_hat"].values
                data_ai = data_ai.drop("M_hat", axis = 1)

                logits, y_ai = self.network(torch.tensor(data_ai.values).float())
                y_ai = y_ai.detach().numpy().transpose()[0]
                logits, y_mza_j = self.network(torch.tensor(data_aiaj.values).float())
                y_mza_j = y_mza_j.detach().numpy().transpose()[0]
            if y == 0.0:
                y_ai = np.ones(len(y_ai)) - y_ai
                y_mza_j = np.ones(len(y_mza_j)) - y_mza_j
                y=1.0

            y_ai = np.nan_to_num(y_ai)
            y_mza_j = np.nan_to_num(y_mza_j)

            DE_ub = 1/p_ai * ub_double - p_aj/p_ai * np.sum(y_mza_j * (np.round(y_mza_j)==y) * p_m_ai * p_z) - np.sum(y_ai * (np.round(y_ai)==y) * p_m_ai * p_z)
            DE_lb = 1/p_ai * lb_double - p_aj/p_ai * np.sum(y_mza_j * (np.round(y_mza_j)==y) * p_m_ai * p_z) - np.sum(y_ai * (np.round(y_ai)==y) * p_m_ai * p_z)

        elif self.task == "multiclass":
            ub_double, lb_double = self.bounds.bounds_double(a_i = a_j, a_j=a_i, y=y) 
            
            data_ai = self.full_data.drop("Y", axis = 1).loc[self.full_data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"]).drop_duplicates()[self.column_order]
            data_aj = self.full_data.drop("Y", axis = 1).loc[self.full_data["A"] == a_j].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"]).drop_duplicates()[self.column_order]
            p_z = data_ai.merge(self.rel_freq_z, how="outer", on= ["Z1", "Z2", "Z3", "Z4"]).drop_duplicates().sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"])["proportion"]  
            p_ai = np.sum(self.full_data["A"] == a_i)/self.full_data.shape[0]
            p_aj = np.sum(self.full_data["A"] == a_j)/self.full_data.shape[0]

            with torch.no_grad():
                m_hat_i_index = torch.argmax(self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())[1], dim = 1).detach().numpy().reshape(-1,1)
                data_ai["M_hat"] = np.repeat(np.expand_dims(sorted(np.unique(self.full_data["M"])), axis = 0), np.shape(data_ai)[0], axis=0)[:,m_hat_i_index]
                p_m_ai = data_ai["M_hat"].values.squeeze()
                data_ai = data_ai.drop("M_hat", axis = 1)

                y_ai = self.network(torch.tensor(data_ai.drop_duplicates().values).float())[:, sorted(np.unique(self.full_data["Y"])).index(y)]
                y_ai = np.nan_to_num(y_ai.detach().numpy())
                y_mza_j = self.network(torch.tensor(data_aj.drop_duplicates().values).float())[:, sorted(np.unique(self.full_data["Y"])).index(y)]
                y_mza_j = np.nan_to_num(y_mza_j.detach().numpy())
            
            DE_ub = 1/p_ai * ub_double - p_aj/p_ai * np.mean(y_mza_j * p_m_ai * p_z) - np.sum(y_ai * p_m_ai * p_z)
            DE_lb = 1/p_ai * lb_double - p_aj/p_ai * np.mean(y_mza_j * p_m_ai * p_z) - np.sum(y_ai * p_m_ai * p_z)

        else:
            ub_double, lb_double = self.bounds.bounds_double_continuous(a_i = a_j, a_j=a_i, y=y) 
            
            data_ai = self.full_data.drop("Y", axis = 1).loc[self.full_data["A"] == a_i]
            data_aj = self.full_data.drop("Y", axis = 1).loc[self.full_data["A"] == a_j]
            data_aiaj = data_ai.copy()
            data_aiaj["A"] = np.repeat(a_j, len(data_aiaj["A"]))
            
            p_ai = np.sum(self.full_data["A"] == a_i)/self.full_data.shape[0]
            p_aj = np.sum(self.full_data["A"] == a_j)/self.full_data.shape[0]
            
            with torch.no_grad():
                logits, m_hat = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
                m_hat = m_hat.detach().numpy().transpose()[0]
                data_ai["M_hat"] = m_hat
                data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
                p_m_ai = data_ai["M_hat"].values
                data_ai = data_ai.drop("M_hat", axis = 1)

                logits, y_ai = self.network(torch.tensor(data_ai.values).float())
                y_ai = y_ai.detach().numpy().transpose()[0]
                logits, y_mza_j = self.network(torch.tensor(data_aiaj.values).float())
                y_mza_j = y_mza_j.detach().numpy().transpose()[0]
            if y == 0.0:
                y_ai = np.ones(len(y_ai)) - y_ai
                y_mza_j = np.ones(len(y_mza_j)) - y_mza_j
                y=1.0

            y_ai = np.nan_to_num(y_ai)
            y_mza_j = np.nan_to_num(y_mza_j)

            DE_ub = 1/p_ai * ub_double - p_aj/p_ai * np.mean(y_mza_j * (np.round(y_mza_j)==y) * p_m_ai) - np.mean(y_ai * (np.round(y_ai)==y) * p_m_ai)
            DE_lb = 1/p_ai * lb_double - p_aj/p_ai * np.mean(y_mza_j * (np.round(y_mza_j)==y) * p_m_ai) - np.mean(y_ai * (np.round(y_ai)==y) * p_m_ai)         

        return DE_ub, DE_lb
    

    def IE_binary(self, a_i, a_j, y):
        """
        Function to calculate bounds for  IE_{a_i, a_j}(y|a_j) for binary A
        """

        if self.task == "binary":

            ub_double, lb_double = self.bounds.bounds_double(a_i, a_j, y)
            ub_single, lb_single = self.bounds.bounds_single(a_i, y)

            data_ai = self.full_data.drop("Y", axis = 1).loc[self.full_data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', 'Z']).sort_values(by=['M', 'Z']).drop_duplicates()[self.column_order]
            data_aj = self.full_data.drop("Y", axis = 1).loc[self.full_data["A"] == a_j].merge(self.covariates, how="outer", on= ['M', 'Z']).sort_values(by=['M', 'Z']).drop_duplicates()[self.column_order]
            data_aiaj = data_ai.copy()
            data_aiaj["A"] = np.repeat(a_j, len(data_aiaj["A"]))

            p_ai = np.sum(self.full_data["A"] == a_i)/self.full_data.shape[0]
            p_aj = np.sum(self.full_data["A"] == a_j)/self.full_data.shape[0]
            p_z = data_ai.merge(self.rel_freq_z, how="outer", on= ['Z']).drop_duplicates().sort_values(by=['M', 'Z'])["proportion"]

            with torch.no_grad():
                logits, m_hat_i = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
                logits, m_hat_j = self.density_estimator(torch.tensor(data_aj.drop("M", axis = 1).values).float())
                m_hat_i = m_hat_i.detach().numpy().transpose()[0]            
                m_hat_j = m_hat_j.detach().numpy().transpose()[0]

                data_ai["M_hat"] = m_hat_i
                data_aj["M_hat"] = m_hat_j
                data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
                p_m_ai = data_ai["M_hat"].values
                data_aj.loc[data_aj["M"] == 0.0,"M_hat"] = np.ones(shape=data_aj.loc[data_aj["M"] == 0.0,"M_hat"].shape) - data_aj.loc[data_aj["M"] == 0.0,"M_hat"]
                p_m_aj = data_aj["M_hat"].values
                data_ai = data_ai.drop("M_hat", axis = 1)
                data_aj = data_aj.drop("M_hat", axis = 1)

                logits, y_mza = self.network(torch.tensor(data_ai.values).float())
                y_mza = y_mza.detach().numpy().transpose()[0]
            if y == 0.0:
                y_mza = np.ones(len(y_mza)) - y_mza
                y=1.0
            y_mza = np.nan_to_num(y_mza)

            IE_ub = 1/p_aj * (ub_double-lb_single) - p_ai/p_aj * np.sum(y_mza * (np.round(y_mza)==y) * (p_m_aj - p_m_ai) * p_z)
            IE_lb = 1/p_aj * (lb_double-ub_single) - p_ai/p_aj * np.sum(y_mza * (np.round(y_mza)==y) * (p_m_aj - p_m_ai) * p_z)

        elif self.task == "multiclass":
            ub_double, lb_double = self.bounds.bounds_double(a_i, a_j, y)
            ub_single, lb_single = self.bounds.bounds_single(a_i, y)

            data_ai = self.full_data.drop("Y", axis = 1).loc[self.full_data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"]).drop_duplicates()[self.column_order]
            data_aj = self.full_data.drop("Y", axis = 1).loc[self.full_data["A"] == a_j].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"]).drop_duplicates()[self.column_order]
            p_z = data_ai.merge(self.rel_freq_z, how="outer", on= ["Z1", "Z2", "Z3", "Z4"]).drop_duplicates().sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"])["proportion"]   
            p_ai = np.sum(self.full_data["A"] == a_i)/self.full_data.shape[0]
            p_aj = np.sum(self.full_data["A"] == a_j)/self.full_data.shape[0]

            with torch.no_grad():
                m_hat_i_index = torch.argmax(self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())[1], dim = 1).detach().numpy().reshape(-1,1)
                m_hat_j_index = torch.argmax(self.density_estimator(torch.tensor(data_aj.drop("M", axis = 1).values).float())[1], dim = 1).detach().numpy().reshape(-1,1)
                data_ai["M_hat"] = np.repeat(np.expand_dims(sorted(np.unique(self.full_data["M"])), axis = 0), np.shape(data_ai)[0], axis=0)[:,m_hat_i_index]
                data_aj["M_hat"] = np.repeat(np.expand_dims(sorted(np.unique(self.full_data["M"])), axis = 0), np.shape(data_aj)[0], axis=0)[:,m_hat_j_index]
                p_m_ai = data_ai["M_hat"].values.squeeze()
                p_m_aj = data_aj["M_hat"].values.squeeze()
                data_ai = data_ai.drop("M_hat", axis = 1)
                data_aj = data_aj.drop("M_hat", axis = 1)

                y_mza = self.network(torch.tensor(data_ai.drop_duplicates().values).float())[:, sorted(np.unique(self.full_data["Y"])).index(y)]
                y_mza = np.nan_to_num(y_mza.detach().numpy())
            
            IE_ub = 1/p_aj * (ub_double-lb_single) - p_ai/p_aj * np.mean(y_mza * (p_m_aj - p_m_ai) * p_z)
            IE_lb = 1/p_aj * (lb_double-ub_single) - p_ai/p_aj * np.mean(y_mza * (p_m_aj - p_m_ai) * p_z)

        else:
            ub_double, lb_double = self.bounds.bounds_double_continuous(a_i, a_j, y)
            ub_single, lb_single = self.bounds.bounds_single_continuous(a_i, y)

            data_ai = self.full_data.drop("Y", axis = 1).loc[self.full_data["A"] == a_i]
            data_aj = self.full_data.drop("Y", axis = 1).loc[self.full_data["A"] == a_j]
            data_aiaj = data_ai.copy()
            data_aiaj["A"] = np.repeat(a_j, len(data_aiaj["A"]))

            p_ai = np.sum(self.full_data["A"] == a_i)/self.full_data.shape[0]
            p_aj = np.sum(self.full_data["A"] == a_j)/self.full_data.shape[0]

            with torch.no_grad():
                logits, m_hat_i = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
                logits, m_hat_ij = self.density_estimator(torch.tensor(data_aiaj.drop("M", axis = 1).values).float())
                m_hat_i = m_hat_i.detach().numpy().transpose()[0]            
                m_hat_ij = m_hat_ij.detach().numpy().transpose()[0]

                data_ai["M_hat"] = m_hat_i
                data_aiaj["M_hat"] = m_hat_ij
                data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
                p_m_ai = data_ai["M_hat"].values
                data_aiaj.loc[data_aiaj["M"] == 0.0,"M_hat"] = np.ones(shape=data_aiaj.loc[data_aiaj["M"] == 0.0,"M_hat"].shape) - data_aiaj.loc[data_aiaj["M"] == 0.0,"M_hat"]
                p_m_aiaj = data_aiaj["M_hat"].values
                data_ai = data_ai.drop("M_hat", axis = 1)
                data_aiaj = data_aiaj.drop("M_hat", axis = 1)

                logits, y_mza = self.network(torch.tensor(data_ai.values).float())
                y_mza = y_mza.detach().numpy().transpose()[0]
            if y == 0.0:
                y_mza = np.ones(len(y_mza)) - y_mza
                y=1.0
            y_mza = np.nan_to_num(y_mza)

            IE_ub = 1/p_aj * (ub_double-lb_single) - p_ai/p_aj * np.mean(y_mza * (np.round(y_mza)==y) * (p_m_aiaj - p_m_ai))
            IE_lb = 1/p_aj * (lb_double-ub_single) - p_ai/p_aj * np.mean(y_mza * (np.round(y_mza)==y) * (p_m_aiaj - p_m_ai))

        return IE_ub, IE_lb


    def SE_binary(self, a_i, a_j, y):
        """
        Function to calculate bounds for  SE_{a_i, a_j}(y) for binary A
        """  

        if self.task == "binary":
            ub_single, lb_single = self.bounds.bounds_single(a_i, y)  

            data_ai = self.full_data.drop("Y", axis=1).loc[self.full_data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', 'Z']).sort_values(by=['M', 'Z']).drop_duplicates()[self.column_order]
            
            p_ai = np.sum(self.full_data["A"] == a_i)/self.full_data.shape[0]
            p_aj = np.sum(self.full_data["A"] == a_j)/self.full_data.shape[0]
            p_z = data_ai.merge(self.rel_freq_z, how="outer", on= ['Z']).drop_duplicates().sort_values(by=['M', 'Z'])["proportion"]     

            with torch.no_grad():
                logits, m_hat_i = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
                m_hat_i = m_hat_i.detach().numpy().transpose()[0]

                data_ai["M_hat"] = m_hat_i
                data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
                p_m_ai = data_ai["M_hat"].values
                data_ai = data_ai.drop("M_hat", axis = 1)

                logits, y_ai = self.network(torch.tensor(data_ai.values).float())
                y_ai = y_ai.detach().numpy().transpose()[0]
            if y == 0.0:
                y_ai = np.ones(len(y_ai)) - y_ai
                y=1.0
            y_ai = np.nan_to_num(y_ai)

            SE_ub = 1/p_aj * ub_single - (1 + p_ai/p_aj) * np.sum(y_ai * (np.round(y_ai)==y) * p_m_ai * p_z)
            SE_lb = 1/p_aj * lb_single - (1 + p_ai/p_aj) * np.sum(y_ai * (np.round(y_ai)==y) * p_m_ai * p_z)

        elif self.task == "multiclass":
            ub_single, lb_single = self.bounds.bounds_single(a_i, y)  

            data_ai = self.full_data.drop("Y", axis = 1).loc[self.full_data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"]).drop_duplicates()[self.column_order]
            p_z = data_ai.merge(self.rel_freq_z, how="outer", on= ["Z1", "Z2", "Z3", "Z4"]).drop_duplicates().sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"])["proportion"]   
            p_ai = np.sum(self.full_data["A"] == a_i)/self.full_data.shape[0]
            p_aj = np.sum(self.full_data["A"] == a_j)/self.full_data.shape[0]

            with torch.no_grad():
                m_hat_i_index = torch.argmax(self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())[1], dim = 1).detach().numpy().reshape(-1,1)
                data_ai["M_hat"] = np.repeat(np.expand_dims(sorted(np.unique(self.full_data["M"])), axis = 0), np.shape(data_ai)[0], axis=0)[:,m_hat_i_index]
                p_m_ai = data_ai["M_hat"].values.squeeze()
                data_ai = data_ai.drop("M_hat", axis = 1)
                y_ai = self.network(torch.tensor(data_ai.drop_duplicates().values).float())[:, sorted(np.unique(self.full_data["Y"])).index(y)]
                y_ai = np.nan_to_num(y_ai.detach().numpy())
            
            SE_ub = 1/p_aj * ub_single - (1 + p_ai/p_aj) * np.mean(y_ai * p_m_ai * p_z)
            SE_lb = 1/p_aj * lb_single - (1 + p_ai/p_aj) * np.mean(y_ai * p_m_ai * p_z)

        else:
            ub_single, lb_single = self.bounds.bounds_single_continuous(a_i, y)  

            data_ai = self.full_data.drop("Y", axis=1).loc[self.full_data["A"] == a_i]
            
            p_ai = np.sum(self.full_data["A"] == a_i)/self.full_data.shape[0]
            p_aj = np.sum(self.full_data["A"] == a_j)/self.full_data.shape[0]

            with torch.no_grad():
                logits, m_hat_i = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
                m_hat_i = m_hat_i.detach().numpy().transpose()[0]

                data_ai["M_hat"] = m_hat_i
                data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
                p_m_ai = data_ai["M_hat"].values
                data_ai = data_ai.drop("M_hat", axis = 1)

                logits, y_ai = self.network(torch.tensor(data_ai.values).float())
                y_ai = y_ai.detach().numpy().transpose()[0]
            if y == 0.0:
                y_ai = np.ones(len(y_ai)) - y_ai
                y=1.0
            y_ai = np.nan_to_num(y_ai)

            SE_ub = 1/p_aj * ub_single - (1 + p_ai/p_aj) * np.mean(y_ai * (np.round(y_ai)==y) * p_m_ai)
            SE_lb = 1/p_aj * lb_single - (1 + p_ai/p_aj) * np.mean(y_ai * (np.round(y_ai)==y) * p_m_ai)

        return SE_ub, SE_lb
   

    def unbounded_effects(self, a_i, a_j, y): 
        """
        Function to calculate the original (unbounded) effects
        """
        if self.task == "binary":
            data_ai = self.full_data.loc[self.full_data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', 'Z']).drop("Y", axis = 1).sort_values(by=['M', 'Z']).drop_duplicates()[self.column_order]
            data_aj = self.full_data.loc[self.full_data["A"] == a_j].merge(self.covariates, how="outer", on= ['M', 'Z']).drop("Y", axis = 1).sort_values(by=['M', 'Z']).drop_duplicates()[self.column_order]
            p_z = data_ai.merge(self.rel_freq_z, how="outer", on= ['Z']).drop_duplicates().sort_values(by=['M', 'Z'])["proportion"]    
            p_ai = np.sum(self.full_data["A"] == a_i)/self.full_data.shape[0]
            p_aj = np.sum(self.full_data["A"] == a_j)/self.full_data.shape[0]
        
            with torch.no_grad():
                logits, m_hat_i = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
                logits, m_hat_j = self.density_estimator(torch.tensor(data_aj.drop("M", axis = 1).values).float())
                logits, m_hat_ij = self.density_estimator(torch.tensor(data_aiaj.drop("M", axis = 1).values).float())  
                logits, m_hat_ji = self.density_estimator(torch.tensor(data_ajai.drop("M", axis = 1).values).float())  
                m_hat_i = m_hat_i.detach().numpy().transpose()[0]            
                m_hat_j = m_hat_j.detach().numpy().transpose()[0]

                data_ai["M_hat"] = m_hat_i
                data_aj["M_hat"] = m_hat_j

                data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
                p_m_ai = data_ai["M_hat"].values
                data_aj.loc[data_aj["M"] == 0.0,"M_hat"] = np.ones(shape=data_aj.loc[data_aj["M"] == 0.0,"M_hat"].shape) - data_aj.loc[data_aj["M"] == 0.0,"M_hat"]
                p_m_aj = data_aj["M_hat"].values
                data_ai = data_ai.drop("M_hat", axis = 1)
                data_aj = data_aj.drop("M_hat", axis = 1)
       
                logits, y_ai = self.network(torch.tensor(data_ai.values).float())
                y_ai = y_ai.detach().numpy().transpose()[0]
                logits, y_aj = self.network(torch.tensor(data_aj.values).float())
                y_aj = y_aj.detach().numpy().transpose()[0]

            if y == 0.0:
                y_ai = np.ones(y_ai.size()) - y_ai
                y_aj = np.ones(y_aj.size()) - y_aj
                y = 1.0
                
            y_ai = np.nan_to_num(y_ai)
            y_aj = np.nan_to_num(y_aj)
            
            de = np.sum(y_aj * (np.round(y_aj)==y) * p_m_ai * p_z) - np.sum(y_ai * (np.round(y_ai)==y) * p_m_ai * p_z)
            ie = np.sum(y_aj * (np.round(y_aj)==y) * (p_m_ai - p_m_aj) * p_z) 
            se = np.sum(y_ai * (np.round(y_ai)==y) * p_m_ai * p_z)
        
        if self.task == "multiclass":  
            data_ai = self.full_data.drop("Y", axis = 1).loc[self.full_data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"]).drop_duplicates()[self.column_order]
            data_aj = self.full_data.drop("Y", axis = 1).loc[self.full_data["A"] == a_j].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"]).drop_duplicates()[self.column_order]

            p_z = data_ai.merge(self.rel_freq_z, how="outer", on= ["Z1", "Z2", "Z3", "Z4"]).drop_duplicates().sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"])["proportion"]    

            with torch.no_grad():
                m_hat_i_index = torch.argmax(self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())[1], dim = 1).detach().numpy().reshape(-1,1)
                m_hat_j_index = torch.argmax(self.density_estimator(torch.tensor(data_aj.drop("M", axis = 1).values).float())[1], dim = 1).detach().numpy().reshape(-1,1)

                data_ai["M_hat"] = np.repeat(np.expand_dims(sorted(np.unique(self.full_data["M"])), axis = 0), np.shape(data_ai)[0], axis=0)[:,m_hat_i_index]
                data_aj["M_hat"] = np.repeat(np.expand_dims(sorted(np.unique(self.full_data["M"])), axis = 0), np.shape(data_aj)[0], axis=0)[:,m_hat_j_index]
                p_m_ai = data_ai["M_hat"].values.squeeze()
                p_m_aj = data_aj["M_hat"].values.squeeze()
                data_ai = data_ai.drop("M_hat", axis = 1)
                data_aj = data_aj.drop("M_hat", axis = 1)
                y_ai = self.network(torch.tensor(data_ai.drop_duplicates().values).float())[:, sorted(np.unique(self.full_data["Y"])).index(y)]
                y_aj = self.network(torch.tensor(data_aj.drop_duplicates().values).float())[:, sorted(np.unique(self.full_data["Y"])).index(y)]

            y_ai = np.nan_to_num(y_ai.detach().numpy())
            y_aj = np.nan_to_num(y_aj.detach().numpy())

            de = np.sum(y_aj * p_m_ai * p_z) - np.sum(y_ai * p_m_ai * p_z)
            ie = np.sum(y_aj * (p_m_ai - p_m_aj) * p_z) 
            se = 0

        else:
            data_ai = self.full_data.drop("Y", axis = 1).loc[self.full_data["A"] == a_i].copy()
            data_aj = self.full_data.drop("Y", axis = 1).loc[self.full_data["A"] == a_j].copy()
            data_aiaj = data_ai.copy()
            data_aiaj["A"] = np.repeat(a_j, len(data_aiaj["A"]))
            data_ajai = data_aj.copy()
            data_ajai["A"] = np.repeat(a_i, len(data_ajai["A"]))

            p_ai = np.sum(self.full_data["A"] == a_i)/self.full_data.shape[0]
            p_aj = np.sum(self.full_data["A"] == a_j)/self.full_data.shape[0]
        
            with torch.no_grad():
                logits, m_hat_i = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
                logits, m_hat_ij = self.density_estimator(torch.tensor(data_aiaj.drop("M", axis = 1).values).float())  
                logits, m_hat_ji = self.density_estimator(torch.tensor(data_ajai.drop("M", axis = 1).values).float())  
                m_hat_i = m_hat_i.detach().numpy().transpose()[0]   
                m_hat_ij = m_hat_ij.detach().numpy().transpose()[0]
                m_hat_ji = m_hat_ji.detach().numpy().transpose()[0]

                data_ai["M_hat"] = m_hat_i
                data_aiaj["M_hat"] = m_hat_ij
                data_ajai["M_hat"] = m_hat_ji

                data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
                p_m_ai = data_ai["M_hat"].values
                data_aiaj.loc[data_aiaj["M"] == 0.0,"M_hat"] = np.ones(shape=data_aiaj.loc[data_aiaj["M"] == 0.0,"M_hat"].shape) - data_aiaj.loc[data_aiaj["M"] == 0.0,"M_hat"]
                p_m_aiaj = data_aiaj["M_hat"].values
                data_ajai.loc[data_ajai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ajai.loc[data_ajai["M"] == 0.0,"M_hat"].shape) - data_ajai.loc[data_ajai["M"] == 0.0,"M_hat"]
                p_m_ajai = data_ajai["M_hat"].values
                data_ai = data_ai.drop("M_hat", axis = 1)
                data_aiaj = data_aiaj.drop("M_hat", axis = 1)
                data_ajai = data_ajai.drop("M_hat", axis = 1)
       
                logits, y_ai = self.network(torch.tensor(data_ai.values).float())
                y_ai = y_ai.detach().numpy().transpose()[0]
                logits, y_aj = self.network(torch.tensor(data_aj.values).float())
                y_aj = y_aj.detach().numpy().transpose()[0]

            if y == 0.0:
                y_ai = np.ones(y_ai.size()) - y_ai
                y_aj = np.ones(y_aj.size()) - y_aj
                y = 1.0
                
            y_ai = np.nan_to_num(y_ai)
            y_aj = np.nan_to_num(y_aj)

            de = np.mean(y_aj * (np.round(y_aj)==y) * p_m_ajai) - np.mean(y_ai * (np.round(y_ai)==y) * p_m_ai)
            ie = np.mean(y_aj * (np.round(y_aj)==y) * (p_m_ajai - p_m_aj)) 
            se = (np.tensor(1) - p_ai/p_aj) * np.mean(y_ai * (np.round(y_ai)==y) * p_m_ai)
            

        return de, ie, se


    


class Bounds_regression:
    def __init__(self, network, full_data, density_estimator, sensitivity_param):
        if network == None:
            self.network = None
        else:
            self.network = network.eval()
        self.full_data = full_data
        self.data = full_data
        if density_estimator == None:
            self.density_estimator = None
        else:
            self.density_estimator = density_estimator.eval()
        self.sensitivity_param = sensitivity_param
        self.rel_freq_z = self.full_data[["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]].value_counts(normalize=True)[sorted(self.full_data[["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]].unique())]
        self.rel_freq_z = self.rel_freq_z.reset_index().sort_values(by=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"])
        self.rel_freq_a_z = self.full_data[["A",["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]]].value_counts(normalize=True)[sorted(self.full_data[["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]].unique())]
        self.rel_freq_a_z = self.rel_freq_a_z.reset_index().sort_values(by=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]) 
        self.order = self.full_data.sort_values(by=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"])[["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]].drop_duplicates()
        self.covariates = full_data.columns

    def build_score(self,a):
        prop_table_a  = self.rel_freq_a_z.loc[self.rel_freq_a_z["A"] == a].sort_values(by=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"])
        prop_table_a = prop_table_a.merge(self.order, on=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]).fillna(0).sort_values(by=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"])
        prop_a = prop_table_a["proportion"].values/self.rel_freq_z["proportion"].values
        s_plus_inverse = (1-1/self.sensitivity_param)*prop_a + 1/self.sensitivity_param 
        s_minus_inverse = (1-self.sensitivity_param)*prop_a + self.sensitivity_param
        c_plus = np.float(self.sensitivity_param/(1+self.sensitivity_param))
        c_minus = np.float(1/(1+self.sensitivity_param))
        return s_plus_inverse, s_minus_inverse, c_plus, c_minus

    def bounds_single(self, a):
        s_m_plus, s_m_minus, c_m_plus, c_m_minus = self.build_score(a) 
        
        temp_df = self.full_data.loc[self.full_data["A"]==a].drop_duplicates()

        with torch.no_grad():
            density_logits, density_estimates_all = self.density_estimator(torch.tensor(temp_df.drop("M", axis = 1).values, requires_grad=False).float())
            density_estimates_all = density_estimates_all.detach().numpy()

        temp_df["M_hat"] = density_estimates_all[temp_df["M"].values]
        temp_df = temp_df.drop_duplicates()

        F_m = np.empty(shape=(1,len(temp_df["Z"].unique())))
        for z in temp_df["Z"].unique().sort():
            F_m_z = np.cumsum(temp_df.loc[temp_df["Z"]==z].sort_values(by="M")["M_hat"].values)
            F_m = np.append(F_m, F_m_z.reshape(1,len(F_m_z)), axis = 0)
        F_m = F_m.transpose()

        ind = 0
        Q_plus = np.zeros(shape=np.shape(self.order))
        Q_minus = np.zeros(shape=np.shape(self.order))

        for m in np.unique(self.full_data["M"]): 
            maz = self.full_data.loc[self.full_data["M"] == m,["M", "A", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]].drop_duplicates().copy()
            maz =  maz.loc[maz["A"] == a]
            maz = maz[self.covariates]
            maz.merge(self.order, on =["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"], how = "outer").sort_values(by="Z") 
            maz["A"] = maz["A"].fillna(a)
            maz["M"] = maz["M"].fillna(m)
          
            with torch.no_grad():
                logits, exp_maz = self.network(torch.tensor(maz.values).float())
                exp_maz = exp_maz.detach().numpy()

                density_logits, p_m_az = self.density_estimator(torch.tensor(maz.drop(["M"], axis = 1).values, requires_grad=False).float())
                p_m_az = p_m_az.detach().numpy()[:,m]              
                p_m_az = np.nan_to_num(p_m_az)

            if ind == 0:
                p_plus_m_az = (F_m[ind]<c_m_plus) * (s_m_plus * p_m_az) + (F_m[ind]>=c_m_plus) * (s_m_plus * c_m_plus + s_m_minus * (F_m[ind] - c_m_plus))
                p_minus_m_az = (F_m[ind]<c_m_minus) * (s_m_minus * p_m_az) + (F_m[ind]>=c_m_minus) * (s_m_minus * c_m_minus + s_m_plus * (F_m[ind] - c_m_minus))

            elif ind > 0:
                p_plus_m_az = (F_m[ind]<c_m_plus) * (s_m_plus * p_m_az) + (F_m[ind-1]>c_m_plus)* (s_m_minus * p_m_az) + (F_m[ind]>=c_m_plus) * (F_m[ind-1]<=c_m_plus) * (s_m_plus * (c_m_plus - F_m[ind -1]) + s_m_minus * (F_m[ind] - c_m_plus))               
                p_minus_m_az = (F_m[ind]<c_m_minus) * (s_m_minus * p_m_az) + (F_m[ind-1]>c_m_minus) * (s_m_plus * p_m_az) + (F_m[ind]>=c_m_minus) * (F_m[ind-1]<=c_m_minus) * (s_m_minus * (c_m_minus - F_m[ind -1]) + s_m_plus * (F_m[ind] - c_m_minus))

            ind = ind + 1

            Q_plus = Q_plus + (p_plus_m_az * exp_maz)
            Q_minus = Q_minus + (p_minus_m_az * exp_maz)

        ub = np.mean(Q_plus * self.rel_freq_z["proportion"])
        lb = np.mean(Q_minus * self.rel_freq_z["proportion"])
        return ub, lb
    

    def bounds_double(self, a_i, a_j):
        """
        Calculate bounds of the form \sum_z Q(z,(a_i, a_j),S)*P(Z=z),
        where Q(z,(a_i, a_j),S) = \sum_m P(Y=y| M=m, Z=z, do(A=a_i))*P(M=m|Z=z, do(A=a_j)) 
        """

        s_m_plus, s_m_minus, c_m_plus, c_m_minus = self.build_score(a_j) 

        temp_df = self.full_data.loc[self.full_data["A"]==a_j].drop_duplicates()

        with torch.no_grad():
            density_logits, density_estimates_all = self.density_estimator(torch.tensor(temp_df.drop("M", axis = 1).values, requires_grad=False).float())
            density_estimates_all = density_estimates_all.detach().numpy()

        temp_df["M_hat"] = density_estimates_all[temp_df["M"].values]
        temp_df = temp_df.drop_duplicates()

        F_m = np.empty(shape=(1,len(temp_df["Z"].unique())))
        for z in temp_df["Z"].unique().sort():
            F_m_z = np.cumsum(temp_df.loc[temp_df["Z"]==z].sort_values(by="M")["M_hat"].values)
            F_m = np.append(F_m, F_m_z.reshape(1,len(F_m_z)), axis = 0)
        F_m = F_m.transpose()

        ind = 0
        Q_plus = np.zeros(shape=np.shape(self.order))
        Q_minus = np.zeros(shape=np.shape(self.order))
        
        for m in np.unique(self.full_data["M"]): 
            maz = self.full_data.loc[self.full_data["M"] == m,["M", "A", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]].drop_duplicates().copy()
            y_maz =  maz.loc[maz["A"] == a_i]
            y_maz = y_maz[self.covariates]
            y_maz.merge(self.order, on =["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"], how = "outer").sort_values(by=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"])
            y_maz["A"] = y_maz["A"].fillna(a_i)
            y_maz["M"] = y_maz["M"].fillna(m)

            m_za_j = maz.loc[maz["A"] == a_j]
            m_za_j = m_za_j[self.covariates]
            m_za_j.merge(self.order, on =["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"], how = "outer").sort_values(by=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"])
            m_za_j["A"] = m_za_j["A"].fillna(a_j)
            m_za_j["M"] = m_za_j["M"].fillna(m)

            with torch.no_grad():
                logits, exp_y_maz = self.network(torch.tensor(y_maz.values).float())
                exp_y_maz = exp_y_maz.detach().numpy()

                if len(self.order) >= 3: 
                    raise NotImplementedError
                
                logits, p_m_az = self.density_estimator(torch.tensor(m_za_j.drop(["M"], axis = 1).values, requires_grad=False).float())
                p_m_az = p_m_az.detach().numpy()

            if ind == 0:
                p_plus_m_az = (np.repeat(F_m[ind], len(self.order))<c_m_plus) * (s_m_plus * p_m_az) + (np.repeat(F_m[ind], len(self.order))>=c_m_plus) * (s_m_plus * c_m_plus  + s_m_minus * (F_m[ind] - c_m_plus))
                p_minus_m_az = (np.repeat(F_m[ind], len(self.order))<c_m_minus) * (s_m_minus * p_m_az) + (np.repeat(F_m[ind], len(self.order))>=c_m_minus) * (s_m_minus * c_m_minus + s_m_plus * (F_m[ind] - c_m_minus))

            else:
                p_plus_m_az = (np.repeat(F_m[ind], len(self.order))<c_m_plus) * (s_m_plus * p_m_az) + (np.repeat(F_m[ind-1], len(self.order))>c_m_plus)* (s_m_minus * p_m_az) + (np.repeat(F_m[ind], len(self.order))>=c_m_plus) * (np.repeat(F_m[ind-1], len(self.order))<=c_m_plus) * (s_m_plus * (c_m_plus - F_m[ind -1]) + s_m_minus * (F_m[ind] - c_m_plus))               
                p_minus_m_az = (np.repeat(F_m[ind], len(self.order))<c_m_minus) * (s_m_minus * p_m_az) + (np.repeat(F_m[ind-1], len(self.order))>c_m_minus)* (s_m_plus * p_m_az) + (np.repeat(F_m[ind], len(self.order))>=c_m_minus) * (np.repeat(F_m[ind-1], len(self.order))<=c_m_minus) * (s_m_minus * (c_m_minus - F_m[ind -1]) + s_m_plus * (F_m[ind] - c_m_minus))

            Q_plus = Q_plus + (p_plus_m_az * exp_y_maz)
            Q_minus = Q_minus + (p_minus_m_az * exp_y_maz)
        
            ind = ind + 1

        ub = np.mean(Q_plus * self.rel_freq_z["proportion"])
        lb = np.mean(Q_minus * self.rel_freq_z["proportion"])
        return ub, lb



class Ctf_effects_regression:
    def __init__(self, network, full_data, task, sensitivity_param=None, density_estimator=None):
        if network == None:
            self.network = None
        else:
            self.network = network.eval()
        self.full_data = full_data
        self.task = task
        if density_estimator==None:
            self.density_estimator = None
        else:
            self.density_estimator = density_estimator.eval()
        self.sensitivity_param = sensitivity_param
        self.bounds = Bounds_regression(network=self.network, full_data = self.full_data.drop("Y", axis = 1), density_estimator=self.density_estimator, sensitivity_param=self.sensitivity_param)
        self.covariates = pd.merge(full_data["M"].drop_duplicates(), self.full_data[["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]].drop_duplicates(), how = "cross")
        self.rel_freq_z = self.full_data[["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]].value_counts(normalize=True)[sorted(self.full_data[["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]].unique())]
        self.rel_freq_z = self.rel_freq_z.reset_index()   
        self.column_order = full_data.drop("Y", axis = 1).columns   

    def DE_binary(self, a_i, a_j):

        ub_double, lb_double = self.bounds.bounds_double(a_j, a_i) 

        data_ai = self.full_data.loc[self.full_data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]).drop("Y", axis = 1).drop_duplicates()[self.column_order]
        data_aj = self.full_data.loc[self.full_data["A"] == a_j].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]).drop("Y", axis = 1).drop_duplicates()[self.column_order]

        p_ai = np.sum(self.full_data["A"] == a_i)/self.full_data.shape[0]
        p_aj = np.sum(self.full_data["A"] == a_j)/self.full_data.shape[0]
        p_z = data_ai.merge(self.rel_freq_z, how="outer", on= [["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]]).drop_duplicates().sort_values(by=['M', "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"])["proportion"]

        with torch.no_grad():
            logits, m_hat = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
            m_hat = m_hat.detach().numpy().transpose()[0]
            data_ai["M_hat"] = m_hat
            data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
            p_m_ai = data_ai["M_hat"].values
            data_ai = data_ai.drop("M_hat", axis = 1)

            logits, y_ai = self.network(torch.tensor(data_ai.values).float())
            y_ai = y_ai.detach().numpy().transpose()[0]
            logits, y_mza_j = self.network(torch.tensor(data_aj.values).float())
            y_mza_j = y_mza_j.detach().numpy().transpose()[0]

            y_ai = np.nan_to_num(y_ai)
            y_mza_j = np.nan_to_num(y_mza_j)

            DE_ub = 1/p_ai * ub_double - p_aj/p_ai * np.mean(y_mza_j * p_m_ai * p_z) - np.mean(y_ai *  p_m_ai * p_z)
            DE_lb = 1/p_ai * lb_double - p_aj/p_ai * np.mean(y_mza_j * p_m_ai * p_z) - np.mean(y_ai * p_m_ai * p_z)

        return DE_ub, DE_lb


    def IE_binary(self, a_i, a_j):
        """
        Function to calculate bounds for  IE_{a_i, a_j}(y|a_j) for binary A
        """
        ub_double, lb_double = self.bounds.bounds_double(a_i, a_j)
        ub_single, lb_single = self.bounds.bounds_single(a_i)

        data_ai = self.full_data.loc[self.full_data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', 'Z']).sort_values(by=['M', 'Z']).drop("Y", axis = 1).drop_duplicates()[self.column_order]
        data_aj = self.full_data.loc[self.full_data["A"] == a_j].merge(self.covariates, how="outer", on= ['M', 'Z']).sort_values(by=['M', 'Z']).drop("Y", axis = 1).drop_duplicates()[self.column_order]

        p_ai = np.sum(self.full_data["A"] == a_i)/self.full_data.shape[0]
        p_aj = np.sum(self.full_data["A"] == a_j)/self.full_data.shape[0]
        p_z = data_ai.merge(self.rel_freq_z, how="outer", on= ['Z']).drop_duplicates().sort_values(by=['M', 'Z'])["proportion"]

        with torch.no_grad():
            logits, m_hat_i = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
            logits, m_hat_j = self.density_estimator(torch.tensor(data_aj.drop("M", axis = 1).values).float())
            m_hat_i = m_hat_i.detach().numpy().transpose()[0]            
            m_hat_j = m_hat_j.detach().numpy().transpose()[0]
            data_ai["M_hat"] = m_hat_i[:,1]
            data_aj["M_hat"] = m_hat_j[:,1]
            data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
            p_m_ai = data_ai["M_hat"].values
            data_aj.loc[data_aj["M"] == 0.0,"M_hat"] = np.ones(shape=data_aj.loc[data_aj["M"] == 0.0,"M_hat"].shape) - data_aj.loc[data_aj["M"] == 0.0,"M_hat"]
            p_m_aj = data_aj["M_hat"].values
            data_ai = data_ai.drop("M_hat", axis = 1)
            data_aj = data_aj.drop("M_hat", axis = 1)

            logits, y_mza = self.network(torch.tensor(data_ai.values).float())
            y_mza = y_mza.detach().numpy().transpose()[0]

            IE_ub = 1/p_aj * (ub_double-lb_single) - p_ai/p_aj * np.mean(y_mza * (p_m_aj - p_m_ai) * p_z)
            IE_lb = 1/p_aj * (lb_double-ub_single) - p_ai/p_aj * np.mean(y_mza * (p_m_aj - p_m_ai) * p_z)

        return IE_ub, IE_lb


    def SE_binary(self, a_i, a_j):
        """
        Function to calculate bounds for  SE_{a_i, a_j}(y) for binary A
        """
        ub_single, lb_single = self.bounds.bounds_single(a_i)    

        data_ai = self.full_data.loc[self.full_data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', 'Z']).sort_values(by=['M', 'Z']).drop("Y", axis = 1).drop_duplicates()[self.column_order]

        p_ai = np.sum(self.full_data["A"] == a_i)/self.full_data.shape[0]
        p_aj = np.sum(self.full_data["A"] == a_j)/self.full_data.shape[0]
        p_z = data_ai.merge(self.rel_freq_z, how="outer", on= ['Z']).drop_duplicates().sort_values(by=['M', 'Z'])["proportion"]     

        with torch.no_grad():
            logits, m_hat_i = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
            m_hat_i = m_hat_i.detach().numpy().transpose()[0]
            data_ai["M_hat"] = m_hat_i
            data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
            p_m_ai = data_ai["M_hat"].values
            data_ai = data_ai.drop("M_hat", axis = 1)

            logits, y_ai = self.network(torch.tensor(data_ai.values).float())
            y_ai = y_ai.detach().numpy().transpose()[0]

            SE_ub = 1/p_aj * ub_single - (1 + p_ai/p_aj) * np.mean(y_ai * p_m_ai * p_z)
            SE_lb = 1/p_aj * lb_single - (1 + p_ai/p_aj) * np.mean(y_ai * p_m_ai * p_z)

        return SE_ub, SE_lb

















#------------------------------------------------------------------torch implementation--------------------------------------------------------------------

class Bounds_binary_torch:
    def __init__(self, network, full_data, density_estimator, sensitivity_param, task, propensity_model = None):
        self.network = network
        self.data = full_data
        self.density_estimator = density_estimator
        self.sensitivity_param = sensitivity_param
        self.task = task
        self.propensity_model = propensity_model
        if task == "multiclass":
            self.rel_freq_z = self.data.value_counts(normalize=True, subset = ["Z1", "Z2", "Z3", "Z4"])
            self.rel_freq_z = self.rel_freq_z.reset_index().sort_values(by=["Z1", "Z2", "Z3", "Z4"])
            self.rel_freq_a_z = self.data.value_counts(normalize=True, subset = ["A", "Z1", "Z2", "Z3", "Z4"])
            self.rel_freq_a_z = self.rel_freq_a_z.reset_index().sort_values(by=["Z1", "Z2", "Z3", "Z4"]) 
            self.order = pd.DataFrame(list(product(self.data["Z1"].unique(), self.data["Z2"].unique(), self.data["Z3"].unique(), self.data["Z4"].unique())), columns=(["Z1", "Z2", "Z3", "Z4"]))
        elif task == "binary":        
            self.rel_freq_z = self.data["Z"].value_counts(normalize=True)[sorted(self.data["Z"].unique())]
            self.rel_freq_z =  self.rel_freq_z.reset_index()
            self.rel_freq_a_z = self.data[["A","Z"]].value_counts(normalize=True)[sorted(self.data["Z"].unique())]
            self.rel_freq_a_z = self.rel_freq_a_z.reset_index().sort_values(by="Z") 
            self.order = pd.DataFrame(self.data["Z"].drop_duplicates().sort_values())
        else:
            pass
        self.covariates = full_data.columns

    def build_score(self,a, data):
        if self.task == "multiclass":
            prop_table_a  = self.rel_freq_a_z.loc[self.rel_freq_a_z["A"] == a].sort_values(by=["Z1", "Z2", "Z3", "Z4"])
            prop_table_a = prop_table_a.merge(self.order, on=["Z1", "Z2", "Z3", "Z4"], how = "outer").fillna(0).sort_values(by=["Z1", "Z2", "Z3", "Z4"])
            prop_a = prop_table_a["proportion"].values/self.rel_freq_z["proportion"].values
        elif self.task == "binary": 
            prop_table_a  = self.rel_freq_a_z.loc[self.rel_freq_a_z["A"] == a].sort_values(by="Z")
            prop_table_a = prop_table_a.merge(self.order, on="Z").fillna(0).sort_values(by="Z")
            prop_a = prop_table_a["proportion"].values/self.rel_freq_z["proportion"].values
        else:
            logits, prop_a = self.propensity_model(data.float())
            prop_a = prop_a.detach().numpy().transpose()[0]
            prop_table_a = None

        s_plus = 1/((1-1/self.sensitivity_param)*prop_a + 1/self.sensitivity_param)
        s_minus = 1/((1-self.sensitivity_param)*prop_a + self.sensitivity_param)
        c_plus = np.float(self.sensitivity_param/(1+self.sensitivity_param))
        c_minus = np.float(1/(1+self.sensitivity_param))
        return s_plus, s_minus, c_plus, c_minus, prop_table_a 

    def bounds_single(self, a, y):
        """
        Calculate bounds of the form \sum_z Q(z,a_i,S)*P(Z=z),
        where Q(z,a_i,S) = \sum_m P(Y=y| M=m, Z=z, do(A=a_i))*P(M=m|Z=z, do(A=a_i)) 
        """

        f_m_in = self.order.copy()
        f_m_in["A"] = np.repeat(a, self.order.shape[0])
        f_m_in["M"] = np.repeat(np.nan, self.order.shape[0])
        f_m_in = f_m_in[self.covariates]   
        with torch.no_grad():
            logits, f_m_z_array = self.density_estimator(torch.tensor(f_m_in.drop("M", axis = 1).values, requires_grad=False).float())
        F_m = np.cumsum(f_m_z_array.detach().numpy(), axis = 1).transpose()[0]
        
        ind = 0
        Q_plus = torch.zeros(np.shape(self.order)[0])
        Q_minus = torch.zeros(np.shape(self.order)[0])

        for m in sorted(np.unique(self.data["M"])): 
            if self.task == "multiclass":
                y_maz = self.data.loc[self.data["M"] == m,["M", "A", "Z1", "Z2", "Z3", "Z4"]].drop_duplicates().copy()
                y_maz = y_maz.loc[y_maz["A"] == a]
                y_maz= y_maz.merge(self.order.drop(["M", "A"], axis = 1), on =["Z1", "Z2", "Z3", "Z4"], how = "outer").sort_values(by=["Z1", "Z2", "Z3", "Z4"]) 
                y_maz = y_maz[self.covariates]
                y_maz["A"] = y_maz["A"].fillna(a)
                y_maz["M"] = y_maz["M"].fillna(m)
                p_y_maz = self.network(torch.tensor(y_maz.values).float())[:, int(y)] 
                logits, p_m_az = self.density_estimator(torch.tensor(y_maz.drop(["M"], axis = 1).values).float())
                p_m_az = p_m_az.detach().numpy()[:, int(m)]   

            else:
                y_maz = self.data.loc[self.data["M"] == m,["M", "A", "Z"]].drop_duplicates().copy()
                y_maz = y_maz.loc[y_maz["A"] == a]
                y_maz = y_maz[self.covariates]
                y_maz = y_maz.merge(self.order, on ="Z", how = "outer").sort_values(by="Z")
                logits, p_y_maz = self.network(torch.tensor(y_maz.values).float())

                if y == 0.0:
                    p_y_maz = torch.ones(p_y_maz.size()) - p_y_maz
                    y = 1.0
                with torch.no_grad():
                    density_logits, p_m_az = self.density_estimator(torch.tensor(y_maz.drop(["M"], axis = 1).values).float())
                    p_m_az = p_m_az.detach().numpy().transpose()[0]
                f_m = p_m_az.copy()                
                if m == 0:
                    p_m_az = np.ones(len(p_m_az)) - p_m_az
                p_m_az = np.nan_to_num(p_m_az)

            s_m_plus, s_m_minus, c_m_plus, c_m_minus, prop_table_a = self.build_score(a, torch.tensor(y_maz.drop(["M", "A"], axis = 1).values)) 
            F_m = np.concatenate((np.expand_dims((np.ones(len(f_m)) - f_m), axis=0), np.expand_dims(np.ones(len(f_m)), axis=0)), axis = 0)

            if ind == 0:
                p_plus_m_az = torch.tensor((F_m[ind]<c_m_plus) * (1/s_m_plus * p_m_az) + (F_m[ind]>=c_m_plus) * (1/s_m_plus * c_m_plus + 1/s_m_minus * (F_m[ind] - c_m_plus)))
                p_minus_m_az = torch.tensor((F_m[ind]<c_m_minus) * (1/s_m_minus * p_m_az) + (F_m[ind]>=c_m_minus) * (1/s_m_minus * c_m_minus + 1/s_m_plus * (F_m[ind] - c_m_minus)))
            else:
                p_plus_m_az = torch.tensor((F_m[ind]<c_m_plus) * (1/s_m_plus * p_m_az) + (F_m[ind-1]>c_m_plus)* (1/s_m_minus * p_m_az) + (F_m[ind]>=c_m_plus) * (F_m[ind-1]<=c_m_plus) * (1/s_m_plus * (c_m_plus - F_m[ind -1]) + 1/s_m_minus * (F_m[ind] - c_m_plus)))
                p_minus_m_az = torch.tensor((F_m[ind]<c_m_minus) * (1/s_m_minus * p_m_az) + (F_m[ind-1]>c_m_minus)* (1/s_m_plus * p_m_az) + (F_m[ind]>=c_m_minus) * (F_m[ind-1]<=c_m_minus) * (1/s_m_minus * (c_m_minus - F_m[ind -1]) + 1/s_m_plus * (F_m[ind] - c_m_minus)))

            if self.task == "multiclass":            
                Q_plus = Q_plus + (p_plus_m_az * p_y_maz * (torch.argmax(p_y_maz)==torch.tensor(y)))
                Q_minus = Q_minus + (p_minus_m_az * p_y_maz * (torch.argmax(p_y_maz)==torch.tensor(y)))            
            else:
                Q_plus = Q_plus + (p_plus_m_az * p_y_maz * (torch.round(p_y_maz)==torch.tensor(y)))
                Q_minus = Q_minus + (p_minus_m_az * p_y_maz * (torch.round(p_y_maz)==torch.tensor(y)))
            ind = ind + 1

        ub = torch.sum(Q_plus * torch.tensor(self.rel_freq_z["proportion"].values))
        lb = torch.sum(Q_minus * torch.tensor(self.rel_freq_z["proportion"].values))

        return ub.requires_grad_(), lb.requires_grad_()
    

    def bounds_single_continuous(self, a, y):
        """
        Calculate bounds of the form \sum_z Q(z,a_i,S)*P(Z=z),
        where Q(z,a_i,S) = \sum_m P(Y=y| M=m, Z=z, do(A=a_i))*P(M=m|Z=z, do(A=a_i)) 
        """
        
        ind = 0
        Q_plus = torch.zeros(1)
        Q_minus = torch.zeros(1)

        for m in sorted(np.unique(self.data["M"])): 
            y_maz = self.data.loc[self.data["M"] == m]
            y_maz = y_maz.loc[y_maz["A"] == a]
            y_maz = y_maz[self.covariates]
            logits, p_y_maz = self.network(torch.tensor(y_maz.values).float())

            if y == 0.0:
                p_y_maz = torch.ones(p_y_maz.size()) - p_y_maz
                y = 1.0
            density_logits, p_m_az = self.density_estimator(torch.tensor(y_maz.drop(["M"], axis = 1).values).float())
            p_m_az = p_m_az.detach().numpy().transpose()[0]
            f_m = p_m_az.copy()                
            if m == 0:
                p_m_az = np.ones(len(p_m_az)) - p_m_az
            p_m_az = np.nan_to_num(p_m_az)

            s_m_plus, s_m_minus, c_m_plus, c_m_minus, prop_table_a = self.build_score(a, torch.tensor(y_maz.drop(["M", "A"], axis = 1).values)) 
            F_m = np.concatenate((np.expand_dims((np.ones(len(f_m)) - f_m), axis=0), np.expand_dims(np.ones(len(f_m)), axis=0)), axis = 0)

            if ind == 0:
                p_plus_m_az = torch.tensor((F_m[ind]<c_m_plus) * (1/s_m_plus * p_m_az) + (F_m[ind]>=c_m_plus) * (1/s_m_plus * c_m_plus + 1/s_m_minus * (F_m[ind] - c_m_plus)))
                p_minus_m_az = torch.tensor((F_m[ind]<c_m_minus) * (1/s_m_minus * p_m_az) + (F_m[ind]>=c_m_minus) * (1/s_m_minus * c_m_minus + 1/s_m_plus * (F_m[ind] - c_m_minus)))
            else:
                p_plus_m_az = torch.tensor((F_m[ind]<c_m_plus) * (1/s_m_plus * p_m_az) + (F_m[ind-1]>c_m_plus)* (1/s_m_minus * p_m_az) + (F_m[ind]>=c_m_plus) * (F_m[ind-1]<=c_m_plus) * (1/s_m_plus * (c_m_plus - F_m[ind -1]) + 1/s_m_minus * (F_m[ind] - c_m_plus)))
                p_minus_m_az = torch.tensor((F_m[ind]<c_m_minus) * (1/s_m_minus * p_m_az) + (F_m[ind-1]>c_m_minus)* (1/s_m_plus * p_m_az) + (F_m[ind]>=c_m_minus) * (F_m[ind-1]<=c_m_minus) * (1/s_m_minus * (c_m_minus - F_m[ind -1]) + 1/s_m_plus * (F_m[ind] - c_m_minus)))

            Q_plus = Q_plus + torch.mean(p_plus_m_az * p_y_maz * (torch.round(p_y_maz)==torch.tensor(y)))
            Q_minus = Q_minus + torch.mean(p_minus_m_az * p_y_maz * (torch.round(p_y_maz)==torch.tensor(y)))

            ind = ind + 1

        return Q_plus.requires_grad_(), Q_minus.requires_grad_()
    
    

    def bounds_double(self, a_i, a_j, y):
        """
        Calculate bounds of the form \sum_z Q(z,(a_i, a_j),S)*P(Z=z),
        where Q(z,(a_i, a_j),S) = \sum_m P(Y=y| M=m, Z=z, do(A=a_i))*P(M=m|Z=z, do(A=a_j)) 
        """

        f_m_in = self.order.copy()
        f_m_in["A"] = np.repeat(a_j, self.order.shape[0])
        f_m_in["M"] = np.repeat(np.nan, self.order.shape[0])
        f_m_in = f_m_in[self.covariates]   

        logits, f_m_z_array = self.density_estimator(torch.tensor(f_m_in.drop("M", axis = 1).values, requires_grad=False).float())
        F_m = np.cumsum(f_m_z_array.detach().numpy(), axis = 1).transpose()[0]
        
        ind = 0
        Q_plus = torch.zeros(np.shape(self.order)[0])
        Q_minus = torch.zeros(np.shape(self.order)[0])

        for m in sorted(np.unique(self.data["M"])): 

            if self.task == "multiclass":
                maz = self.data.loc[self.data["M"] == m,["M", "A", "Z1", "Z2", "Z3", "Z4"]].drop_duplicates().copy()
                y_maz = maz.loc[maz["A"] == a_i]
                y_maz= y_maz.merge(self.order.drop(["M", "A"], axis = 1), on =["Z1", "Z2", "Z3", "Z4"], how = "outer").sort_values(by=["Z1", "Z2", "Z3", "Z4"]) 
                y_maz = y_maz[self.covariates]
                y_maz["A"] = y_maz["A"].fillna(a_i)
                y_maz["M"] = y_maz["M"].fillna(m)

                m_za_j = maz.loc[maz["A"] == a_j]
                m_za_j = m_za_j.merge(self.order.drop(["M", "A"], axis = 1), on =["Z1", "Z2", "Z3", "Z4"], how = "outer").sort_values(by=["Z1", "Z2", "Z3", "Z4"])
                m_za_j = m_za_j[self.covariates]
                m_za_j["A"] = m_za_j["A"].fillna(a_j)
                m_za_j["M"] = m_za_j["M"].fillna(m)

                p_y_maz = self.network(torch.tensor(y_maz.values).float())[:, int(y)]
                with torch.no_grad():
                    logits, p_m_az = self.density_estimator(torch.tensor(m_za_j.drop(["M"], axis = 1).values, requires_grad=False).float())
                    p_m_az = p_m_az.detach().numpy()[:, int(m)]   
            
            else:
                maz = self.data.loc[self.data["M"] == m,["M", "A", "Z"]].drop_duplicates().copy()
                y_maz =  maz.loc[maz["A"] == a_i]
                y_maz = y_maz[self.covariates]                
                y_maz = y_maz.merge(self.order, on ="Z", how = "outer").sort_values(by="Z")

                m_za_j = maz.loc[maz["A"] == a_j]
                m_za_j = m_za_j[self.covariates]

                logits, p_y_maz = self.network(torch.tensor(y_maz.drop_duplicates().values).float())
                if y == 0.0:
                    p_y_maz = torch.ones(p_y_maz.size()) - p_y_maz
                    y = 1.0

                logits, p_m_az = self.density_estimator(torch.tensor(m_za_j.drop(["M"], axis = 1).values).float())
                p_m_az = p_m_az.detach().numpy().transpose()[0]
                f_m = p_m_az.copy()
                if m == 0:
                    p_m_az = np.ones(len(p_m_az)) - p_m_az
                p_m_az = np.nan_to_num(p_m_az)

            s_m_plus, s_m_minus, c_m_plus, c_m_minus, prop_table_a = self.build_score(a_j, torch.tensor(m_za_j.drop(["M", "A"], axis = 1).values)) 

            if ind == 0:
                p_plus_m_az = torch.tensor((F_m[ind]<c_m_plus) * (1/s_m_plus * p_m_az) + (F_m[ind]>=c_m_plus) * (1/s_m_plus * c_m_plus + 1/s_m_minus * (F_m[ind] - c_m_plus)))
                p_minus_m_az = torch.tensor((F_m[ind]<c_m_minus) * (1/s_m_minus * p_m_az) + (F_m[ind]>=c_m_minus) * (1/s_m_minus * c_m_minus + 1/s_m_plus * (F_m[ind] - c_m_minus)))
            
            else:
                p_plus_m_az = torch.tensor((F_m[ind]<c_m_plus) * (1/s_m_plus * p_m_az) + (F_m[ind-1]>c_m_plus)* (1/s_m_minus * p_m_az) + (F_m[ind]>=c_m_plus) * (F_m[ind-1]<=c_m_plus) * (1/s_m_plus * (c_m_plus - F_m[ind -1]) + 1/s_m_minus * (F_m[ind] - c_m_plus)))
                p_minus_m_az = torch.tensor((F_m[ind]<c_m_minus) * (1/s_m_minus * p_m_az) + (F_m[ind-1]>c_m_minus)* (1/s_m_plus * p_m_az) + (F_m[ind]>=c_m_minus) * (F_m[ind-1]<=c_m_minus) * (1/s_m_minus * (c_m_minus - F_m[ind -1]) + 1/s_m_plus * (F_m[ind] - c_m_minus)))

            if self.task == "multiclass":      
                Q_plus = Q_plus + (p_plus_m_az * p_y_maz * (torch.argmax(p_y_maz)==torch.tensor(y)))
                Q_minus = Q_minus + (p_minus_m_az * p_y_maz* (torch.argmax(p_y_maz)==torch.tensor(y)))               
            else:                                   
                Q_plus = Q_plus + (p_plus_m_az * p_y_maz * (torch.round(p_y_maz)==torch.tensor(y)))
                Q_minus = Q_minus + (p_minus_m_az * p_y_maz* (torch.round(p_y_maz)==torch.tensor(y)))
            
            ind = ind + 1

        ub = torch.sum(Q_plus * torch.tensor(self.rel_freq_z["proportion"].values))
        lb = torch.sum(Q_minus * torch.tensor(self.rel_freq_z["proportion"].values))

        return ub.requires_grad_(), lb.requires_grad_()


    def bounds_double_continuous(self, a_i, a_j, y):
        """
        Calculate bounds of the form \sum_z Q(z,(a_i, a_j),S)*P(Z=z),
        where Q(z,(a_i, a_j),S) = \sum_m P(Y=y| M=m, Z=z, do(A=a_i))*P(M=m|Z=z, do(A=a_j)) 
        """  
        ind = 0
        Q_plus = torch.zeros(1)
        Q_minus = torch.zeros(1)

        for m in sorted(np.unique(self.data["M"])): 
            maz = self.data.loc[self.data["M"] == m]
            y_maz =  maz.loc[maz["A"] == a_i]
            y_maz = y_maz[self.covariates]

            m_za_j = y_maz
            m_za_j["A"] = np.repeat(a_j, len(m_za_j["A"])) 

            logits, p_y_maz = self.network(torch.tensor(y_maz.drop_duplicates().values).float())
            if y == 0.0:
                p_y_maz = torch.ones(p_y_maz.size()) - p_y_maz
                y = 1.0
            
            logits, p_m_az = self.density_estimator(torch.tensor(m_za_j.drop(["M"], axis = 1).values).float())
            p_m_az = p_m_az.detach().numpy().transpose()[0]
            f_m = p_m_az.copy()
            if m == 0:
                p_m_az = np.ones(len(p_m_az)) - p_m_az
            p_m_az = np.nan_to_num(p_m_az)

            s_m_plus, s_m_minus, c_m_plus, c_m_minus, prop_table_a = self.build_score(a_j, torch.tensor(m_za_j.drop(["M", "A"], axis = 1).values)) 

            F_m = np.concatenate((np.expand_dims((np.ones(len(f_m)) - f_m), axis=0), np.expand_dims(np.ones(len(f_m)), axis=0)), axis = 0)

            if ind == 0:
                p_plus_m_az = torch.tensor((F_m[ind]<c_m_plus) * (1/s_m_plus * p_m_az) + (F_m[ind]>=c_m_plus) * (1/s_m_plus * c_m_plus + 1/s_m_minus * (F_m[ind] - c_m_plus)))
                p_minus_m_az = torch.tensor((F_m[ind]<c_m_minus) * (1/s_m_minus * p_m_az) + (F_m[ind]>=c_m_minus) * (1/s_m_minus * c_m_minus + 1/s_m_plus * (F_m[ind] - c_m_minus)))
            
            else:
                p_plus_m_az = torch.tensor((F_m[ind]<c_m_plus) * (1/s_m_plus * p_m_az) + (F_m[ind-1]>c_m_plus)* (1/s_m_minus * p_m_az) + (F_m[ind]>=c_m_plus) * (F_m[ind-1]<=c_m_plus) * (1/s_m_plus * (c_m_plus - F_m[ind -1]) + 1/s_m_minus * (F_m[ind] - c_m_plus)))
                p_minus_m_az = torch.tensor((F_m[ind]<c_m_minus) * (1/s_m_minus * p_m_az) + (F_m[ind-1]>c_m_minus)* (1/s_m_plus * p_m_az) + (F_m[ind]>=c_m_minus) * (F_m[ind-1]<=c_m_minus) * (1/s_m_minus * (c_m_minus - F_m[ind -1]) + 1/s_m_plus * (F_m[ind] - c_m_minus)))

            if self.task == "multiclass":      
                Q_plus = Q_plus + (p_plus_m_az * p_y_maz * (torch.argmax(p_y_maz)==torch.tensor(y)))
                Q_minus = Q_minus + (p_minus_m_az * p_y_maz* (torch.argmax(p_y_maz)==torch.tensor(y)))               
            else:          
                Q_plus = Q_plus + torch.mean(p_plus_m_az * p_y_maz * (torch.round(p_y_maz)==torch.tensor(y)))
                Q_minus = Q_minus + torch.mean(p_minus_m_az * p_y_maz* (torch.round(p_y_maz)==torch.tensor(y)))
                                
            ind = ind + 1

        return Q_plus.requires_grad_(), Q_minus.requires_grad_()




class Ctf_effects_binary_torch:
    def __init__(self, network, data, task, density_estimator, sensitivity_param, propensity_model = None):
        self.network = network
        self.data = data
        self.task = task
        self.density_estimator = density_estimator
        self.sensitivity_param = sensitivity_param
        self.propensity_model = propensity_model
        self.bounds = Bounds_binary_torch(network=self.network, full_data = self.data.drop("Y", axis = 1), sensitivity_param=self.sensitivity_param, density_estimator=density_estimator, task=task, propensity_model=propensity_model)
        if task == "multiclass":
            self.covariates = pd.merge(data["M"].drop_duplicates(), self.data[["Z1", "Z2", "Z3", "Z4"]].drop_duplicates(), how = "cross")
            self.rel_freq_z = self.data.value_counts(normalize=True, subset = ["Z1", "Z2", "Z3", "Z4"])
            self.rel_freq_z = self.rel_freq_z.reset_index().sort_values(by=["Z1", "Z2", "Z3", "Z4"])
            self.order = self.data.sort_values(by=["Z1", "Z2", "Z3", "Z4"])[["Z1", "Z2", "Z3", "Z4"]].drop_duplicates()
            self.rel_freq_y = self.data["Y"].value_counts(normalize=True)
            self.rel_freq_y = self.rel_freq_y.reset_index()
        else:
            self.covariates = pd.merge(data["M"].drop_duplicates(), self.data["Z"].drop_duplicates(), how = "cross")
            self.order = self.data.sort_values(by=["Z"])["Z"].drop_duplicates()
            self.covariates = pd.merge(data["M"].drop_duplicates(), self.data["Z"].drop_duplicates(), how = "cross")
            self.rel_freq_z = self.data["Z"].value_counts(normalize=True)[sorted(self.data["Z"].unique())]
            self.rel_freq_z = self.rel_freq_z.reset_index()
            self.rel_freq_y = self.data["Y"].value_counts(normalize=True)[sorted(self.data["Y"].unique())]
            self.rel_freq_y = self.rel_freq_y.reset_index()
        self.column_order = data.drop("Y", axis = 1).columns      

    def DE_binary(self, a_i, a_j, y):
        """
        Function to calculate bounds for  DE_{a_i, a_j}(y|a_i) for binary A

        input_data = dataframe of prediction input (does not include column "label")
        density_estimate = dataframe with columns M,A,Z and M_hat, where M_hat contains the class probabilites for M given A,Z (from density estimator function)
        """
       
        if self.task == "binary":
            ub_double, lb_double = self.bounds.bounds_double(a_j, a_i, y) 

            data_ai = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', 'Z']).sort_values(by=['M', 'Z']).drop_duplicates()[self.column_order]
            data_aj = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_j].merge(self.covariates, how="outer", on= ['M', 'Z']).sort_values(by=['M', 'Z']).drop_duplicates()[self.column_order]

            p_z = torch.tensor(data_ai.merge(self.rel_freq_z, how="outer", on= ['Z']).sort_values(by=['M', 'Z'])["proportion"].values)
            p_ai = torch.tensor(np.sum(self.data["A"] == a_i)/self.data.shape[0])
            p_aj = torch.tensor(np.sum(self.data["A"] == a_j)/self.data.shape[0])

            with torch.no_grad():
                logits, m_hat = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
                m_hat = m_hat.detach().numpy().transpose()[0]
                data_ai["M_hat"] = m_hat
                data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
                p_m_ai = torch.tensor(data_ai["M_hat"].values)
                data_ai = data_ai.drop("M_hat", axis = 1)
        
            logits, y_ai = self.network(torch.tensor(data_ai.values, requires_grad = True).float())
            logits, y_mza_j = self.network(torch.tensor(data_aj.values, requires_grad = True).float())
            if y == 0.0:
                y_ai = torch.ones(y_ai.size()) - y_ai
                y_mza_j = torch.ones(y_mza_j.size()) - y_mza_j
                y = 1.0
            y = torch.tensor(y)

            DE_ub = 1/p_ai * ub_double - p_aj/p_ai * torch.sum(y_mza_j * (torch.round(y_mza_j)==y) * p_m_ai * p_z) - torch.sum(y_ai * (torch.round(y_ai)==y) * p_m_ai * p_z)
            DE_lb = 1/p_ai * lb_double - p_aj/p_ai * torch.sum(y_mza_j * (torch.round(y_mza_j)==y) * p_m_ai * p_z) - torch.sum(y_ai * (torch.round(y_ai)==y) * p_m_ai * p_z)
        
        elif self.task == "multiclass":
            ub_double, lb_double = self.bounds.bounds_double(a_j, a_i, y) 

            data_ai = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"]).drop_duplicates()[self.column_order]
            data_aj = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_j].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"]).drop_duplicates()[self.column_order]
            p_z = torch.tensor(data_ai.merge(self.rel_freq_z, how="outer", on= ["Z1", "Z2", "Z3", "Z4"]).drop_duplicates().sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"])["proportion"])    
            p_ai = torch.tensor(np.sum(self.data["A"] == a_i)/self.data.shape[0])
            p_aj = torch.tensor(np.sum(self.data["A"] == a_j)/self.data.shape[0])

            with torch.no_grad():
                m_hat_i_index = torch.argmax(self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())[1], dim = 1).detach().numpy().reshape(-1,1)
                data_ai["M_hat"] = np.repeat(np.expand_dims(sorted(np.unique(self.data["M"])), axis = 0), np.shape(data_ai)[0], axis=0)[:,m_hat_i_index]
                p_m_ai = torch.tensor(data_ai["M_hat"].values)
                data_ai = data_ai.drop("M_hat", axis = 1)

                y_ai = self.network(torch.tensor(data_ai.drop_duplicates().values).float())[:, sorted(np.unique(self.data["Y"])).index(y)]
                y_ai = torch.nan_to_num(y_ai)
                y_mza_j = self.network(torch.tensor(data_aj.drop_duplicates().values).float())[:, sorted(np.unique(self.data["Y"])).index(y)]
                y_mza_j = torch.nan_to_num(y_mza_j)
            
            DE_ub = 1/p_ai * ub_double - p_aj/p_ai * torch.sum(y_mza_j * p_m_ai * p_z) - torch.sum(y_ai * p_m_ai * p_z)
            DE_lb = 1/p_ai * lb_double - p_aj/p_ai * torch.sum(y_mza_j * p_m_ai * p_z) - torch.sum(y_ai * p_m_ai * p_z)
        
        else:
            ub_double, lb_double = self.bounds.bounds_double_continuous(a_j, a_i, y)    

            data_ai = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_i]
            data_aj = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_j]
            data_aiaj = data_ai.copy()
            data_aiaj["A"] = np.repeat(a_j, len(data_aiaj["A"]))
            
            p_z = torch.tensor(data_ai.merge(self.rel_freq_z, how="outer", on= ['Z']).sort_values(by=['M', 'Z'])["proportion"].values)
            p_ai = torch.tensor(np.sum(self.data["A"] == a_i)/self.data.shape[0])
            p_aj = torch.tensor(np.sum(self.data["A"] == a_j)/self.data.shape[0])

            with torch.no_grad():
                logits, m_hat = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
                m_hat = m_hat.detach().numpy().transpose()[0]
                data_ai["M_hat"] = m_hat
                data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
                p_m_ai = torch.tensor(data_ai["M_hat"].values)
                data_ai = data_ai.drop("M_hat", axis = 1)
        
            logits, y_ai = self.network(torch.tensor(data_ai.values, requires_grad = True).float())
            logits, y_mza_j = self.network(torch.tensor(data_aiaj.values, requires_grad = True).float())
            if y == 0.0:
                y_ai = torch.ones(y_ai.size()) - y_ai
                y_mza_j = torch.ones(y_mza_j.size()) - y_mza_j
                y = 1.0
            y = torch.tensor(y)
            
            DE_ub = 1/p_ai * ub_double - p_aj/p_ai * torch.mean(y_mza_j * (torch.round(y_mza_j)==y) * p_m_ai) - torch.mean(y_ai * (torch.round(y_ai)==y) * p_m_ai)
            DE_lb = 1/p_ai * lb_double - p_aj/p_ai * torch.mean(y_mza_j * (torch.round(y_mza_j)==y) * p_m_ai) - torch.mean(y_ai * (torch.round(y_ai)==y) * p_m_ai)

        return DE_ub.requires_grad_(), DE_lb.requires_grad_()
    

    def IE_binary(self, a_i, a_j, y):
        """
        Function to calculate bounds for  IE_{a_i, a_j}(y|a_j) for binary A
        """

        if self.task == "binary":        
            
            ub_double, lb_double = self.bounds.bounds_double(a_i, a_j, y)
            ub_single, lb_single = self.bounds.bounds_single(a_i, y)

            data_ai = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', 'Z']).sort_values(by=['M', 'Z']).drop_duplicates()[self.column_order]
            data_aj = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_j].merge(self.covariates, how="outer", on= ['M', 'Z']).sort_values(by=['M', 'Z']).drop_duplicates()[self.column_order]

            p_z = torch.tensor(data_ai.merge(self.rel_freq_z, how="outer", on= ['Z']).sort_values(by=['M', 'Z'])["proportion"].values)
            p_ai = torch.tensor(np.sum(self.data["A"] == a_i)/self.data.shape[0])
            p_aj = torch.tensor(np.sum(self.data["A"] == a_j)/self.data.shape[0])
        
            with torch.no_grad():
                logits, m_hat_i = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
                logits, m_hat_j = self.density_estimator(torch.tensor(data_aj.drop("M", axis = 1).values).float())
                m_hat_i = m_hat_i.detach().numpy().transpose()[0]            
                m_hat_j = m_hat_j.detach().numpy().transpose()[0]
                
            data_ai["M_hat"] = m_hat_i
            data_aj["M_hat"] = m_hat_j

            data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
            p_m_ai = torch.tensor(data_ai["M_hat"].values)
            data_aj.loc[data_aj["M"] == 0.0,"M_hat"] = np.ones(shape=data_aj.loc[data_aj["M"] == 0.0,"M_hat"].shape) - data_aj.loc[data_aj["M"] == 0.0,"M_hat"]
            p_m_aj = torch.tensor(data_aj["M_hat"].values)
            data_ai = data_ai.drop("M_hat", axis = 1)
            data_aj = data_aj.drop("M_hat", axis = 1)
        
            logits, y_mza = self.network(torch.tensor(data_ai.values, requires_grad = True).float())
            if y == 0.0:
                y_mza = torch.ones(y_mza.size()) - y_mza
                y = 1.0
            y_mza = torch.nan_to_num(y_mza)
            y = torch.tensor(y)

            IE_ub = 1/p_aj * (ub_double-lb_single) - p_ai/p_aj * torch.sum(y_mza * (torch.round(y_mza)==y) * (p_m_aj - p_m_ai) * p_z)
            IE_lb = 1/p_aj * (lb_double-ub_single) - p_ai/p_aj * torch.sum(y_mza * (torch.round(y_mza)==y) * (p_m_aj - p_m_ai) * p_z)

        elif self.task == "multiclass":
            ub_double, lb_double = self.bounds.bounds_double(a_i, a_j, y)
            ub_single, lb_single = self.bounds.bounds_single(a_i, y)

            data_ai = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"]).drop_duplicates()[self.column_order]
            data_aj = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_j].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"]).drop_duplicates()[self.column_order]
            p_z = torch.tensor(data_ai.merge(self.rel_freq_z, how="outer", on= ["Z1", "Z2", "Z3", "Z4"]).drop_duplicates().sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"])["proportion"])    
            p_ai = torch.tensor(np.sum(self.data["A"] == a_i)/self.data.shape[0])
            p_aj = torch.tensor(np.sum(self.data["A"] == a_j)/self.data.shape[0])

            with torch.no_grad():
                m_hat_i_index = torch.argmax(self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())[1], dim = 1).detach().numpy().reshape(-1,1)
                m_hat_j_index = torch.argmax(self.density_estimator(torch.tensor(data_aj.drop("M", axis = 1).values).float())[1], dim = 1).detach().numpy().reshape(-1,1)
                data_ai["M_hat"] = np.repeat(np.expand_dims(sorted(np.unique(self.data["M"])), axis = 0), np.shape(data_ai)[0], axis=0)[:,m_hat_i_index]
                data_aj["M_hat"] = np.repeat(np.expand_dims(sorted(np.unique(self.data["M"])), axis = 0), np.shape(data_aj)[0], axis=0)[:,m_hat_j_index]
                p_m_ai = torch.tensor(data_ai["M_hat"].values)
                p_m_aj = torch.tensor(data_aj["M_hat"].values)
                data_ai = data_ai.drop("M_hat", axis = 1)
                data_aj = data_aj.drop("M_hat", axis = 1)

                y_mza = self.network(torch.tensor(data_ai.drop_duplicates().values).float())[:, sorted(np.unique(self.data["Y"])).index(y)]
                y_mza = torch.nan_to_num(y_mza)
            
            IE_ub = 1/p_aj * (ub_double-lb_single) - p_ai/p_aj * torch.sum(y_mza * (p_m_aj - p_m_ai) * p_z)
            IE_lb = 1/p_aj * (lb_double-ub_single) - p_ai/p_aj * torch.sum(y_mza * (p_m_aj - p_m_ai) * p_z)

        else:            
            ub_double, lb_double = self.bounds.bounds_double_continuous(a_i, a_j, y)
            ub_single, lb_single = self.bounds.bounds_single_continuous(a_i, y)

            data_ai = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_i]
            data_aj = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_j]
            data_aiaj = data_ai.copy()
            data_aiaj["A"] = np.repeat(a_j, len(data_aiaj["A"]))

            p_ai = torch.tensor(np.sum(self.data["A"] == a_i)/self.data.shape[0])
            p_aj = torch.tensor(np.sum(self.data["A"] == a_j)/self.data.shape[0])
        
            with torch.no_grad():
                logits, m_hat_i = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
                logits, m_hat_ij = self.density_estimator(torch.tensor(data_aiaj.drop("M", axis = 1).values).float())   
                m_hat_i = m_hat_i.detach().numpy().transpose()[0]            
                m_hat_ij = m_hat_ij.detach().numpy().transpose()[0]
                
            data_ai["M_hat"] = m_hat_i
            data_aiaj["M_hat"] = m_hat_ij

            data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
            p_m_ai = torch.tensor(data_ai["M_hat"].values)
            data_aiaj.loc[data_aiaj["M"] == 0.0,"M_hat"] = np.ones(shape=data_aiaj.loc[data_aiaj["M"] == 0.0,"M_hat"].shape) - data_aiaj.loc[data_aiaj["M"] == 0.0,"M_hat"]
            p_m_aiaj = torch.tensor(data_aiaj["M_hat"].values)
            data_ai = data_ai.drop("M_hat", axis = 1)
        
            logits, y_mza = self.network(torch.tensor(data_ai.values, requires_grad = True).float())
            if y == 0.0:
                y_mza = torch.ones(y_mza.size()) - y_mza
                y = 1.0
            y_mza = torch.nan_to_num(y_mza)
            y = torch.tensor(y)

            IE_ub = 1/p_aj * (ub_double-lb_single) - p_ai/p_aj * torch.mean(y_mza * (torch.round(y_mza)==y) * (p_m_aiaj - p_m_ai))
            IE_lb = 1/p_aj * (lb_double-ub_single) - p_ai/p_aj * torch.mean(y_mza * (torch.round(y_mza)==y) * (p_m_aiaj - p_m_ai))

        return IE_ub.requires_grad_(), IE_lb.requires_grad_()


    def SE_binary(self, a_i, a_j, y):
        """
        Function to calculate bounds for  SE_{a_i, a_j}(y) for binary A
        """

        if self.task == "binary":
            ub_single, lb_single = self.bounds.bounds_single(a_i, y)  

            data_ai = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', 'Z']).sort_values(by=['M', 'Z']).drop_duplicates()[self.column_order]

            p_z = torch.tensor(data_ai.merge(self.rel_freq_z, how="outer", on= ['Z']).sort_values(by=['M', 'Z']).drop_duplicates()["proportion"].values)#, requires_grad=False)
            p_ai = torch.tensor(np.sum(self.data["A"] == a_i)/self.data.shape[0])
            p_aj = torch.tensor(np.sum(self.data["A"] == a_j)/self.data.shape[0])
        
            with torch.no_grad():
                logits, m_hat_i = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
                m_hat_i = m_hat_i.detach().numpy().transpose()[0]
                data_ai["M_hat"] = m_hat_i
                data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
                p_m_ai = torch.tensor(data_ai["M_hat"].values)
                data_ai = data_ai.drop("M_hat", axis = 1)

            logits, y_ai = self.network(torch.tensor(data_ai.values, requires_grad = True).float())
            if y == 0.0:
                y_ai = torch.ones(y_ai.size()) - y_ai
                y=1.0

            y_ai = torch.nan_to_num(y_ai)
            y = torch.tensor(y)

            SE_ub = 1/p_aj * ub_single - (1 + p_ai/p_aj) * torch.sum(y_ai * (torch.round(y_ai)==y) * p_m_ai * p_z)
            SE_lb = 1/p_aj * lb_single - (1 + p_ai/p_aj) * torch.sum(y_ai * (torch.round(y_ai)==y) * p_m_ai * p_z)

        elif self.task == "mutliclass":
            ub_single, lb_single = self.bounds.bounds_single(a_i, y)    

            data_ai = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"]).drop_duplicates()[self.column_order]
            p_z = torch.tensor(data_ai.merge(self.rel_freq_z, how="outer", on= ["Z1", "Z2", "Z3", "Z4"]).drop_duplicates().sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"])["proportion"])    
            p_ai = torch.tensor(np.sum(self.data["A"] == a_i)/self.data.shape[0])
            p_aj = torch.tensor(np.sum(self.data["A"] == a_j)/self.data.shape[0])

            with torch.no_grad():
                m_hat_i_index = torch.argmax(self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())[1], dim = 1).detach().numpy().reshape(-1,1)
                data_ai["M_hat"] = np.repeat(np.expand_dims(sorted(np.unique(self.data["M"])), axis = 0), np.shape(data_ai)[0], axis=0)[:,m_hat_i_index]
                p_m_ai = torch.tensor(data_ai["M_hat"].values)
                data_ai = data_ai.drop("M_hat", axis = 1)
                y_ai = self.network(torch.tensor(data_ai.drop_duplicates().values).float())[:, sorted(np.unique(self.data["Y"])).index(y)]
                y_ai = torch.nan_to_num(y_ai)
            
            SE_ub = 1/p_aj * ub_single - (1 + p_ai/p_aj) * torch.sum(y_ai * p_m_ai * p_z)
            SE_lb = 1/p_aj * lb_single - (1 + p_ai/p_aj) * torch.sum(y_ai * p_m_ai * p_z)

        else:
            ub_single, lb_single = self.bounds.bounds_single_continuous(a_i, y)  

            data_ai = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_i]
            p_ai = torch.tensor(np.sum(self.data["A"] == a_i)/self.data.shape[0])
            p_aj = torch.tensor(np.sum(self.data["A"] == a_j)/self.data.shape[0])
        
            with torch.no_grad():
                logits, m_hat_i = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
                m_hat_i = m_hat_i.detach().numpy().transpose()[0]
                data_ai["M_hat"] = m_hat_i
                data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
                p_m_ai = torch.tensor(data_ai["M_hat"].values)
                data_ai = data_ai.drop("M_hat", axis = 1)

            logits, y_ai = self.network(torch.tensor(data_ai.values, requires_grad = True).float())
            if y == 0.0:
                y_ai = torch.ones(y_ai.size()) - y_ai
                y=1.0

            y_ai = torch.nan_to_num(y_ai)
            y = torch.tensor(y)

            SE_ub = 1/p_aj * ub_single - (1 + p_ai/p_aj) * torch.mean(y_ai * (torch.round(y_ai)==y) * p_m_ai)
            SE_lb = 1/p_aj * lb_single - (1 + p_ai/p_aj) * torch.mean(y_ai * (torch.round(y_ai)==y) * p_m_ai)


        return SE_ub.requires_grad_(), SE_lb.requires_grad_()

    
    
    def unbounded_effects(self, a_i, a_j, y): 
        """
        Function to calculate the original (unbounded) effects
        """
        
        if self.task == "binary":
            data_ai = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', 'Z']).sort_values(by=['M', 'Z']).drop_duplicates()[self.column_order]
            data_aj = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_j].merge(self.covariates, how="outer", on= ['M', 'Z']).sort_values(by=['M', 'Z']).drop_duplicates()[self.column_order]

            p_z = torch.tensor(data_ai.merge(self.rel_freq_z, how="outer", on= ['Z']).drop_duplicates().sort_values(by=['M', 'Z'])["proportion"])    
            p_ai = torch.tensor(np.sum(self.data["A"] == a_i)/self.data.shape[0])
            p_aj = torch.tensor(np.sum(self.data["A"] == a_j)/self.data.shape[0])

            with torch.no_grad():
                logits, m_hat_i = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
                logits, m_hat_j = self.density_estimator(torch.tensor(data_aj.drop("M", axis = 1).values).float())            
                m_hat_i = m_hat_i.detach().numpy().transpose()[0]            
                m_hat_j = m_hat_j.detach().numpy().transpose()[0]

            data_ai["M_hat"] = m_hat_i
            data_aj["M_hat"] = m_hat_j  
            data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
            p_m_ai = torch.tensor(data_ai["M_hat"].values)
            data_aj.loc[data_aj["M"] == 0.0,"M_hat"] = np.ones(shape=data_aj.loc[data_aj["M"] == 0.0,"M_hat"].shape) - data_aj.loc[data_aj["M"] == 0.0,"M_hat"]
            p_m_aj = torch.tensor(data_aj["M_hat"].values)       
            data_ai = data_ai.drop("M_hat", axis = 1)
            data_aj = data_aj.drop("M_hat", axis = 1)
            logits, y_ai = self.network(torch.tensor(data_ai.drop_duplicates().values).float())
            logits, y_aj = self.network(torch.tensor(data_aj.drop_duplicates().values).float())

            if y == 0.0:

                y_ai = torch.ones(y_ai.size()) - y_ai
                y_aj = torch.ones(y_aj.size()) - y_aj
                y = 1.0
                
            y_ai = torch.nan_to_num(y_ai)
            y_aj = torch.nan_to_num(y_aj)
            y = torch.tensor(y)

            de = torch.sum(y_aj * (torch.round(y_aj)==y) * p_m_ai * p_z) - torch.sum(y_ai * (torch.round(y_ai)==y) * p_m_ai * p_z)
            ie = torch.sum(y_aj * (torch.round(y_aj)==y) * (p_m_ai - p_m_aj) * p_z) 
            se = (torch.tensor(1) - p_ai/p_aj) * torch.sum(y_ai * (torch.round(y_ai)==y) * p_m_ai * p_z)


        elif self.task == "multiclass":
            data_ai = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"]).drop_duplicates()[self.column_order]
            data_aj = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_j].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"]).drop_duplicates()[self.column_order]

            p_z = torch.tensor(data_ai.merge(self.rel_freq_z, how="outer", on= ["Z1", "Z2", "Z3", "Z4"]).drop_duplicates().sort_values(by=['M', "Z1", "Z2", "Z3", "Z4"])["proportion"])    

            with torch.no_grad():
                m_hat_i_index = torch.argmax(self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())[1], dim = 1).detach().numpy().reshape(-1,1)
                m_hat_j_index = torch.argmax(self.density_estimator(torch.tensor(data_aj.drop("M", axis = 1).values).float())[1], dim = 1).detach().numpy().reshape(-1,1)

                data_ai["M_hat"] = np.repeat(np.expand_dims(sorted(np.unique(self.data["M"])), axis = 0), np.shape(data_ai)[0], axis=0)[:,m_hat_i_index]
                data_aj["M_hat"] = np.repeat(np.expand_dims(sorted(np.unique(self.data["M"])), axis = 0), np.shape(data_aj)[0], axis=0)[:,m_hat_j_index]
                p_m_ai = torch.tensor(data_ai["M_hat"].values)
                p_m_aj = torch.tensor(data_aj["M_hat"].values)
                data_ai = data_ai.drop("M_hat", axis = 1)
                data_aj = data_aj.drop("M_hat", axis = 1)
            y_ai = self.network(torch.tensor(data_ai.drop_duplicates().values).float())[:, sorted(np.unique(self.data["Y"])).index(y)]
            y_aj = self.network(torch.tensor(data_aj.drop_duplicates().values).float())[:, sorted(np.unique(self.data["Y"])).index(y)]

            y_ai = torch.nan_to_num(y_ai)
            y_aj = torch.nan_to_num(y_aj)

            de = torch.sum(y_aj * p_m_ai * p_z) - torch.sum(y_ai * p_m_ai * p_z)
            ie = torch.sum(y_aj * (p_m_ai - p_m_aj) * p_z) 
            se = torch.tensor(0)
        
        else: 
            data_ai = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_i]
            data_aj = self.data.drop("Y", axis = 1).loc[self.data["A"] == a_j]
            data_aiaj = data_ai.copy()
            data_ajai = data_aj.copy()
            data_aiaj["A"] = np.repeat(a_j, len(data_aiaj["A"]))
            data_ajai["A"] = np.repeat(a_i, len(data_ajai["A"]))

            p_ai = torch.tensor(np.sum(self.data["A"] == a_i)/self.data.shape[0])
            p_aj = torch.tensor(np.sum(self.data["A"] == a_j)/self.data.shape[0])

            with torch.no_grad():
                logits, m_hat_i = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
                logits, m_hat_j = self.density_estimator(torch.tensor(data_aj.drop("M", axis = 1).values).float())
                logits, m_hat_ij = self.density_estimator(torch.tensor(data_aiaj.drop("M", axis = 1).values).float()) 
                logits, m_hat_ji = self.density_estimator(torch.tensor(data_ajai.drop("M", axis = 1).values).float())               
                m_hat_i = m_hat_i.detach().numpy().transpose()[0]            
                m_hat_j = m_hat_j.detach().numpy().transpose()[0]
                m_hat_ij = m_hat_ij.detach().numpy().transpose()[0]               
                m_hat_ji = m_hat_ji.detach().numpy().transpose()[0]    

            data_ai["M_hat"] = m_hat_i
            data_aj["M_hat"] = m_hat_j
            data_aiaj["M_hat"] = m_hat_ij          
            data_ajai["M_hat"] = m_hat_ji         
            data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
            p_m_ai = torch.tensor(data_ai["M_hat"].values)
            data_aj.loc[data_aj["M"] == 0.0,"M_hat"] = np.ones(shape=data_aj.loc[data_aj["M"] == 0.0,"M_hat"].shape) - data_aj.loc[data_aj["M"] == 0.0,"M_hat"]
            p_m_aj = torch.tensor(data_aj["M_hat"].values)
            data_aiaj.loc[data_aiaj["M"] == 0.0,"M_hat"] = np.ones(shape=data_aiaj.loc[data_aiaj["M"] == 0.0,"M_hat"].shape) - data_aiaj.loc[data_aiaj["M"] == 0.0,"M_hat"]
            p_m_aiaj = torch.tensor(data_aiaj["M_hat"].values)     
            data_ajai.loc[data_ajai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ajai.loc[data_ajai["M"] == 0.0,"M_hat"].shape) - data_ajai.loc[data_ajai["M"] == 0.0,"M_hat"]
            p_m_ajai = torch.tensor(data_ajai["M_hat"].values)              
            data_ai = data_ai.drop("M_hat", axis = 1)
            data_aj = data_aj.drop("M_hat", axis = 1)
            logits, y_ai = self.network(torch.tensor(data_ai.values).float())
            logits, y_aj = self.network(torch.tensor(data_aj.values).float())

            if y == 0.0:
                y_ai = torch.ones(y_ai.size()) - y_ai
                y_aj = torch.ones(y_aj.size()) - y_aj
                y = 1.0
                
            y_ai = torch.nan_to_num(y_ai)
            y_aj = torch.nan_to_num(y_aj)
            y = torch.tensor(y)

            de = torch.mean(y_aj * (torch.round(y_aj)==y) * p_m_ajai) - torch.mean(y_ai * (torch.round(y_ai)==y) * p_m_ai)
            ie = torch.mean(y_aj * (torch.round(y_aj)==y) * (p_m_ai - p_m_aiaj)) 
            se = (torch.tensor(1) - p_ai/p_aj) * torch.mean(y_ai * (torch.round(y_ai)==y) * p_m_ai)

        return torch.tensor([de, ie, se], requires_grad=True)





class Bounds_regression_torch:
    def __init__(self, network, full_data, density_estimator, sensitivity_param):
        if network == None:
            self.network = None
        else:
            self.network = network
        self.full_data = full_data
        self.data = full_data
        if density_estimator == None:
            self.density_estimator = None
        else:
            self.density_estimator = density_estimator.eval()
        self.sensitivity_param = sensitivity_param
        self.rel_freq_z = self.full_data.value_counts(normalize=True, subset = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"])
        self.rel_freq_z = self.rel_freq_z.reset_index().sort_values(by=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"])
        self.rel_freq_a_z = self.full_data.value_counts(normalize=True, subset = ["A", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"])
        self.rel_freq_a_z = self.rel_freq_a_z.reset_index().sort_values(by=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]) 
        self.order = self.full_data.sort_values(by=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"])[["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]].drop_duplicates()
        self.covariates = full_data.columns

    def build_score(self,a):
        prop_table_a  = self.rel_freq_a_z.loc[self.rel_freq_a_z["A"] == a].sort_values(by=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"])
        prop_table_a = prop_table_a.merge(self.order, on=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"], how = "outer").fillna(0).sort_values(by=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"])
        prop_a = prop_table_a["proportion"].values/self.rel_freq_z["proportion"].values
        s_plus_inverse = (1-1/self.sensitivity_param)*prop_a + 1/self.sensitivity_param 
        s_minus_inverse = (1-self.sensitivity_param)*prop_a + self.sensitivity_param
        c_plus = np.float(self.sensitivity_param/(1+self.sensitivity_param))
        c_minus = np.float(1/(1+self.sensitivity_param))
        return s_plus_inverse, s_minus_inverse, c_plus, c_minus

    def bounds_single(self, a):
        s_m_plus, s_m_minus, c_m_plus, c_m_minus = self.build_score(a) 

        f_m_in = self.order
        f_m_in["A"] = np.repeat(a, self.order.shape[0])
        f_m_in["M"] = np.repeat(np.nan, self.order.shape[0])
        f_m_in = f_m_in[self.covariates]   
        with torch.no_grad():
            logits, f_m_z_array = self.density_estimator(torch.tensor(f_m_in.drop("M", axis = 1).values, requires_grad=False).float())
        F_m = np.cumsum(f_m_z_array.detach().numpy(), axis = 1).transpose()

        ind = 0
        Q_plus = torch.zeros(self.order.shape[0], requires_grad=True)
        Q_minus = torch.zeros(self.order.shape[0], requires_grad=True)

        for m in sorted(np.unique(self.full_data["M"])): 
            maz = self.full_data.loc[self.full_data["M"] == m,["M", "A", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]].drop_duplicates().copy()
            maz =  maz.loc[maz["A"] == a]
            maz= maz.merge(self.order.drop(["M", "A"], axis = 1), on =["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"], how = "outer").sort_values(by=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]) 
            maz = maz[self.covariates]
            maz["A"] = maz["A"].fillna(a)
            maz["M"] = maz["M"].fillna(m)
          
            exp_maz = self.network(torch.tensor(maz.values).float()).squeeze().requires_grad_()
            with torch.no_grad():
                density_logits, p_m_az = self.density_estimator(torch.tensor(maz.drop(["M"], axis = 1).values, requires_grad=False).float())
                p_m_az = p_m_az.detach().numpy()[:,np.int(m)]             

            if ind == 0:
                p_plus_m_az = torch.tensor((F_m[ind,:]<c_m_plus) * (s_m_plus * p_m_az) + (F_m[ind,:]>=c_m_plus) * (s_m_plus * c_m_plus + s_m_minus * (F_m[ind,:] - c_m_plus)))
                p_minus_m_az = torch.tensor((F_m[ind,:]<c_m_minus) * (s_m_minus * p_m_az) + (F_m[ind,:]>=c_m_minus) * (s_m_minus * c_m_minus + s_m_plus * (F_m[ind,:] - c_m_minus)))

            else:
                p_plus_m_az = torch.tensor((F_m[ind,:]<c_m_plus) * (s_m_plus * p_m_az) + (F_m[ind-1,:]>c_m_plus)* (s_m_minus * p_m_az) + (F_m[ind,:]>=c_m_plus) * (F_m[ind-1,:]<=c_m_plus) * (s_m_plus * (c_m_plus -F_m[ind-1,:]) + s_m_minus * (F_m[ind,:]-c_m_plus)))               
                p_minus_m_az = torch.tensor((F_m[ind,:]<c_m_minus) * (s_m_minus * p_m_az) + (F_m[ind-1,:]>c_m_minus) * (s_m_plus * p_m_az) + (F_m[ind,:]>=c_m_minus) * (F_m[ind-1,:]<=c_m_minus) * (s_m_minus * (c_m_minus -F_m[ind-1,:]) + s_m_plus * (F_m[ind,:] - c_m_minus)))

            Q_plus = Q_plus + (p_plus_m_az * exp_maz).requires_grad_()
            Q_minus = Q_minus + (p_minus_m_az * exp_maz).requires_grad_()                  
            ind = ind + 1


        ub = torch.mean(Q_plus * torch.tensor(self.rel_freq_z["proportion"].values), dim=(0), keepdim=False).requires_grad_()
        lb = torch.mean(Q_minus * torch.tensor(self.rel_freq_z["proportion"].values), dim=(0), keepdim=False).requires_grad_()

        return ub, lb
    

    def bounds_double(self, a_i, a_j):
        """
        Calculate bounds of the form \sum_z Q(z,(a_i, a_j),S)*P(Z=z),
        where Q(z,(a_i, a_j),S) = \sum_m P(Y=y| M=m, Z=z, do(A=a_i))*P(M=m|Z=z, do(A=a_j)) 
        """

        s_m_plus, s_m_minus, c_m_plus, c_m_minus = self.build_score(a_j) 

        f_m_in = self.order
        f_m_in["A"] = np.repeat(a_j, self.order.shape[0])
        f_m_in["M"] = np.repeat(np.nan, self.order.shape[0])
        f_m_in = f_m_in[self.covariates]   
        with torch.no_grad():
            logits, f_m_z_array = self.density_estimator(torch.tensor(f_m_in.drop("M", axis = 1).values, requires_grad=False).float())
        F_m = np.cumsum(f_m_z_array.detach().numpy(), axis = 1).transpose()


        ind = 0
        Q_plus = torch.zeros(self.order.shape[0], requires_grad=True)
        Q_minus = torch.zeros(self.order.shape[0], requires_grad=True)
        
        for m in sorted(np.unique(self.full_data["M"])): 
            maz = self.full_data.loc[self.full_data["M"] == m,["M", "A", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]].drop_duplicates().copy()
            y_maz = maz.loc[maz["A"] == a_i]
            y_maz = y_maz[self.covariates]
            y_maz = y_maz.merge(self.order.drop(["M", "A"], axis = 1), on =["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"], how = "outer").sort_values(by=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"])
            y_maz["A"] = y_maz["A"].fillna(a_i)
            y_maz["M"] = y_maz["M"].fillna(m)

            m_za_j = maz.loc[maz["A"] == a_j]
            m_za_j = m_za_j.merge(self.order.drop(["M", "A"], axis = 1), on =["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"], how = "outer").sort_values(by=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"])
            m_za_j = m_za_j[self.covariates]
            m_za_j["A"] = m_za_j["A"].fillna(a_j)
            m_za_j["M"] = m_za_j["M"].fillna(m)

            exp_y_maz = self.network(torch.tensor(y_maz.values).float()).squeeze().requires_grad_()
            with torch.no_grad():
                logits, p_m_az = self.density_estimator(torch.tensor(m_za_j.drop(["M"], axis = 1).values, requires_grad=False).float())
                p_m_az = p_m_az.detach().numpy()[:,np.int(m)]
          
            if ind == 0:
                p_plus_m_az = torch.tensor((F_m[ind,:]<c_m_plus) * (s_m_plus * p_m_az) + (F_m[ind,:]>=c_m_plus) * (s_m_plus * c_m_plus  + s_m_minus * (F_m[ind,:] - c_m_plus)))
                p_minus_m_az = torch.tensor((F_m[ind,:]<c_m_minus) * (s_m_minus * p_m_az) + (F_m[ind,:]>=c_m_minus) * (s_m_minus * c_m_minus + s_m_plus * (F_m[ind,:] - c_m_minus)))

            else:
                p_plus_m_az = torch.tensor((F_m[ind,:]<c_m_plus) * (s_m_plus * p_m_az) + (F_m[ind-1,:]>c_m_plus)* (s_m_minus * p_m_az) + (F_m[ind,:]>=c_m_plus) * (F_m[ind-1,:]<=c_m_plus) * (s_m_plus * (c_m_plus - F_m[ind -1,:]) + s_m_minus * (F_m[ind,:] - c_m_plus)))
                p_minus_m_az = torch.tensor((F_m[ind,:]<c_m_minus) * (s_m_minus * p_m_az) + (F_m[ind-1,:]>c_m_minus)* (s_m_plus * p_m_az) + (F_m[ind,:]>=c_m_minus) * (F_m[ind-1,:]<=c_m_minus) * (s_m_minus * (c_m_minus - F_m[ind -1,:]) + s_m_plus * (F_m[ind,:] - c_m_minus)))

            
            Q_plus = Q_plus + (p_plus_m_az * exp_y_maz).requires_grad_()
            Q_minus = Q_minus + (p_minus_m_az * exp_y_maz).requires_grad_()
        
            ind = ind + 1

        ub = torch.mean(Q_plus * torch.tensor(self.rel_freq_z["proportion"].values), dim=(0), keepdim=False).requires_grad_()
        lb = torch.mean(Q_minus * torch.tensor(self.rel_freq_z["proportion"].values), dim=(0), keepdim=False).requires_grad_()

        return ub, lb
    



class Ctf_effects_regression_torch:
    def __init__(self, network, full_data, sensitivity_param=None, density_estimator=None):
        if network == None:
            self.network = None
        else:
            self.network = network
        self.full_data = full_data
        if density_estimator==None:
            self.density_estimator = None
        else:
            self.density_estimator = density_estimator.eval()
        self.sensitivity_param = sensitivity_param
        self.bounds = Bounds_regression_torch(network=self.network, full_data = self.full_data.drop("Y", axis = 1), density_estimator=self.density_estimator, sensitivity_param=self.sensitivity_param)
        self.covariates = pd.merge(full_data["M"].drop_duplicates(), self.full_data[["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]].drop_duplicates(), how = "cross")
        self.rel_freq_z = self.full_data.value_counts(normalize=True, subset = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"])
        self.rel_freq_z = self.rel_freq_z.reset_index()
        self.column_order = full_data.drop("Y", axis = 1).columns   

    def DE_binary(self, a_i, a_j):

        ub_double, lb_double = self.bounds.bounds_double(a_j, a_i) 

        data_ai = self.full_data.loc[self.full_data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]).drop("Y", axis = 1).drop_duplicates()[self.column_order]
        data_aj = self.full_data.loc[self.full_data["A"] == a_j].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]).drop("Y", axis = 1).drop_duplicates()[self.column_order]
        data_ai["A"] = data_ai["A"].fillna(a_i)
        data_aj["A"] = data_aj["A"].fillna(a_j)

        p_ai = torch.tensor(np.sum(self.full_data["A"] == a_i)/self.full_data.shape[0])
        p_aj = torch.tensor(np.sum(self.full_data["A"] == a_j)/self.full_data.shape[0])
        p_z = torch.tensor(data_ai.merge(self.rel_freq_z, how="outer", on= ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]).drop_duplicates().sort_values(by=['M', "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"])["proportion"])

        with torch.no_grad():
            logits, m_hat = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
            m_hat = m_hat.detach().numpy()

            indices = np.array(data_ai["M"].values.astype(int))
            data_ai["M_hat"] = np.array(m_hat[np.arange(len(indices)).reshape(-1,1), indices.reshape(-1,1)]).reshape(-1,1)
            data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
            p_m_ai = torch.tensor(data_ai["M_hat"].values)
            data_ai = data_ai.drop("M_hat", axis = 1)

        y_ai = self.network(torch.tensor(data_ai.values).float())
        y_mza_j = self.network(torch.tensor(data_aj.values).float())

        DE_ub = 1/p_ai * ub_double - p_aj/p_ai * torch.mean(y_mza_j * p_m_ai * p_z) - torch.mean(y_ai *  p_m_ai * p_z)
        DE_lb = 1/p_ai * lb_double - p_aj/p_ai * torch.mean(y_mza_j * p_m_ai * p_z) - torch.mean(y_ai * p_m_ai * p_z)

        return DE_ub.requires_grad_(), DE_lb.requires_grad_()



    def IE_binary(self, a_i, a_j):
        """
        Function to calculate bounds for  IE_{a_i, a_j}(y|a_j) for binary A
        """
        ub_double, lb_double = self.bounds.bounds_double(a_i, a_j)
        ub_single, lb_single = self.bounds.bounds_single(a_i)

        data_ai = self.full_data.loc[self.full_data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]).drop("Y", axis = 1).drop_duplicates()[self.column_order]
        data_aj = self.full_data.loc[self.full_data["A"] == a_j].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]).drop("Y", axis = 1).drop_duplicates()[self.column_order]
        data_ai["A"] = data_ai["A"].fillna(a_i)
        data_aj["A"] = data_aj["A"].fillna(a_j)

        p_ai = torch.tensor(np.sum(self.full_data["A"] == a_i)/self.full_data.shape[0], requires_grad=False)
        p_aj = torch.tensor(np.sum(self.full_data["A"] == a_j)/self.full_data.shape[0], requires_grad=False)
        p_z = torch.tensor(data_ai.merge(self.rel_freq_z, how="outer", on= ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]).drop_duplicates().sort_values(by=['M', "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"])["proportion"], requires_grad=False)

        with torch.no_grad():
            logits, m_hat_i = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
            logits, m_hat_j = self.density_estimator(torch.tensor(data_aj.drop("M", axis = 1).values).float())
            m_hat_i = m_hat_i.detach().numpy()     
            m_hat_j = m_hat_j.detach().numpy()
  
            indices_i = np.array(data_ai["M"].values.astype(int))
            data_ai["M_hat"] = np.array(m_hat_i[np.arange(len(indices_i)).reshape(-1,1), indices_i.reshape(-1,1)]).reshape(-1,1)
            indices_j = np.array(data_aj["M"].values.astype(int))
            data_aj["M_hat"] = np.array(m_hat_j[np.arange(len(indices_j)).reshape(-1,1), indices_j.reshape(-1,1)]).reshape(-1,1)
            data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
            p_m_ai = torch.tensor(data_ai["M_hat"].values)
            data_aj.loc[data_aj["M"] == 0.0,"M_hat"] = np.ones(shape=data_aj.loc[data_aj["M"] == 0.0,"M_hat"].shape) - data_aj.loc[data_aj["M"] == 0.0,"M_hat"]
            p_m_aj = torch.tensor(data_aj["M_hat"].values)
            data_ai = data_ai.drop("M_hat", axis = 1)
            data_aj = data_aj.drop("M_hat", axis = 1)

        y_mza = self.network(torch.tensor(data_ai.values).float())

        IE_ub = 1/p_aj * (ub_double-lb_single) - p_ai/p_aj * torch.mean(y_mza * (p_m_aj - p_m_ai) * p_z)
        IE_lb = 1/p_aj * (lb_double-ub_single) - p_ai/p_aj * torch.mean(y_mza * (p_m_aj - p_m_ai) * p_z)

        return IE_ub.requires_grad_(), IE_lb.requires_grad_()

    def SE_binary(self, a_i, a_j):
        """
        Function to calculate bounds for  SE_{a_i, a_j}(y) for binary A
        """
        ub_single, lb_single = self.bounds.bounds_single(a_i)    

        data_ai = self.full_data.loc[self.full_data["A"] == a_i].merge(self.covariates, how="outer", on= ['M', "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]).sort_values(by=['M', "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]).drop("Y", axis = 1).drop_duplicates()[self.column_order]
        data_ai["A"] = data_ai["A"].fillna(a_i)

        p_ai = torch.tensor(np.sum(self.full_data["A"] == a_i)/self.full_data.shape[0])
        p_aj = torch.tensor(np.sum(self.full_data["A"] == a_j)/self.full_data.shape[0])
        p_z = torch.tensor(data_ai.merge(self.rel_freq_z, how="outer", on= ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]).drop_duplicates().sort_values(by=['M', "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"])["proportion"])     

        with torch.no_grad():
            logits, m_hat_i = self.density_estimator(torch.tensor(data_ai.drop("M", axis = 1).values).float())
            m_hat_i = m_hat_i.detach().numpy()
            indices_i = np.array(data_ai["M"].values.astype(int))
            data_ai["M_hat"] = np.array(m_hat_i[np.arange(len(indices_i)).reshape(-1,1), indices_i.reshape(-1,1)]).reshape(-1,1)
            data_ai.loc[data_ai["M"] == 0.0,"M_hat"] = np.ones(shape=data_ai.loc[data_ai["M"] == 0.0,"M_hat"].shape) - data_ai.loc[data_ai["M"] == 0.0,"M_hat"]
            p_m_ai = torch.tensor(data_ai["M_hat"].values)
            data_ai = data_ai.drop("M_hat", axis = 1)

        y_ai = self.network(torch.tensor(data_ai.values).float())

        SE_ub = 1/p_aj * ub_single - (1 + p_ai/p_aj) * torch.mean(y_ai * p_m_ai * p_z)
        SE_lb = 1/p_aj * lb_single - (1 + p_ai/p_aj) * torch.mean(y_ai * p_m_ai * p_z)

        return SE_ub.requires_grad_(), SE_lb.requires_grad_()
    