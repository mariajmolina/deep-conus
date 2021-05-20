import numpy as np
import xarray as xr
import pandas as pd
import glob
import matplotlib.pyplot as plt
# import local paths
from configpaths import current_dl_preprocess, current_dl_models, deep_conus_home
from configpaths import future_dl_models, model25_dir_future, future_dl_preprocess
exec(open(f'{deep_conus_home}/08_dlmodel_evaluator.py').read())

def grab_all_indices(variable, total_perm, perc_min, perc_max):
    
    """
    From all permutations of specified variable, grab the subset of permutations that were below the perc_min and above the perc_max of all permuted cases.
    
    Args:
        variable (int): Variable index (integer) for permutation cases.
        total_perm (int): Total number of permutations.
        perc_min (float): Lower percentile for confidence intervals.
        perc_max (float): Maximum percentile for confidence intervals.
        
    Returns:
        Xarray dataset containing the indices of the permutations for cases the >perc_max, <perc_min, and > perc_min & < perc_max.
    
    """
    li_1 = []
    for p_num in range(total_perm):
        path = f'{model25_dir_future}/scalar_results_nomask_model25_random1_pfivar{variable}_perm{p_num+1}.csv'
        df = pd.read_csv(path, index_col=None, header=0)
        li_1.append(df)
        li_1_concat = pd.concat(li_1, axis=0, ignore_index=True)
        csi_1_bot_indx = li_1_concat["CSI"].where(li_1_concat["CSI"]<=li_1_concat["CSI"].quantile(perc_min)).dropna().index.values
        csi_1_bot_vals = li_1_concat["CSI"].where(li_1_concat["CSI"]<=li_1_concat["CSI"].quantile(perc_min)).dropna().values
        csi_1_bot_rand = np.ones(csi_1_bot_indx.shape[0])
        csi_1_top_indx = li_1_concat["CSI"].where(li_1_concat["CSI"]>=li_1_concat["CSI"].quantile(perc_max)).dropna().index.values
        csi_1_top_vals = li_1_concat["CSI"].where(li_1_concat["CSI"]>=li_1_concat["CSI"].quantile(perc_max)).dropna().values
        csi_1_top_rand = np.ones(csi_1_top_indx.shape[0])
        auc_1_bot_indx = li_1_concat["AUC"].where(li_1_concat["AUC"]<=li_1_concat["AUC"].quantile(perc_min)).dropna().index.values
        auc_1_bot_vals = li_1_concat["AUC"].where(li_1_concat["AUC"]<=li_1_concat["AUC"].quantile(perc_min)).dropna().values
        auc_1_bot_rand = np.ones(auc_1_bot_indx.shape[0])
        auc_1_top_indx = li_1_concat["AUC"].where(li_1_concat["AUC"]>=li_1_concat["AUC"].quantile(perc_max)).dropna().index.values
        auc_1_top_vals = li_1_concat["AUC"].where(li_1_concat["AUC"]>=li_1_concat["AUC"].quantile(perc_max)).dropna().values
        auc_1_top_rand = np.ones(auc_1_top_indx.shape[0])
        csi_1_mid_indx = li_1_concat["CSI"].where(
            (li_1_concat["CSI"]>li_1_concat["CSI"].quantile(perc_min))&(li_1_concat["CSI"]<li_1_concat["CSI"].quantile(perc_max))).dropna().index.values
        csi_1_mid_vals = li_1_concat["CSI"].where(
            (li_1_concat["CSI"]>li_1_concat["CSI"].quantile(perc_min))&(li_1_concat["CSI"]<li_1_concat["CSI"].quantile(perc_max))).dropna().values
        csi_1_mid_rand = np.ones(csi_1_mid_indx.shape[0])
        auc_1_mid_indx = li_1_concat["AUC"].where(
            (li_1_concat["AUC"]>li_1_concat["AUC"].quantile(perc_min))&(li_1_concat["AUC"]<li_1_concat["AUC"].quantile(perc_max))).dropna().index.values
        auc_1_mid_vals = li_1_concat["AUC"].where(
            (li_1_concat["AUC"]>li_1_concat["AUC"].quantile(perc_min))&(li_1_concat["AUC"]<li_1_concat["AUC"].quantile(perc_max))).dropna().values
        auc_1_mid_rand = np.ones(auc_1_mid_indx.shape[0])

    li_2 = []
    for p_num in range(total_perm):
        path = f'{model25_dir_future}/scalar_results_nomask_model25_random2_pfivar{variable}_perm{p_num+1}.csv'
        df = pd.read_csv(path, index_col=None, header=0)
        li_2.append(df)
        li_2_concat = pd.concat(li_2, axis=0, ignore_index=True)
        csi_2_bot_indx = li_2_concat["CSI"].where(li_2_concat["CSI"]<=li_2_concat["CSI"].quantile(perc_min)).dropna().index.values
        csi_2_bot_vals = li_2_concat["CSI"].where(li_2_concat["CSI"]<=li_2_concat["CSI"].quantile(perc_min)).dropna().values
        csi_2_bot_rand = np.ones(csi_2_bot_indx.shape[0])*2
        csi_2_top_indx = li_2_concat["CSI"].where(li_2_concat["CSI"]>=li_2_concat["CSI"].quantile(perc_max)).dropna().index.values
        csi_2_top_vals = li_2_concat["CSI"].where(li_2_concat["CSI"]>=li_2_concat["CSI"].quantile(perc_max)).dropna().values
        csi_2_top_rand = np.ones(csi_2_top_indx.shape[0])*2
        auc_2_bot_indx = li_2_concat["AUC"].where(li_2_concat["AUC"]<=li_2_concat["AUC"].quantile(perc_min)).dropna().index.values
        auc_2_bot_vals = li_2_concat["AUC"].where(li_2_concat["AUC"]<=li_2_concat["AUC"].quantile(perc_min)).dropna().values
        auc_2_bot_rand = np.ones(auc_2_bot_indx.shape[0])*2
        auc_2_top_indx = li_2_concat["AUC"].where(li_2_concat["AUC"]>=li_2_concat["AUC"].quantile(perc_max)).dropna().index.values
        auc_2_top_vals = li_2_concat["AUC"].where(li_2_concat["AUC"]>=li_2_concat["AUC"].quantile(perc_max)).dropna().values
        auc_2_top_rand = np.ones(auc_2_top_indx.shape[0])*2
        csi_2_mid_indx = li_2_concat["CSI"].where(
            (li_2_concat["CSI"]>li_2_concat["CSI"].quantile(perc_min))&(li_2_concat["CSI"]<li_2_concat["CSI"].quantile(perc_max))).dropna().index.values
        csi_2_mid_vals = li_2_concat["CSI"].where(
            (li_2_concat["CSI"]>li_2_concat["CSI"].quantile(perc_min))&(li_2_concat["CSI"]<li_2_concat["CSI"].quantile(perc_max))).dropna().values
        csi_2_mid_rand = np.ones(csi_2_mid_indx.shape[0])*2
        auc_2_mid_indx = li_2_concat["AUC"].where(
            (li_2_concat["AUC"]>li_2_concat["AUC"].quantile(perc_min))&(li_2_concat["AUC"]<li_2_concat["AUC"].quantile(perc_max))).dropna().index.values
        auc_2_mid_vals = li_2_concat["AUC"].where(
            (li_2_concat["AUC"]>li_2_concat["AUC"].quantile(perc_min))&(li_2_concat["AUC"]<li_2_concat["AUC"].quantile(perc_max))).dropna().values
        auc_2_mid_rand = np.ones(auc_2_mid_indx.shape[0])*2

    li_3 = []
    for p_num in range(total_perm):
        path = f'{model25_dir_future}/scalar_results_nomask_model25_random3_pfivar{variable}_perm{p_num+1}.csv'
        df = pd.read_csv(path, index_col=None, header=0)
        li_3.append(df)
        li_3_concat = pd.concat(li_3, axis=0, ignore_index=True)
        csi_3_bot_indx = li_3_concat["CSI"].where(li_3_concat["CSI"]<=li_3_concat["CSI"].quantile(perc_min)).dropna().index.values
        csi_3_bot_vals = li_3_concat["CSI"].where(li_3_concat["CSI"]<=li_3_concat["CSI"].quantile(perc_min)).dropna().values
        csi_3_bot_rand = np.ones(csi_3_bot_indx.shape[0])*3
        csi_3_top_indx = li_3_concat["CSI"].where(li_3_concat["CSI"]>=li_3_concat["CSI"].quantile(perc_max)).dropna().index.values
        csi_3_top_vals = li_3_concat["CSI"].where(li_3_concat["CSI"]>=li_3_concat["CSI"].quantile(perc_max)).dropna().values
        csi_3_top_rand = np.ones(csi_3_top_indx.shape[0])*3
        auc_3_bot_indx = li_3_concat["AUC"].where(li_3_concat["AUC"]<=li_3_concat["AUC"].quantile(perc_min)).dropna().index.values
        auc_3_bot_vals = li_3_concat["AUC"].where(li_3_concat["AUC"]<=li_3_concat["AUC"].quantile(perc_min)).dropna().values
        auc_3_bot_rand = np.ones(auc_3_bot_indx.shape[0])*3
        auc_3_top_indx = li_3_concat["AUC"].where(li_3_concat["AUC"]>=li_3_concat["AUC"].quantile(perc_max)).dropna().index.values
        auc_3_top_vals = li_3_concat["AUC"].where(li_3_concat["AUC"]>=li_3_concat["AUC"].quantile(perc_max)).dropna().values
        auc_3_top_rand = np.ones(auc_3_top_indx.shape[0])*3
        csi_3_mid_indx = li_3_concat["CSI"].where(
            (li_3_concat["CSI"]>li_3_concat["CSI"].quantile(perc_min))&(li_3_concat["CSI"]<li_3_concat["CSI"].quantile(perc_max))).dropna().index.values
        csi_3_mid_vals = li_3_concat["CSI"].where(
            (li_3_concat["CSI"]>li_3_concat["CSI"].quantile(perc_min))&(li_3_concat["CSI"]<li_3_concat["CSI"].quantile(perc_max))).dropna().values
        csi_3_mid_rand = np.ones(csi_3_mid_indx.shape[0])*3
        auc_3_mid_indx = li_3_concat["AUC"].where(
            (li_3_concat["AUC"]>li_3_concat["AUC"].quantile(perc_min))&(li_3_concat["AUC"]<li_3_concat["AUC"].quantile(perc_max))).dropna().index.values
        auc_3_mid_vals = li_3_concat["AUC"].where(
            (li_3_concat["AUC"]>li_3_concat["AUC"].quantile(perc_min))&(li_3_concat["AUC"]<li_3_concat["AUC"].quantile(perc_max))).dropna().values
        auc_3_mid_rand = np.ones(auc_3_mid_indx.shape[0])*3

    li_4 = []
    for p_num in range(total_perm):
        path = f'{model25_dir_future}/scalar_results_nomask_model25_random4_pfivar{variable}_perm{p_num+1}.csv'
        df = pd.read_csv(path, index_col=None, header=0)
        li_4.append(df)
        li_4_concat = pd.concat(li_4, axis=0, ignore_index=True)
        csi_4_bot_indx = li_4_concat["CSI"].where(li_4_concat["CSI"]<=li_4_concat["CSI"].quantile(perc_min)).dropna().index.values
        csi_4_bot_vals = li_4_concat["CSI"].where(li_4_concat["CSI"]<=li_4_concat["CSI"].quantile(perc_min)).dropna().values
        csi_4_bot_rand = np.ones(csi_4_bot_indx.shape[0])*4
        csi_4_top_indx = li_4_concat["CSI"].where(li_4_concat["CSI"]>=li_4_concat["CSI"].quantile(perc_max)).dropna().index.values
        csi_4_top_vals = li_4_concat["CSI"].where(li_4_concat["CSI"]>=li_4_concat["CSI"].quantile(perc_max)).dropna().values
        csi_4_top_rand = np.ones(csi_4_top_indx.shape[0])*4
        auc_4_bot_indx = li_4_concat["AUC"].where(li_4_concat["AUC"]<=li_4_concat["AUC"].quantile(perc_min)).dropna().index.values
        auc_4_bot_vals = li_4_concat["AUC"].where(li_4_concat["AUC"]<=li_4_concat["AUC"].quantile(perc_min)).dropna().values
        auc_4_bot_rand = np.ones(auc_4_bot_indx.shape[0])*4
        auc_4_top_indx = li_4_concat["AUC"].where(li_4_concat["AUC"]>=li_4_concat["AUC"].quantile(perc_max)).dropna().index.values
        auc_4_top_vals = li_4_concat["AUC"].where(li_4_concat["AUC"]>=li_4_concat["AUC"].quantile(perc_max)).dropna().values
        auc_4_top_rand = np.ones(auc_4_top_indx.shape[0])*4
        csi_4_mid_indx = li_4_concat["CSI"].where(
            (li_4_concat["CSI"]>li_4_concat["CSI"].quantile(perc_min))&(li_4_concat["CSI"]<li_4_concat["CSI"].quantile(perc_max))).dropna().index.values
        csi_4_mid_vals = li_4_concat["CSI"].where(
            (li_4_concat["CSI"]>li_4_concat["CSI"].quantile(perc_min))&(li_4_concat["CSI"]<li_4_concat["CSI"].quantile(perc_max))).dropna().values
        csi_4_mid_rand = np.ones(csi_4_mid_indx.shape[0])*4
        auc_4_mid_indx = li_4_concat["AUC"].where(
            (li_4_concat["AUC"]>li_4_concat["AUC"].quantile(perc_min))&(li_4_concat["AUC"]<li_4_concat["AUC"].quantile(perc_max))).dropna().index.values
        auc_4_mid_vals = li_4_concat["AUC"].where(
            (li_4_concat["AUC"]>li_4_concat["AUC"].quantile(perc_min))&(li_4_concat["AUC"]<li_4_concat["AUC"].quantile(perc_max))).dropna().values
        auc_4_mid_rand = np.ones(auc_4_mid_indx.shape[0])*4

    li_5 = []
    for p_num in range(total_perm):
        path = f'{model25_dir_future}/scalar_results_nomask_model25_random5_pfivar{variable}_perm{p_num+1}.csv'
        df = pd.read_csv(path, index_col=None, header=0)
        li_5.append(df)
        li_5_concat = pd.concat(li_5, axis=0, ignore_index=True)
        csi_5_bot_indx = li_5_concat["CSI"].where(li_5_concat["CSI"]<=li_5_concat["CSI"].quantile(perc_min)).dropna().index.values
        csi_5_bot_vals = li_5_concat["CSI"].where(li_5_concat["CSI"]<=li_5_concat["CSI"].quantile(perc_min)).dropna().values
        csi_5_bot_rand = np.ones(csi_5_bot_indx.shape[0])*5
        csi_5_top_indx = li_5_concat["CSI"].where(li_5_concat["CSI"]>=li_5_concat["CSI"].quantile(perc_max)).dropna().index.values
        csi_5_top_vals = li_5_concat["CSI"].where(li_5_concat["CSI"]>=li_5_concat["CSI"].quantile(perc_max)).dropna().values
        csi_5_top_rand = np.ones(csi_5_top_indx.shape[0])*5
        auc_5_bot_indx = li_5_concat["AUC"].where(li_5_concat["AUC"]<=li_5_concat["AUC"].quantile(perc_min)).dropna().index.values
        auc_5_bot_vals = li_5_concat["AUC"].where(li_5_concat["AUC"]<=li_5_concat["AUC"].quantile(perc_min)).dropna().values
        auc_5_bot_rand = np.ones(auc_5_bot_indx.shape[0])*5
        auc_5_top_indx = li_5_concat["AUC"].where(li_5_concat["AUC"]>=li_5_concat["AUC"].quantile(perc_max)).dropna().index.values
        auc_5_top_vals = li_5_concat["AUC"].where(li_5_concat["AUC"]>=li_5_concat["AUC"].quantile(perc_max)).dropna().values
        auc_5_top_rand = np.ones(auc_5_top_indx.shape[0])*5
        csi_5_mid_indx = li_5_concat["CSI"].where(
            (li_5_concat["CSI"]>li_5_concat["CSI"].quantile(perc_min))&(li_5_concat["CSI"]<li_5_concat["CSI"].quantile(perc_max))).dropna().index.values
        csi_5_mid_vals = li_5_concat["CSI"].where(
            (li_5_concat["CSI"]>li_5_concat["CSI"].quantile(perc_min))&(li_5_concat["CSI"]<li_5_concat["CSI"].quantile(perc_max))).dropna().values
        csi_5_mid_rand = np.ones(csi_5_mid_indx.shape[0])*5
        auc_5_mid_indx = li_5_concat["AUC"].where(
            (li_5_concat["AUC"]>li_5_concat["AUC"].quantile(perc_min))&(li_5_concat["AUC"]<li_5_concat["AUC"].quantile(perc_max))).dropna().index.values
        auc_5_mid_vals = li_5_concat["AUC"].where(
            (li_5_concat["AUC"]>li_5_concat["AUC"].quantile(perc_min))&(li_5_concat["AUC"]<li_5_concat["AUC"].quantile(perc_max))).dropna().values
        auc_5_mid_rand = np.ones(auc_5_mid_indx.shape[0])*5

    li_1 = []
    for p_num in range(total_perm):
        path = f'{model25_dir_future}/bss_scalar_results_nomask_model25_random1_0.05_pfivar{variable}_perm{p_num+1}.csv'
        df = pd.read_csv(path, index_col=None, header=0)
        li_1.append(df)
        li_1_concat = pd.concat(li_1, axis=0, ignore_index=True)
        bss_1_bot_indx = li_1_concat["BSS"].where(li_1_concat["BSS"]<=li_1_concat["BSS"].quantile(perc_min)).dropna().index.values
        bss_1_bot_vals = li_1_concat["BSS"].where(li_1_concat["BSS"]<=li_1_concat["BSS"].quantile(perc_min)).dropna().values
        bss_1_bot_rand = np.ones(bss_1_bot_indx.shape[0])
        bss_1_top_indx = li_1_concat["BSS"].where(li_1_concat["BSS"]>=li_1_concat["BSS"].quantile(perc_max)).dropna().index.values
        bss_1_top_vals = li_1_concat["BSS"].where(li_1_concat["BSS"]>=li_1_concat["BSS"].quantile(perc_max)).dropna().values
        bss_1_top_rand = np.ones(bss_1_top_indx.shape[0])
        bss_1_mid_indx = li_1_concat["BSS"].where(
            (li_1_concat["BSS"]>li_1_concat["BSS"].quantile(perc_min))&(li_1_concat["BSS"]<li_1_concat["BSS"].quantile(perc_max))).dropna().index.values
        bss_1_mid_vals = li_1_concat["BSS"].where(
            (li_1_concat["BSS"]>li_1_concat["BSS"].quantile(perc_min))&(li_1_concat["BSS"]<li_1_concat["BSS"].quantile(perc_max))).dropna().values
        bss_1_mid_rand = np.ones(bss_1_mid_indx.shape[0])

    li_2 = []
    for p_num in range(total_perm):
        path = f'{model25_dir_future}/bss_scalar_results_nomask_model25_random2_0.05_pfivar{variable}_perm{p_num+1}.csv'
        df = pd.read_csv(path, index_col=None, header=0)
        li_2.append(df)
        li_2_concat = pd.concat(li_2, axis=0, ignore_index=True)
        bss_2_bot_indx = li_2_concat["BSS"].where(li_2_concat["BSS"]<=li_2_concat["BSS"].quantile(perc_min)).dropna().index.values
        bss_2_bot_vals = li_2_concat["BSS"].where(li_2_concat["BSS"]<=li_2_concat["BSS"].quantile(perc_min)).dropna().values
        bss_2_bot_rand = np.ones(bss_2_bot_indx.shape[0])*2
        bss_2_top_indx = li_2_concat["BSS"].where(li_2_concat["BSS"]>=li_2_concat["BSS"].quantile(perc_max)).dropna().index.values
        bss_2_top_vals = li_2_concat["BSS"].where(li_2_concat["BSS"]>=li_2_concat["BSS"].quantile(perc_max)).dropna().values
        bss_2_top_rand = np.ones(bss_2_top_indx.shape[0])*2
        bss_2_mid_indx = li_2_concat["BSS"].where(
            (li_2_concat["BSS"]>li_2_concat["BSS"].quantile(perc_min))&(li_2_concat["BSS"]<li_2_concat["BSS"].quantile(perc_max))).dropna().index.values
        bss_2_mid_vals = li_2_concat["BSS"].where(
            (li_2_concat["BSS"]>li_2_concat["BSS"].quantile(perc_min))&(li_2_concat["BSS"]<li_2_concat["BSS"].quantile(perc_max))).dropna().values
        bss_2_mid_rand = np.ones(bss_2_mid_indx.shape[0])*2

    li_3 = []
    for p_num in range(total_perm):
        path = f'{model25_dir_future}/bss_scalar_results_nomask_model25_random3_0.05_pfivar{variable}_perm{p_num+1}.csv'
        df = pd.read_csv(path, index_col=None, header=0)
        li_3.append(df)
        li_3_concat = pd.concat(li_3, axis=0, ignore_index=True)
        bss_3_bot_indx = li_3_concat["BSS"].where(li_3_concat["BSS"]<=li_3_concat["BSS"].quantile(perc_min)).dropna().index.values
        bss_3_bot_vals = li_3_concat["BSS"].where(li_3_concat["BSS"]<=li_3_concat["BSS"].quantile(perc_min)).dropna().values
        bss_3_bot_rand = np.ones(bss_3_bot_indx.shape[0])*3
        bss_3_top_indx = li_3_concat["BSS"].where(li_3_concat["BSS"]>=li_3_concat["BSS"].quantile(perc_max)).dropna().index.values
        bss_3_top_vals = li_3_concat["BSS"].where(li_3_concat["BSS"]>=li_3_concat["BSS"].quantile(perc_max)).dropna().values
        bss_3_top_rand = np.ones(bss_3_top_indx.shape[0])*3
        bss_3_mid_indx = li_3_concat["BSS"].where(
            (li_3_concat["BSS"]>li_3_concat["BSS"].quantile(perc_min))&(li_3_concat["BSS"]<li_3_concat["BSS"].quantile(perc_max))).dropna().index.values
        bss_3_mid_vals = li_3_concat["BSS"].where(
            (li_3_concat["BSS"]>li_3_concat["BSS"].quantile(perc_min))&(li_3_concat["BSS"]<li_3_concat["BSS"].quantile(perc_max))).dropna().values
        bss_3_mid_rand = np.ones(bss_3_mid_indx.shape[0])*3

    li_4 = []
    for p_num in range(total_perm):
        path = f'{model25_dir_future}/bss_scalar_results_nomask_model25_random4_0.05_pfivar{variable}_perm{p_num+1}.csv'
        df = pd.read_csv(path, index_col=None, header=0)
        li_4.append(df)
        li_4_concat = pd.concat(li_4, axis=0, ignore_index=True)
        bss_4_bot_indx = li_4_concat["BSS"].where(li_4_concat["BSS"]<=li_4_concat["BSS"].quantile(perc_min)).dropna().index.values
        bss_4_bot_vals = li_4_concat["BSS"].where(li_4_concat["BSS"]<=li_4_concat["BSS"].quantile(perc_min)).dropna().values
        bss_4_bot_rand = np.ones(bss_4_bot_indx.shape[0])*4
        bss_4_top_indx = li_4_concat["BSS"].where(li_4_concat["BSS"]>=li_4_concat["BSS"].quantile(perc_max)).dropna().index.values
        bss_4_top_vals = li_4_concat["BSS"].where(li_4_concat["BSS"]>=li_4_concat["BSS"].quantile(perc_max)).dropna().values
        bss_4_top_rand = np.ones(bss_4_top_indx.shape[0])*4
        bss_4_mid_indx = li_4_concat["BSS"].where(
            (li_4_concat["BSS"]>li_4_concat["BSS"].quantile(perc_min))&(li_4_concat["BSS"]<li_4_concat["BSS"].quantile(perc_max))).dropna().index.values
        bss_4_mid_vals = li_4_concat["BSS"].where(
            (li_4_concat["BSS"]>li_4_concat["BSS"].quantile(perc_min))&(li_4_concat["BSS"]<li_4_concat["BSS"].quantile(perc_max))).dropna().values
        bss_4_mid_rand = np.ones(bss_4_mid_indx.shape[0])*4

    li_5 = []
    for p_num in range(total_perm):
        path = f'{model25_dir_future}/bss_scalar_results_nomask_model25_random5_0.05_pfivar{variable}_perm{p_num+1}.csv'
        df = pd.read_csv(path, index_col=None, header=0)
        li_5.append(df)
        li_5_concat = pd.concat(li_5, axis=0, ignore_index=True)
        bss_5_bot_indx = li_5_concat["BSS"].where(li_5_concat["BSS"]<=li_5_concat["BSS"].quantile(perc_min)).dropna().index.values
        bss_5_bot_vals = li_5_concat["BSS"].where(li_5_concat["BSS"]<=li_5_concat["BSS"].quantile(perc_min)).dropna().values
        bss_5_bot_rand = np.ones(bss_5_bot_indx.shape[0])*5
        bss_5_top_indx = li_5_concat["BSS"].where(li_5_concat["BSS"]>=li_5_concat["BSS"].quantile(perc_max)).dropna().index.values
        bss_5_top_vals = li_5_concat["BSS"].where(li_5_concat["BSS"]>=li_5_concat["BSS"].quantile(perc_max)).dropna().values
        bss_5_top_rand = np.ones(bss_5_top_indx.shape[0])*5
        bss_5_mid_indx = li_5_concat["BSS"].where(
            (li_5_concat["BSS"]>li_5_concat["BSS"].quantile(perc_min))&(li_5_concat["BSS"]<li_5_concat["BSS"].quantile(perc_max))).dropna().index.values
        bss_5_mid_vals = li_5_concat["BSS"].where(
            (li_5_concat["BSS"]>li_5_concat["BSS"].quantile(perc_min))&(li_5_concat["BSS"]<li_5_concat["BSS"].quantile(perc_max))).dropna().values
        bss_5_mid_rand = np.ones(bss_5_mid_indx.shape[0])*5

    error_metrics = xr.Dataset({
            "bss_mid_random": (['i'], np.hstack([bss_1_mid_rand,bss_2_mid_rand,bss_3_mid_rand,bss_4_mid_rand,bss_5_mid_rand])),
            "bss_mid_indx": (['i'], np.hstack([bss_1_mid_indx,bss_2_mid_indx,bss_3_mid_indx,bss_4_mid_indx,bss_5_mid_indx])),
            "bss_mid_val": (['i'], np.hstack([bss_1_mid_vals,bss_2_mid_vals,bss_3_mid_vals,bss_4_mid_vals,bss_5_mid_vals])),

            "bss_bot_random": (['g'], np.hstack([bss_1_bot_rand,bss_2_bot_rand,bss_3_bot_rand,bss_4_bot_rand,bss_5_bot_rand])),
            "bss_top_random": (['h'], np.hstack([bss_1_top_rand,bss_2_top_rand,bss_3_top_rand,bss_4_top_rand,bss_5_top_rand])),
            "bss_bot_indx": (['g'], np.hstack([bss_1_bot_indx,bss_2_bot_indx,bss_3_bot_indx,bss_4_bot_indx,bss_5_bot_indx])),
            "bss_top_indx": (['h'], np.hstack([bss_1_top_indx,bss_2_top_indx,bss_3_top_indx,bss_4_top_indx,bss_5_top_indx])),
            "bss_bot_val": (['g'], np.hstack([bss_1_bot_vals,bss_2_bot_vals,bss_3_bot_vals,bss_4_bot_vals,bss_5_bot_vals])),
            "bss_top_val": (['h'], np.hstack([bss_1_top_vals,bss_2_top_vals,bss_3_top_vals,bss_4_top_vals,bss_5_top_vals])),

            "auc_mid_random": (['e'], np.hstack([auc_1_mid_rand,auc_2_mid_rand,auc_3_mid_rand,auc_4_mid_rand,auc_5_mid_rand])),
            "csi_mid_random": (['f'], np.hstack([csi_1_mid_rand,csi_2_mid_rand,csi_3_mid_rand,csi_4_mid_rand,csi_5_mid_rand])),
            "auc_mid_indx": (['e'], np.hstack([auc_1_mid_indx,auc_2_mid_indx,auc_3_mid_indx,auc_4_mid_indx,auc_5_mid_indx])),
            "csi_mid_indx": (['f'], np.hstack([csi_1_mid_indx,csi_2_mid_indx,csi_3_mid_indx,csi_4_mid_indx,csi_5_mid_indx])),
            "auc_mid_val": (['e'], np.hstack([auc_1_mid_vals,auc_2_mid_vals,auc_3_mid_vals,auc_4_mid_vals,auc_5_mid_vals])),
            "csi_mid_val": (['f'], np.hstack([csi_1_mid_vals,csi_2_mid_vals,csi_3_mid_vals,csi_4_mid_vals,csi_5_mid_vals])),    

            "auc_bot_random": (['a'], np.hstack([auc_1_bot_rand,auc_2_bot_rand,auc_3_bot_rand,auc_4_bot_rand,auc_5_bot_rand])),
            "auc_top_random": (['b'], np.hstack([auc_1_top_rand,auc_2_top_rand,auc_3_top_rand,auc_4_top_rand,auc_5_top_rand])),
            "csi_bot_random": (['c'], np.hstack([csi_1_bot_rand,csi_2_bot_rand,csi_3_bot_rand,csi_4_bot_rand,csi_5_bot_rand])),
            "csi_top_random": (['d'], np.hstack([csi_1_top_rand,csi_2_top_rand,csi_3_top_rand,csi_4_top_rand,csi_5_top_rand])),

            "auc_bot_indx": (['a'], np.hstack([auc_1_bot_indx,auc_2_bot_indx,auc_3_bot_indx,auc_4_bot_indx,auc_5_bot_indx])),
            "auc_top_indx": (['b'], np.hstack([auc_1_top_indx,auc_2_top_indx,auc_3_top_indx,auc_4_top_indx,auc_5_top_indx])),
            "csi_bot_indx": (['c'], np.hstack([csi_1_bot_indx,csi_2_bot_indx,csi_3_bot_indx,csi_4_bot_indx,csi_5_bot_indx])),
            "csi_top_indx": (['d'], np.hstack([csi_1_top_indx,csi_2_top_indx,csi_3_top_indx,csi_4_top_indx,csi_5_top_indx])),

            "auc_bot_val": (['a'], np.hstack([auc_1_bot_vals,auc_2_bot_vals,auc_3_bot_vals,auc_4_bot_vals,auc_5_bot_vals])),
            "auc_top_val": (['b'], np.hstack([auc_1_top_vals,auc_2_top_vals,auc_3_top_vals,auc_4_top_vals,auc_5_top_vals])),
            "csi_bot_val": (['c'], np.hstack([csi_1_bot_vals,csi_2_bot_vals,csi_3_bot_vals,csi_4_bot_vals,csi_5_bot_vals])),
            "csi_top_val": (['d'], np.hstack([csi_1_top_vals,csi_2_top_vals,csi_3_top_vals,csi_4_top_vals,csi_5_top_vals])),})
    
    return error_metrics

def grab_testdata(random_number):
    
    """
    Grab future climate test data.
    
    """
    
    file = EvaluateDLModel(climate='future', method='random', variables=np.array(['TK','EV','EU','QVAPOR','PRESS']), 
                             var_directory=f'{future_dl_preprocess}', 
                             model_directory=f'{current_dl_models}', 
                             model_num=25,
                             eval_directory=f'{future_dl_models}', 
                             mask=False, mask_train=False, unbalanced=True, validation=False, isotonic=False, bin_res=0.05,
                             random_choice=random_number, 
                             obs_threshold=0.5, print_sequential=True, 
                             perm_feat_importance=False, pfi_variable=None, outliers=False)

    data = xr.open_dataset(f'{file.eval_directory}/testdata_{file.mask_str}_model{file.model_num}_random{file.random_choice}.nc')
    testdata = data.X_test.astype('float16').values
    data = None
    return testdata

def grab_onerand_patches(variable, error_metrics, random_number):

    filt_auc_top_rand = error_metrics["auc_top_random"].values[error_metrics["auc_top_random"].values==random_number]
    filt_auc_top_indx = error_metrics["auc_top_indx"].values[error_metrics["auc_top_random"].values==random_number]
    filt_auc_bot_rand = error_metrics["auc_bot_random"].values[error_metrics["auc_bot_random"].values==random_number]
    filt_auc_bot_indx = error_metrics["auc_bot_indx"].values[error_metrics["auc_bot_random"].values==random_number]

    filt_csi_top_rand = error_metrics["csi_top_random"].values[error_metrics["csi_top_random"].values==random_number]
    filt_csi_top_indx = error_metrics["csi_top_indx"].values[error_metrics["csi_top_random"].values==random_number]
    filt_csi_bot_rand = error_metrics["csi_bot_random"].values[error_metrics["csi_bot_random"].values==random_number]
    filt_csi_bot_indx = error_metrics["csi_bot_indx"].values[error_metrics["csi_bot_random"].values==random_number]

    filt_auc_mid_rand = error_metrics["auc_mid_random"].values[error_metrics["auc_mid_random"].values==random_number]
    filt_auc_mid_indx = error_metrics["auc_mid_indx"].values[error_metrics["auc_mid_random"].values==random_number]
    filt_csi_mid_rand = error_metrics["csi_mid_random"].values[error_metrics["csi_mid_random"].values==random_number]
    filt_csi_mid_indx = error_metrics["csi_mid_indx"].values[error_metrics["csi_mid_random"].values==random_number]

    filt_bss_top_rand = error_metrics["bss_top_random"].values[error_metrics["bss_top_random"].values==random_number]
    filt_bss_top_indx = error_metrics["bss_top_indx"].values[error_metrics["bss_top_random"].values==random_number]
    filt_bss_bot_rand = error_metrics["bss_bot_random"].values[error_metrics["bss_bot_random"].values==random_number]
    filt_bss_bot_indx = error_metrics["bss_bot_indx"].values[error_metrics["bss_bot_random"].values==random_number]
    filt_bss_mid_rand = error_metrics["bss_mid_random"].values[error_metrics["bss_mid_random"].values==random_number]
    filt_bss_mid_indx = error_metrics["bss_mid_indx"].values[error_metrics["bss_mid_random"].values==random_number]

    originals_auc_top = {}
    originals_auc_top_preshuf = {}
    for c, (a, b) in enumerate(zip(filt_auc_top_rand, filt_auc_top_indx)):
        path = f'{future_dl_models}/comppfi_results_nomask_model25_random{int(a)}_pfivar{variable}_perm{b}.nc'
        data = xr.open_dataset(path)
        originals_auc_top[c] = data.fp_indx.astype('int').values
        originals_auc_top_preshuf[c] = data.orig_indx.astype('int').values[data.fp_indx.astype('int').values]
    originals_auc_top, temp_indx = np.unique(np.array([d for i in originals_auc_top.values() for d in i]), return_index=True)
    originals_auc_top_preshuf = np.array([d for i in originals_auc_top_preshuf.values() for d in i])[temp_indx]

    originals_auc_bot = {}
    originals_auc_bot_preshuf = {}
    for c, (a, b) in enumerate(zip(filt_auc_bot_rand, filt_auc_bot_indx)):
        path = f'{future_dl_models}/comppfi_results_nomask_model25_random{int(a)}_pfivar{variable}_perm{b}.nc'
        data = xr.open_dataset(path)
        originals_auc_bot[c] = data.fp_indx.astype('int').values
        originals_auc_bot_preshuf[c] = data.orig_indx.astype('int').values[data.fp_indx.astype('int').values]
    originals_auc_bot, temp_indx = np.unique(np.array([d for i in originals_auc_bot.values() for d in i]), return_index=True)
    originals_auc_bot_preshuf = np.array([d for i in originals_auc_bot_preshuf.values() for d in i])[temp_indx]

    originals_csi_top = {}
    originals_csi_top_preshuf = {}
    for c, (a, b) in enumerate(zip(filt_csi_top_rand, filt_csi_top_indx)):
        path = f'{future_dl_models}/comppfi_results_nomask_model25_random{int(a)}_pfivar{variable}_perm{b}.nc'
        data = xr.open_dataset(path)
        originals_csi_top[c] = data.fp_indx.astype('int').values
        originals_csi_top_preshuf[c] = data.orig_indx.astype('int').values[data.fp_indx.astype('int').values]
    originals_csi_top, temp_indx = np.unique(np.array([d for i in originals_csi_top.values() for d in i]), return_index=True)
    originals_csi_top_preshuf = np.array([d for i in originals_csi_top_preshuf.values() for d in i])[temp_indx]

    originals_csi_bot = {}
    originals_csi_bot_preshuf = {}
    for c, (a, b) in enumerate(zip(filt_csi_bot_rand, filt_csi_bot_indx)):
        path = f'{future_dl_models}/comppfi_results_nomask_model25_random{int(a)}_pfivar{variable}_perm{b}.nc'
        data = xr.open_dataset(path)
        originals_csi_bot[c] = data.fp_indx.astype('int').values
        originals_csi_bot_preshuf[c] = data.orig_indx.astype('int').values[data.fp_indx.astype('int').values]
    originals_csi_bot, temp_indx = np.unique(np.array([d for i in originals_csi_bot.values() for d in i]), return_index=True)
    originals_csi_bot_preshuf = np.array([d for i in originals_csi_bot_preshuf.values() for d in i])[temp_indx]

    originals_auc_mid = {}
    originals_auc_mid_preshuf = {}
    for c, (a, b) in enumerate(zip(filt_auc_mid_rand, filt_auc_mid_indx)):
        path = f'{future_dl_models}/comppfi_results_nomask_model25_random{int(a)}_pfivar{variable}_perm{b}.nc'
        data = xr.open_dataset(path)
        originals_auc_mid[c] = data.fp_indx.astype('int').values
        originals_auc_mid_preshuf[c] = data.orig_indx.astype('int').values[data.fp_indx.astype('int').values]
    originals_auc_mid, temp_indx = np.unique(np.array([d for i in originals_auc_mid.values() for d in i]), return_index=True)
    originals_auc_mid_preshuf = np.array([d for i in originals_auc_mid_preshuf.values() for d in i])[temp_indx]

    originals_csi_mid = {}
    originals_csi_mid_preshuf = {}
    for c, (a, b) in enumerate(zip(filt_csi_mid_rand, filt_csi_mid_indx)):
        path = f'{future_dl_models}/comppfi_results_nomask_model25_random{int(a)}_pfivar{variable}_perm{b}.nc'
        data = xr.open_dataset(path)
        originals_csi_mid[c] = data.fp_indx.astype('int').values
        originals_csi_mid_preshuf[c] = data.orig_indx.astype('int').values[data.fp_indx.astype('int').values]
    originals_csi_mid, temp_indx = np.unique(np.array([d for i in originals_csi_mid.values() for d in i]), return_index=True)
    originals_csi_mid_preshuf = np.array([d for i in originals_csi_mid_preshuf.values() for d in i])[temp_indx]

    originals_bss_top = {}
    originals_bss_top_preshuf = {}
    for c, (a, b) in enumerate(zip(filt_bss_top_rand, filt_bss_top_indx)):
        path = f'{future_dl_models}/comppfi_results_nomask_model25_random{int(a)}_pfivar{variable}_perm{b}.nc'
        data = xr.open_dataset(path)
        originals_bss_top[c] = data.fp_indx.astype('int').values
        originals_bss_top_preshuf[c] = data.orig_indx.astype('int').values[data.fp_indx.astype('int').values]
    originals_bss_top, temp_indx = np.unique(np.array([d for i in originals_bss_top.values() for d in i]), return_index=True)
    originals_bss_top_preshuf = np.array([d for i in originals_bss_top_preshuf.values() for d in i])[temp_indx]

    originals_bss_bot = {}
    originals_bss_bot_preshuf = {}
    for c, (a, b) in enumerate(zip(filt_bss_bot_rand, filt_bss_bot_indx)):
        path = f'{future_dl_models}/comppfi_results_nomask_model25_random{int(a)}_pfivar{variable}_perm{b}.nc'
        data = xr.open_dataset(path)
        originals_bss_bot[c] = data.fp_indx.astype('int').values
        originals_bss_bot_preshuf[c] = data.orig_indx.astype('int').values[data.fp_indx.astype('int').values]
    originals_bss_bot, temp_indx = np.unique(np.array([d for i in originals_bss_bot.values() for d in i]), return_index=True)
    originals_bss_bot_preshuf = np.array([d for i in originals_bss_bot_preshuf.values() for d in i])[temp_indx]

    originals_bss_mid = {}
    originals_bss_mid_preshuf = {}
    for c, (a, b) in enumerate(zip(filt_bss_mid_rand, filt_bss_mid_indx)):
        path = f'{future_dl_models}/comppfi_results_nomask_model25_random{int(a)}_pfivar{variable}_perm{b}.nc'
        data = xr.open_dataset(path)
        originals_bss_mid[c] = data.fp_indx.astype('int').values
        originals_bss_mid_preshuf[c] = data.orig_indx.astype('int').values[data.fp_indx.astype('int').values]
    originals_bss_mid, temp_indx = np.unique(np.array([d for i in originals_bss_mid.values() for d in i]), return_index=True)
    originals_bss_mid_preshuf = np.array([d for i in originals_bss_mid_preshuf.values() for d in i])[temp_indx]

    path = f'{future_dl_models}/compindx_results_nomask_model25_random{random_number}.nc'
    data = xr.open_dataset(path)
    originals_results = data.tn_indx.astype('int').values
    
    orig_csi_top = originals_csi_top[~np.isin(originals_csi_top, np.unique(np.hstack([originals_csi_mid, originals_csi_bot])))]
    orig_auc_top = originals_auc_top[~np.isin(originals_auc_top, np.unique(np.hstack([originals_auc_mid, originals_auc_bot])))]
    orig_bss_top = originals_bss_top[~np.isin(originals_bss_top, np.unique(np.hstack([originals_bss_mid, originals_bss_bot])))]

    orig_csi_tp = orig_csi_top[np.isin(orig_csi_top,originals_results)]
    orig_auc_tp = orig_auc_top[np.isin(orig_auc_top,originals_results)]
    orig_bss_tp = orig_bss_top[np.isin(orig_bss_top,originals_results)]

    orig_csi_top_preshuf = originals_csi_top_preshuf[~np.isin(originals_csi_top, np.unique(np.hstack([originals_csi_mid, originals_csi_bot])))]
    orig_auc_top_preshuf = originals_auc_top_preshuf[~np.isin(originals_auc_top, np.unique(np.hstack([originals_auc_mid, originals_auc_bot])))]
    orig_bss_top_preshuf = originals_bss_top_preshuf[~np.isin(originals_bss_top, np.unique(np.hstack([originals_bss_mid, originals_bss_bot])))]

    orig_csi_tp_preshuf = orig_csi_top_preshuf[np.isin(orig_csi_top,originals_results)]
    orig_auc_tp_preshuf = orig_auc_top_preshuf[np.isin(orig_auc_top,originals_results)]
    orig_bss_tp_preshuf = orig_bss_top_preshuf[np.isin(orig_bss_top,originals_results)]

    orig_csi_bot = originals_csi_bot[~np.isin(originals_csi_bot, np.unique(np.hstack([originals_csi_mid, originals_csi_top])))]
    orig_auc_bot = originals_auc_bot[~np.isin(originals_auc_bot, np.unique(np.hstack([originals_auc_mid, originals_auc_top])))]
    orig_bss_bot = originals_bss_bot[~np.isin(originals_bss_bot, np.unique(np.hstack([originals_bss_mid, originals_bss_top])))]

    orig_csi_bt = orig_csi_bot[np.isin(orig_csi_bot,originals_results)]
    orig_auc_bt = orig_auc_bot[np.isin(orig_auc_bot,originals_results)]
    orig_bss_bt = orig_bss_bot[np.isin(orig_bss_bot,originals_results)]

    orig_csi_bot_preshuf = originals_csi_bot_preshuf[~np.isin(originals_csi_bot, np.unique(np.hstack([originals_csi_mid, originals_csi_top])))]
    orig_auc_bot_preshuf = originals_auc_bot_preshuf[~np.isin(originals_auc_bot, np.unique(np.hstack([originals_auc_mid, originals_auc_top])))]
    orig_bss_bot_preshuf = originals_bss_bot_preshuf[~np.isin(originals_bss_bot, np.unique(np.hstack([originals_bss_mid, originals_bss_top])))]

    orig_csi_bt_preshuf = orig_csi_bot_preshuf[np.isin(orig_csi_bot,originals_results)]
    orig_auc_bt_preshuf = orig_auc_bot_preshuf[np.isin(orig_auc_bot,originals_results)]
    orig_bss_bt_preshuf = orig_bss_bot_preshuf[np.isin(orig_bss_bot,originals_results)]

    testdata = grab_testdata(random_number)
    
    orig_auc_bt_random = testdata[orig_auc_bt.astype('int'),:,:,:].astype('float32')
    orig_auc_btpm_random = testdata[orig_auc_bt_preshuf.astype('int'),:,:,:].astype('float32')
    orig_auc_tp_random = testdata[orig_auc_tp.astype('int'),:,:,:].astype('float32')
    orig_auc_tppm_random = testdata[orig_auc_tp_preshuf.astype('int'),:,:,:].astype('float32')
    
    orig_csi_bt_random = testdata[orig_csi_bt.astype('int'),:,:,:].astype('float32')
    orig_csi_btpm_random = testdata[orig_csi_bt_preshuf.astype('int'),:,:,:].astype('float32')
    orig_csi_tp_random = testdata[orig_csi_tp.astype('int'),:,:,:].astype('float32')
    orig_csi_tppm_random = testdata[orig_csi_tp_preshuf.astype('int'),:,:,:].astype('float32')
    
    orig_bss_bt_random = testdata[orig_bss_bt.astype('int'),:,:,:].astype('float32')
    orig_bss_btpm_random = testdata[orig_bss_bt_preshuf.astype('int'),:,:,:].astype('float32')
    orig_bss_tp_random = testdata[orig_bss_tp.astype('int'),:,:,:].astype('float32')
    orig_bss_tppm_random = testdata[orig_bss_tp_preshuf.astype('int'),:,:,:].astype('float32')
    
    pfi_patches = xr.Dataset({
            "orig_auc_bt": (['a', 'x', 'y', 'var'], orig_auc_bt_random),
            "orig_auc_btpm": (['a', 'x', 'y', 'var'], orig_auc_btpm_random),
            "orig_auc_tp": (['aa', 'x', 'y', 'var'], orig_auc_tp_random),
            "orig_auc_tppm": (['aa', 'x', 'y', 'var'], orig_auc_tppm_random),
        
            "orig_csi_bt": (['b', 'x', 'y', 'var'], orig_csi_bt_random),
            "orig_csi_btpm": (['b', 'x', 'y', 'var'], orig_csi_btpm_random),
            "orig_csi_tp": (['bb', 'x', 'y', 'var'], orig_csi_tp_random),
            "orig_csi_tppm": (['bb', 'x', 'y', 'var'], orig_csi_tppm_random),
        
            "orig_bss_bt": (['c', 'x', 'y', 'var'], orig_bss_bt_random),
            "orig_bss_btpm": (['c', 'x', 'y', 'var'], orig_bss_btpm_random),
            "orig_bss_tp": (['cc', 'x', 'y', 'var'], orig_bss_tp_random),
            "orig_bss_tppm": (['cc', 'x', 'y', 'var'], orig_bss_tppm_random),})
    
    return pfi_patches
