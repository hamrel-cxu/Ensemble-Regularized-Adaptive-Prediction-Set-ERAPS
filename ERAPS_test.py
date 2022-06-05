import matplotlib
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import OneHotEncoder
import ERAPS_class as ERAPS
from sklearn.ensemble import RandomForestClassifier
import sys
import importlib as ipb
import utils_ERAPS as utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pickle
import seaborn as sns
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# Run ERPAS on fire size data:


''' Helpers '''


def run_over_hyperparameters(lam_ls, kreg_ls, regr, regr_name, dataset, first_run=False):
    '''
        Get Heatmap of marginal coverage and set sizes over the regularization parameters
        NOTE, we assume that the LOO predictions on training and test data have been made so that we just need to load them and calibrate the non-conformity scores
    '''
    alpha = 0.05
    Y_tk_train_actual = Y_tk_train.copy()
    X_tk_train_actual = X_tk_train.copy()
    Lams = len(lam_ls)
    Kregs = len(kreg_ls)
    ERAPS_cov_results = np.zeros((Lams, Kregs))
    ERAPS_size_results = np.zeros((Lams, Kregs))
    SRAPS_cov_results = np.zeros((Lams, Kregs))
    SRAPS_size_results = np.zeros((Lams, Kregs))
    for l in range(Lams):
        for k in range(Kregs):
            lam = lam_ls[l]
            kreg = kreg_ls[k]
            start = time.time()
            if regr.__class__.__name__ == 'Sequential':
                # Need to ecnode Y as an array of vectors of size K
                # NOTE: it is possible that the training or testing data have missing category. Hence, I need to insert a 0 column in such cases
                enc = OneHotEncoder(handle_unknown='ignore')
                Y_tk_train_vec, enc = utils.OneHotEncode_with_missing_category(
                    Y_tk_train_actual, enc, K)
                # print(Y_tk_train_vec.shape)  # Should be T-by-K
                regr_result = ERAPS.RPAS_class(regr, X_tk_train_actual, X_tk_test, Y_tk_train_vec,
                                               Y_tk_test, alpha, [lam, kreg], agg, B, s, K, encoder=enc)
            else:
                regr_result = ERAPS.RPAS_class(regr, X_tk_train_actual, X_tk_test, Y_tk_train_actual,
                                               Y_tk_test, alpha, [lam, kreg], agg, B, s, K, encoder=[])
            ERAPS_probas_test_sorted_s, ERAPS_probas_labels_s = regr_result.get_scores_tau_i(
                dataset, first_run=first_run)
            if regr.__class__.__name__ == 'Sequential':
                P_T1 = regr_result.SRAPS_tau_cal(dataset)
                sets_SRPAS = regr_result.SRAPS_prediction_sets_marginal(
                    P_T1, alpha)
            else:
                P_T1 = regr_result.SRAPS_tau_cal(dataset)
                sets_SRPAS = regr_result.SRAPS_prediction_sets_marginal(
                    P_T1, alpha)
            sets_ERPAS = regr_result.ERAPS_prediction_sets_marginal(
                ERAPS_probas_test_sorted_s, ERAPS_probas_labels_s, alpha)
            mc_examined_idx = np.arange(len(Y_tk_test))
            coverage_SRAPS, _ = utils.examine_coverage(
                Y_tk_test, sets_SRPAS, mc_examined_idx)
            coverage_ERAPS, _ = utils.examine_coverage(
                Y_tk_test, sets_ERPAS, mc_examined_idx)
            ave_size_SRPAS = np.mean([len(sets_SRPAS[i])
                                      for i in mc_examined_idx])
            ave_size_ERPAS = np.mean([len(sets_ERPAS[i])
                                      for i in mc_examined_idx])
            ERAPS_cov_results[l, k] = coverage_ERAPS
            ERAPS_size_results[l, k] = ave_size_ERPAS
            SRAPS_cov_results[l, k] = coverage_SRAPS
            SRAPS_size_results[l, k] = ave_size_SRPAS
            print(
                f'Finish running {regr_name} with lam:{lam} & kreg:{kreg}, took {time.time()-start} seconds')
    return [ERAPS_cov_results, ERAPS_size_results, SRAPS_cov_results, SRAPS_size_results]


def run_over_alpha(downsample, alphas, regr, regr_name, penalties, dataset, type='ERAPS', first_run=False, marginal=True, strata_ls=[]):
    '''
        Downsample = True/False.
            If true, try downsampling of Y_tk_train to get reasonable conditional coverage, because classes 4 and 5 indicating large fires have rare occurrences
        regr_name: save name for results
            It is RF or NN_classifiers
        penalties = [lam,kreg]
            lam: Penalty for violation
            kreg: Allow # of entries in a prediction set
    '''
    # 1. Data processing and get LOO predictions for ERAPS
    lam, kreg = penalties
    # Currently set penalties based on Candes Proposition 2
    if downsample:
        print('Downsamping to get good conditional coverage, \n but too few data for training so marginal coverage can be poor')
        Y_tk_train_down = Y_tk_train.copy()
        X_tk_train_down = X_tk_train.copy()
        # Default is incidence report data
        label_ls = [0, 1, 2]
        few_label_occur = 10  # Sum occurrences of class 3 and 4, which are rare
        if dataset == 'general_fire':
            few_label_occur = 18
            label_ls = np.arange(5)
        for label in label_ls:
            label_retain = np.where(Y_tk_train_down != label)[0]
            label_idx = np.where(Y_tk_train_down == label)[0]
            label_idx_retain = np.random.choice(
                label_idx, few_label_occur, replace=False)
            label_retain = np.append(label_retain, label_idx_retain)
            Y_tk_train_down = Y_tk_train_down[label_retain]
            X_tk_train_down = X_tk_train_down[label_retain]
        print(X_tk_train_down.shape)
        Y_tk_train_actual = Y_tk_train_down
        X_tk_train_actual = X_tk_train_down
        fig, ax = plt.subplots(figsize=(14, 3))
        sns.histplot(Y_tk_train_actual, kde=True, bins=K, ax=ax)
        ax.set_xlabel('Class')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.savefig(f'{dataset}_histogram_downsample.pdf',
                    dpi=300, bbox_inches='tight', pad_inches=0)
        unique, counts = np.unique(Y_tk_train_actual, return_counts=True)
        occur_dict = {u: c for (u, c) in zip(unique, counts)}
        print('See below {class : num occurrences} to see downsampled data')
        print(occur_dict)
    else:
        print('Fitting on ALL training data. No downsampling so conditional coverage can be poor')
        Y_tk_train_actual = Y_tk_train.copy()
        X_tk_train_actual = X_tk_train.copy()
    alpha_placeholder = 0
    if regr.__class__.__name__ == 'Sequential':
        # Need to ecnode Y as an array of vectors of size K
        # NOTE: it is possible that the training or testing data have missing category. Hence, I need to insert a 0 column in such cases
        enc = OneHotEncoder(handle_unknown='ignore')
        Y_tk_train_vec, enc = utils.OneHotEncode_with_missing_category(
            Y_tk_train_actual, enc, K)
        print(Y_tk_train_vec.shape)  # Should be T-by-K
        regr_result = ERAPS.RPAS_class(regr, X_tk_train_actual, X_tk_test, Y_tk_train_vec,
                                       Y_tk_test, alpha_placeholder, [lam, kreg], agg, B, s, K, encoder=enc)
    else:
        regr_result = ERAPS.RPAS_class(regr, X_tk_train_actual, X_tk_test, Y_tk_train_actual,
                                       Y_tk_test, alpha_placeholder, [lam, kreg], agg, B, s, K, encoder=[])
    # return regr_result
    # Train LOO predictor for ERAPS. Take some time but just need to run once
    if type == 'ERAPS':
        ERAPS_probas_test_sorted_s, ERAPS_probas_labels_s = regr_result.get_scores_tau_i(
            dataset, first_run=first_run)
    elif type == 'SRAPS':
        P_T1 = regr_result.SRAPS_tau_cal(dataset)
    elif type == 'SAPS':
        P_T1 = regr_result.SAPS_tau_cal()
    else:
        P_T1 = regr_result.naive_train_mod()
    # 2. Run to get marginal and conditional coverage
    alpha_marginal = {}
    # Each row records conditional coverage on that label
    alpha_to_cond_coverage = np.zeros((len(alphas), K))
    alpha_to_cond_size = np.zeros((len(alphas), K))
    alpha_to_cond_coverage_strata = np.zeros((len(alphas), len(strata_ls)))
    alpha_marginal_failed_idxs = {}
    for num in np.arange(len(alphas)):
        alpha = alphas[num]
        # NOTE: The label with the most occurrence in training almost always gets the highest predicted probability.
        # return regr_result
        if type == 'ERAPS':
            if marginal:
                print('Running ERAPS, marginal version')
                sets = regr_result.ERAPS_prediction_sets_marginal(
                    ERAPS_probas_test_sorted_s, ERAPS_probas_labels_s, alpha)
            else:
                print('Running ERAPS, class-conditional version')
                sets, binned_scores_train, binned_scores_predict = regr_result.ERAPS_prediction_sets_conditional(
                    ERAPS_probas_test_sorted_s, ERAPS_probas_labels_s, alpha)
        elif type == 'SRAPS':
            if marginal:
                print('Running SRAPS, marginal version')
                sets = regr_result.SRAPS_prediction_sets_marginal(
                    P_T1, alpha)
            else:
                print('Running SRAPS, class-conditional version')
                sets = regr_result.SRAPS_prediction_sets_conditional(
                    P_T1, alpha)
        elif type == 'SAPS':
            if marginal:
                print('Running SAPS, marginal version')
                sets = regr_result.SAPS_prediction_sets_marginal(
                    P_T1, alpha)
            else:
                raise ValueError(
                    'Not yet implemented SAPS class-cond. score version')
        else:
            if marginal:
                print('Running naive, marginal version')
                sets = regr_result.naive_prediction_sets_marginal(
                    P_T1, alpha)
            else:
                raise ValueError(
                    'Not yet implemented naive class-cond. score  version')
        # Directly check the number of 0 in each class
        # Examine coverage & indices at which the prediction sets do not cover, by using labels=Y_tk_test & sets above
        # 1. Marginal Coverage
        mc_examined_idx = np.arange(len(Y_tk_test))
        coverage, failed_idxs = utils.examine_coverage(
            Y_tk_test, sets, mc_examined_idx)
        print(
            f'{type}: Marginal Coverage for alpha={alpha} is {np.round(coverage*100,2)}%')
        ave_size = np.mean([len(sets[i])
                            for i in mc_examined_idx])  # Fractional
        alpha_marginal[alpha] = {
            'Marginal Coverage': coverage, 'Average Set Size': ave_size}
        alpha_marginal_failed_idxs[alpha] = {
            'Failed Indices': np.array(failed_idxs)}
        # 2. Class-Conditional Coverage
        for lab in np.arange(K):
            cond_examined_idx = np.where(Y_tk_test == lab)[0]
            if len(cond_examined_idx) == 0:
                continue
            else:
                cond_coverage, _ = utils.examine_coverage(
                    Y_tk_test, sets, cond_examined_idx)
            alpha_to_cond_coverage[num, lab] = cond_coverage
            ave_cond_size = np.mean([len(sets[i])
                                     for i in cond_examined_idx])  # Fractional
            alpha_to_cond_size[num, lab] = ave_cond_size
        # 3. Set-stratefied Conditional Coverage
        set_sizes = [len(sets[i]) for i in range(len(sets))]
        for i, strata in enumerate(strata_ls):
            cond_examined_idx = np.where(
                (set_sizes >= strata[0]) & (set_sizes < strata[1]))[0]
            if len(cond_examined_idx) == 0:
                continue
            else:
                cond_coverage, _ = utils.examine_coverage(
                    Y_tk_test, sets, cond_examined_idx)
            alpha_to_cond_coverage_strata[num, i] = cond_coverage
    # Conditional Coverage: Show table of coverage and width.
    # Each row is indexed by alpha, and the first 5 are coverage,
    # The last five are for width.
    if downsample:
        regr_name += '_downsample'
    regr_name += type
    alpha_cond_cov_width = np.r_[alpha_to_cond_coverage, alpha_to_cond_size]
    alpha_cond_cov_width = np.round(
        np.c_[np.tile(alphas, 2), alpha_cond_cov_width], 2)
    alpha_to_cond_coverage_strata = np.round(alpha_to_cond_coverage_strata, 2)
    # np.savetxt(f"cond_cov_{regr_name}_{dataset}.csv",
    #            alpha_cond_cov_width, delimiter=",", fmt='%.2f')
    col_names = ['alpha'] + [f'class {i}' for i in range(K)]
    indices = np.concatenate(
        [np.repeat('Coverage', len(alphas)), np.repeat('Size', len(alphas))])
    alpha_cond_cov_width = pd.DataFrame(
        alpha_cond_cov_width, columns=col_names, index=indices)
    indices = np.repeat('Coverage', len(alphas))
    alpha_to_cond_coverage_strata = pd.DataFrame(
        alpha_to_cond_coverage_strata, index=indices)
    alpha_to_cond_coverage_strata.to_csv(
        f"cond_cov_{regr_name}_{dataset}_set_stratified.csv", index=False)
    if downsample == False and dataset in ['incidence_report', 'general_fire']:
        if marginal:
            alpha_cond_cov_width.to_csv(
                f"Disregard_cond_cov_{regr_name}_{dataset}.csv", index=False)
        else:
            alpha_cond_cov_width.to_csv(
                f"Disregard_cond_cov_{regr_name}_{dataset}_class_conditional.csv", index=False)
    else:
        if marginal:
            alpha_cond_cov_width.to_csv(
                f"cond_cov_{regr_name}_{dataset}.csv", index=False)
        else:
            alpha_cond_cov_width.to_csv(
                f"cond_cov_{regr_name}_{dataset}_class_conditional.csv", index=False)
    print(alpha_cond_cov_width)
    print(alpha_to_cond_coverage_strata)
    if marginal:
        return [alpha_marginal, alpha_cond_cov_width]
    else:
        if type == 'ERAPS':
            return [alpha_marginal, alpha_cond_cov_width, binned_scores_train, binned_scores_predict]
        else:
            return [alpha_marginal, alpha_cond_cov_width]


'''All together '''

'''1. Marginal coverage over regularization parameters'''
agg = np.mean  # Aggregation function. It was median for incidence report
B = 30  # Total # bootstrap models
s = 1  # Stride
RF_classifier = RandomForestClassifier(random_state=0, bootstrap=False)
regr_dic = {'RF': RF_classifier}
# for dataset in ['MelbournePedestrian', 'PenDigits', 'Crop']:
first_run = False
for dataset in ['Crop', 'PenDigits', 'MelbournePedestrian']:
    # Prepare data
    X_tk_train, X_tk_test, Y_tk_train, Y_tk_test, K = utils.load_ts_dataset(
        dataset)
    idx_keep = ~np.isnan(X_tk_train).any(axis=1)
    X_tk_train = X_tk_train[idx_keep, :]
    Y_tk_train = Y_tk_train[idx_keep]
    idx_keep = ~np.isnan(X_tk_test).any(axis=1)
    X_tk_test = X_tk_test[idx_keep, :]
    Y_tk_test = Y_tk_test[idx_keep]
    print(K)
    lam_ls = np.linspace(0.01, 10, num=10, endpoint=True)
    kreg_ls = np.linspace(1, K - 1, num=10, endpoint=True)
    unique, counts = np.unique(Y_tk_train, return_counts=True)
    occur_dict = {u: c for (u, c) in zip(unique, counts)}
    print(
        'See below {class : num occurrences} to decide which classes are rare in training data')
    print(occur_dict)
    unique, counts = np.unique(Y_tk_test, return_counts=True)
    occur_dict = {u: c for (u, c) in zip(unique, counts)}
    print(
        'See below {class : num occurrences} to decide which classes are rare in TEST data')
    print(occur_dict)
    dataset_results = {}
    # for regr_name in ['RF', 'NN']:
    for regr_name in ['NN']:
        if regr_name == 'NN':
            classifier = utils.keras_classifier(
                input_dim=X_tk_train.shape[1], num_classes=K)
            # NN_classifier.summary()  # See number of parameters and details
        else:
            classifier = regr_dic[regr_name]
        ERAPS_cov_results, ERAPS_size_results, SRAPS_cov_results, SRAPS_size_results = run_over_hyperparameters(
            lam_ls, kreg_ls, classifier, regr_name, dataset, first_run=first_run)
        dataset_results[regr_name] = [ERAPS_cov_results,
                                      ERAPS_size_results, SRAPS_cov_results, SRAPS_size_results]
    with open(f'{dataset}_marginal_results_over_regularizers_{regr_name}.pickle', 'wb') as handle:
        pickle.dump(dataset_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Plot


def plot_heatmap(dataset, regr_name):
    K_dict = {'MelbournePedestrian': 10, 'PenDigits': 10, 'Crop': 24}
    K = K_dict[dataset]
    lam_ls = np.linspace(0.01, 10, num=10, endpoint=True)
    kreg_ls = np.linspace(1, K - 1, num=10, endpoint=True)
    with open(f'{dataset}_marginal_results_over_regularizers.pickle', 'rb') as handle:
        dataset_results = pickle.load(handle)
    ERAPS_cov_results, ERAPS_size_results, SRAPS_cov_results, SRAPS_size_results = dataset_results[
        regr_name]
    ERAPS_cov_results = pd.DataFrame(ERAPS_cov_results, index=np.round(
        lam_ls, 1), columns=np.round(kreg_ls, 1))
    ERAPS_size_results = pd.DataFrame(ERAPS_size_results, index=np.round(
        lam_ls, 1), columns=np.round(kreg_ls, 1))
    SRAPS_cov_results = pd.DataFrame(SRAPS_cov_results, index=np.round(
        lam_ls, 1), columns=np.round(kreg_ls, 1))
    SRAPS_size_results = pd.DataFrame(SRAPS_size_results, index=np.round(
        lam_ls, 1), columns=np.round(kreg_ls, 1))
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(1, 4, figsize=(14, 3))
    cov_vmin = 0.95
    cov_vmax = 1
    size_vmin = 1
    size_vmax = K_dict[dataset]
    cov_kws = {'label': 'Coverage'}
    size_kws = {'label': 'Set size'}
    map_name = 'tab20c'
    sns.heatmap(ERAPS_cov_results, ax=ax[0], xticklabels=True,
                yticklabels=True, vmin=cov_vmin, vmax=cov_vmax, cmap=map_name, cbar_kws=cov_kws)
    sns.heatmap(SRAPS_cov_results, ax=ax[1], xticklabels=True,
                yticklabels=True, vmin=cov_vmin, vmax=cov_vmax, cmap=map_name, cbar_kws=cov_kws)
    sns.heatmap(ERAPS_size_results, ax=ax[2], xticklabels=True,
                yticklabels=True, vmin=size_vmin, vmax=size_vmax, cmap=map_name, cbar_kws=size_kws)
    sns.heatmap(SRAPS_size_results, ax=ax[3], xticklabels=True,
                yticklabels=True, vmin=size_vmin, vmax=size_vmax, cmap=map_name, cbar_kws=size_kws)
    for i in range(4):
        ax[i].set_xlabel(r'$k_{reg}$')
        ax[i].set_ylabel(r'$\lambda$', rotation=0)
        ax[i].tick_params(axis='y', labelrotation=0)
    fig.tight_layout()
    fig.savefig(f'{dataset}_heatmap_{regr_name}.pdf',
                dpi=300, bbox_inches='tight', pad_inches=0)
    return fig


# for regr_name in ['RF', 'NN']:
for regr_name in ['NN']:
    fig_heatmap_MelbournePedestrian = plot_heatmap(
        'MelbournePedestrian', regr_name)
    fig_heatmap_Crop = plot_heatmap('Crop', regr_name)
    fig_heatmap_PenDigits = plot_heatmap('PenDigits', regr_name)

'''2. Marginal & Conditional coverage over alpha
   Cond. cov has both the set-stratified version and the class-conditional version'''
agg = np.mean  # Aggregation function. It was median for incidence report
B = 30  # Total # bootstrap models
s = 1  # Stride
RF_classifier = RandomForestClassifier(random_state=0, bootstrap=False)
regr_dic = {'RF': RF_classifier}
# downsample = True is very fast (because there are few training data), we have True for conditional coverage purposes (e.g. balanced class)
alphas = [0.05, 0.075, 0.1, 0.15, 0.2]
lam = 1  # larger lambda = stronger penalty
kreg = 2  # smaller kreg = stronger penalty
first_run = True  # NOTE: as long as we DO NOT change training size, just run this one is enough for each dataset, regardless of 'marginal' or 'run_SRAPS'
marginal = True  # TODO: not sure why class-conditional is poor EVEN on balanced data
run_SRAPS = True
run_ERAPS = True
run_SAPS = True
run_naive = True
downsample = False
full_results = {}
for dataset in ['Crop', 'PenDigits', 'MelbournePedestrian']:
    # Prepare data
    X_tk_train, X_tk_test, Y_tk_train, Y_tk_test, K = utils.load_ts_dataset(
        dataset)
    idx_keep = ~np.isnan(X_tk_train).any(axis=1)
    X_tk_train = X_tk_train[idx_keep, :]
    Y_tk_train = Y_tk_train[idx_keep]
    idx_keep = ~np.isnan(X_tk_test).any(axis=1)
    X_tk_test = X_tk_test[idx_keep, :]
    Y_tk_test = Y_tk_test[idx_keep]
    print(K)
    # Consider five strata bins
    strata_sep = np.linspace(0, K, num=6, dtype=int)
    strata_ls = [[strata_sep[i], strata_sep[i + 1]]
                 for i in range(len(strata_sep) - 1)]
    unique, counts = np.unique(Y_tk_train, return_counts=True)
    occur_dict = {u: c for (u, c) in zip(unique, counts)}
    print(
        'See below {class : num occurrences} to decide which classes are rare in training data')
    print(occur_dict)
    unique, counts = np.unique(Y_tk_test, return_counts=True)
    occur_dict = {u: c for (u, c) in zip(unique, counts)}
    print(
        'See below {class : num occurrences} to decide which classes are rare in TEST data')
    print(occur_dict)
    # for regr_name in ['RF', 'NN']:
    dataset_results = {}
    for regr_name in ['NN']:
        if regr_name == 'NN':
            classifier = utils.keras_classifier(
                input_dim=X_tk_train.shape[1], num_classes=K)
            # NN_classifier.summary()  # See number of parameters and details
        else:
            classifier = regr_dic[regr_name]
        # Choose which CP method to use
        if run_ERAPS:
            result_ERAPS = run_over_alpha(downsample=downsample, alphas=alphas, regr=classifier,
                                          regr_name=regr_name, penalties=[lam, kreg], dataset=dataset, type='ERAPS', first_run=first_run, marginal=marginal, strata_ls=strata_ls)
        if run_SRAPS:
            result_SRAPS = run_over_alpha(downsample=downsample, alphas=alphas, regr=classifier,
                                          regr_name=regr_name, penalties=[lam, kreg], dataset=dataset, type='SRAPS', marginal=marginal, strata_ls=strata_ls)
        if run_SAPS:
            result_SAPS = run_over_alpha(downsample=downsample, alphas=alphas, regr=classifier,
                                         regr_name=regr_name, penalties=[lam, kreg], dataset=dataset, type='SAPS', marginal=marginal, strata_ls=strata_ls)
        if run_naive:
            result_naive = run_over_alpha(downsample=downsample, alphas=alphas, regr=classifier,
                                          regr_name=regr_name, penalties=[lam, kreg], dataset=dataset, type='naive', marginal=marginal, strata_ls=strata_ls)
        # Choose coverage type
        if marginal:
            if run_ERAPS:
                alpha_marginal_ERAPS, alpha_cond_cov_width_ERAPS = result_ERAPS
            if run_SRAPS:
                alpha_marginal_SRAPS, alpha_cond_cov_width_SRAPS = result_SRAPS
            if run_SAPS:
                alpha_marginal_SAPS, alpha_cond_cov_width_SAPS = result_SAPS
            if run_naive:
                alpha_marginal_naive, alpha_cond_cov_width_naive = result_naive
        if marginal == False:
            if run_ERAPS:
                alpha_marginal_ERAPS, alpha_cond_cov_width_ERAPS, binned_scores_train_ERAPS, binned_scores_predict_ERAPS = result_ERAPS
            if run_SRAPS:
                alpha_marginal_SRAPS, alpha_cond_cov_width_SRAPS = result_SRAPS
        alpha_marginal_ls = []
        if run_ERAPS:
            alpha_marginal_ls.append(alpha_marginal_ERAPS)
        if run_SRAPS:
            alpha_marginal_ls.append(alpha_marginal_SRAPS)
        if run_SAPS:
            alpha_marginal_ls.append(alpha_marginal_SAPS)
        if run_naive:
            alpha_marginal_ls.append(alpha_marginal_naive)
        dataset_results[regr_name] = alpha_marginal_ls
    full_results[dataset] = dataset_results


def get_marginal_latex(full_results):
    for dataset in full_results.keys():
        dataset_results = full_results[dataset]
        for regr_name, result in dataset_results.items():
            print(f'Latex for {dataset}:{regr_name}')
            result_array = np.zeros((4, len(alphas)*2))
            ERAPS_result = result[0]
            SRAPS_result = result[1]
            SAPS_result = result[2]
            naive_result = result[3]
            i = 0
            for key in ERAPS_result.keys():
                ERAPS_cov, ERAPS_size = ERAPS_result[key].values()
                SRAPS_cov, SRAPS_size = SRAPS_result[key].values()
                SAPS_cov, SAPS_size = SAPS_result[key].values()
                naive_cov, naive_size = naive_result[key].values()
                result_array[0, i] = ERAPS_cov
                result_array[0, i+1] = ERAPS_size
                result_array[1, i] = SRAPS_cov
                result_array[1, i+1] = SRAPS_size
                result_array[2, i] = SAPS_cov
                result_array[2, i+1] = SAPS_size
                result_array[3, i] = naive_cov
                result_array[3, i+1] = naive_size
                i += 2
            result_array = pd.DataFrame(
                np.round(result_array, 2), index=[f'\ERAPS', f'\SRAPS', f'\SAPS', f'Naive'], columns=np.repeat(alphas, 2))
            result_array.to_csv(
                f'CP_marginal_results_{dataset}_{regr_name}.csv')
            print(result_array.to_latex(index=False, escape=False))


# NOTE: this is the resulting table for each dataset & regressor pair
get_marginal_latex(full_results)
regr_name = 'NN'
for dataset in ['MelbournePedestrian', 'Crop', 'PenDigits']:
    df = pd.read_csv(f'CP_marginal_results_{dataset}_{regr_name}.csv')
    print(df.to_latex(index=False, escape=False))

############
