from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from sktime.datatypes._panel._convert import from_nested_to_2d_array
from sktime.datasets import load_from_tsfile_to_dataframe
from keras.models import Sequential
import matplotlib.pyplot as plt
import matplotlib
import os
from keras.layers import Dense
import seaborn as sns
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
'''Data Loading'''


### Other time-series classification as in NIPS###


def load_ts_dataset(filename):
    # All stored in "Data/"
    train_x, train_y = load_from_tsfile_to_dataframe(
        os.path.join("Data", f"{filename}_TRAIN.ts")
    )
    test_x, test_y = load_from_tsfile_to_dataframe(
        os.path.join("Data", f"{filename}_TEST.ts")
    )
    drop_idx = []  # Sometime the test_x has different feature from the training data
    if filename == 'MelbournePedestrian':
        drop_idx = [1836, 2110]
    test_x = test_x.drop(drop_idx)
    test_y = np.delete(test_y, drop_idx)
    # transform to np.arrays
    train_x = from_nested_to_2d_array(train_x).to_numpy()
    test_x = from_nested_to_2d_array(test_x).to_numpy()
    train_y = train_y.astype(int)
    test_y = test_y.astype(int)
    if min(train_y) == 1:
        # Meaning that start at class 1, not 0
        train_y = train_y - 1
        test_y = test_y - 1
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.histplot(train_y, kde=True, bins=max(train_y + 1), ax=ax)
    # sns.histplot(test_y, kde=True, bins=20, ax=ax[1])
    ax.set_xlabel('Class')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # for i in [0, 1]:
    #     ax[i].set_xlabel('Class')
    #     ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(f'{filename}_histogram.pdf', dpi=300,
                bbox_inches='tight', pad_inches=0)
    K = np.max(train_y) - np.min(train_y) + 1
    return [train_x, test_x, train_y, test_y, K]


'''Data Processing'''


def one_hot_encode_feature(data, categorical_features):
    '''
    Encode original columns of categorical observations
    Replace these categorical columns with their one-hot encoded versions
    '''
    for feature in categorical_features:
        print(feature)
        col_val = data[feature]
        encoded_col = pd.get_dummies(col_val, prefix=feature)
        data = data.drop(feature, axis='columns')
        data = pd.concat([data, encoded_col], axis=1)
    return data


def OneHotEncode_with_missing_category(data, enc, K):
    # data is pre-transformed
    tot_classes = np.arange(K)
    not_in_category = []
    for i in range(K):
        if i not in data:
            not_in_category.append(i)
    if len(not_in_category) > 0:
        print(
            f'These categories are not in plot_marginal_coveragedata \n : {not_in_category}')
    # All that occurred in data
    tot_classes = np.delete(tot_classes, not_in_category)
    data_vec = enc.fit_transform(data.reshape(-1, 1)).toarray()
    data_vec_actual = np.zeros((len(data), K))
    for i in range(K):
        idx = np.where(i == tot_classes)[0]
        if len(idx) > 0:
            idx = idx[0]
        else:
            continue
        data_vec_actual[:, i] = data_vec[:, idx]
    return [data_vec_actual, enc]


'''Neural Networks Classifier'''
# :(. It seems image classifiers require inputs to be images, so that using something like VGG16 does not make sense.
# Yet, I can still use reasonably deep neural networks


def keras_classifier(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(500, activation='relu', input_dim=input_dim))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


'''ERAPS Helpers '''


def generate_bootstrap_samples(n, m, B):
    '''
      Return: B-by-m matrix, where row b gives the indices for b-th bootstrap sample
    '''
    samples_idx = np.zeros((B, m), dtype=int)
    for b in range(B):
        sample_idx = np.random.choice(n, m)
        samples_idx[b, :] = sample_idx
    return(samples_idx)


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S*n, n))


def predict_proba_ordered(probs, classes_, all_classes):
    """
    Description:
        Make sure ALL models return probabilities for ALL classes, because bootstrap models sometimes do not access ALL labels that Y can take in training
        This function comes from: https://stackoverflow.com/questions/30036473/can-i-explicitly-set-the-list-of-possible-classes-for-an-sklearn-svm
    Inputs:
        probs: list of probabilities, output of predict_proba
        classes_: clf.classes_
        all_classes: all possible classes (superset of classes_)
    """
    proba_ordered = np.zeros(
        (probs.shape[0], all_classes.size),  dtype=np.float)
    # http://stackoverflow.com/a/32191125/395857
    sorter = np.argsort(all_classes)
    idx = sorter[np.searchsorted(all_classes, classes_, sorter=sorter)]
    proba_ordered[:, idx] = probs
    return proba_ordered


'''Plot '''
# TODO: a bunch of plotting functions and functions that examine conditional coverage OR just marginal coverage


def examine_coverage(labels, prediction_sets, examined_idx):
    '''
    Input:
        examined_idx: a list of indices at which coverage is examined
    Description:
        Examine ERAPS Prediction Set Coverage on particular set of 'examined_idx'.'''
    # Note, prediction_sets is a DICTIONARY of {index:sets}
    failed_idxs = []  # Contain indices not covering true values
    for idx in examined_idx:
        if labels[idx] not in prediction_sets[idx]:
            failed_idxs.append(idx)
    coverage = (len(examined_idx)-len(failed_idxs))/len(examined_idx)
    return [coverage, failed_idxs]


def plot_marginal_coverage(regr_name, alphas, alpha_marginal_ls, dataset):
    # NOTE: alpha_marginal_ls first contain dictionary of results for ERAPS, then for SRAPS
    plt.rcParams.update({'font.size': 18})
    # For my own paper
    figsize = (14, 4)
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True)
    ls = []
    for i in range(len(alpha_marginal_ls)):
        alpha_marginal = alpha_marginal_ls[i]
        marginal_covs = [alpha_marginal[alpha]['Marginal Coverage']
                         for alpha in alphas]
        marginal_set_size = [alpha_marginal[alpha]
                             ['Average Set Size'] for alpha in alphas]
        ls.append(marginal_set_size)
        i0 = 0
        i1 = 1
        mtd_label = 'ERAPS'
        mtd_color = 'red'
        if i == 1:
            mtd_label = 'SRAPS'
            mtd_color = 'blue'
        ax[i0].plot(alphas, marginal_set_size, linestyle='-',
                    marker='o', color=mtd_color, label=mtd_label)
        ax[i1].plot(alphas, marginal_covs, linestyle='-',
                    marker='o', color=mtd_color, label=mtd_label)
        ticks = np.round(np.linspace(0.01, 0.2, 5), 2)
        ax[i0].xaxis.set_ticks(ticks)
        ax[i0].tick_params(axis='x', labelrotation=45)
        ax[i0].set_xlabel(r'$\alpha$')
        ax[i0].set_ylabel('Set Size')
        ax[i1].xaxis.set_ticks(ticks)
        ax[i1].tick_params(axis='x', labelrotation=45)
        ax[i1].set_xlabel(r'$\alpha$')
        ax[i1].set_ylabel('Coverage')
        # if dataset == 'incidence_report':
        fig.suptitle(f'{regr_name}: Marginal Coverage/Size', y=0.95)
        fig.tight_layout()  # Place this BEFORE legend, otherwise is weird
    max_val = np.ceil(max([max(i) for i in ls]))
    ax[i0].set_ylim(0, max_val)
    ax[i0].yaxis.set_major_locator(MaxNLocator(integer=True))
    ax[i1].plot(alphas, 1-np.array(alphas), linestyle='-',
                marker='o', color='green', label='Target Coverage')
    ax[i1].set_ylim(min(min(1-np.array(alphas)), min(marginal_covs))-0.03, 1)
    # if dataset == 'incidence_report':
    ax[i1].legend(loc='upper center',
                  bbox_to_anchor=(-0.08, -0.5), ncol=3)  # Below plot
    # else:
    #     ax[i1].legend(loc='center',
    #                   bbox_to_anchor=(1.3, 0.15), ncol=1, title=f'{regr_name}')  # Below plot
    fig.savefig(f'{regr_name}_marginal_coverage_{dataset}.pdf', dpi=300, bbox_inches='tight',
                pad_inches=0)
