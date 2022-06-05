import warnings
import utils_ERAPS as utils
import numpy as np
import keras
import pickle
from keras.models import clone_model
import time
warnings.filterwarnings("ignore")

'''This file implements the Ensemble Regularized Adaptive Prediction Set (ERAPS)'''


class RPAS_class():
    def __init__(self, fit_func, X_train, X_predict, Y_train, Y_predict, alpha, regu_para, agg, B, s, K, encoder=[]):
        '''
        Inputs:
            fit_func: a classifiers that has .fit() method to output a predictor, which has .predict() method.
                      It is assumed that .predict_proba(X_t) gives a probability distribution over all labels.
            alpha: tolerable non-coverage level
            regu_para: [lam, kreg]
            agg: aggregation function, such as mean, median, trimmed mean
            B: total bootstrap models.
            s: stride s that controls how frequently we slide scores
            K: range for Y_t (an integer)
        '''
        np.random.seed(100)
        # From inputs
        self.regressor = fit_func
        self.X_train = X_train
        self.X_predict = X_predict
        self.Y_train = Y_train
        self.Y_predict = Y_predict
        self.alpha = alpha  # Actually not used anymore
        self.lam = regu_para[0]
        self.kreg = regu_para[1]
        self.agg = agg
        self.Unif_RV = np.random.uniform(size=len(X_train) + len(X_predict))
        self.B = B
        self.s = s
        self.K = K
        # To be computed later (NOTE: may NOT need be stored)
        self.ERAPS_score = []  # It will append \tau_i for each i >= 1
        self.SRAPS_score = []
        self.enc = encoder
    '''1. ERAPS '''
    '''A. Fit bootstrap classifiers (which outputs probabilities for labels)'''

    def fit_bootstrap_models_with_P_b(self):
        '''
        Output:
            P_b_s: np.array(T+T1, K, B), which holds predicted probabilities from each bootstrap model on ALL data (including training)
            leave_i_idx: A dictionary {i: {b: i \notin S_b}} that shows which arrays should be aggregated
        Description:
          1. Train B bootstrap estimators from subsets of (X_train, Y_train)
          2. Output probabilities Y_t(c), t=1,...,T+T1, c=1,...,K in a dictionary {b: P_b \in R^(T+T1)-by-K}
          3. (Useful later for aggregation) Save a dictionary {i: {b: i \notin S_b}}
        '''
        T = len(self.X_train)
        K = self.K
        T1 = len(self.X_predict)
        B = self.B
        # hold indices of training data for each f^b
        boot_samples_idx = utils.generate_bootstrap_samples(T, T, B)
        # for i^th column, it shows which f^b uses i in training (so exclude in aggregation)
        in_boot_sample = np.zeros((B, T), dtype=bool)
        P_b_s = np.zeros((T + T1, K, B))  # Save P_b, b=1,...,B
        all_classes = np.arange(K)
        for b in range(B):
            model = self.regressor
            if self.regressor.__class__.__name__ == 'Sequential':
                start1 = time.time()
                model = clone_model(self.regressor)
                model.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
                callback = keras.callbacks.EarlyStopping(
                    monitor='loss', patience=10)
                bsize = int(0.1 * len(np.unique(boot_samples_idx[b])))
                model.fit(self.X_train[boot_samples_idx[b], :], self.Y_train[boot_samples_idx[b], ],
                          epochs=25, batch_size=bsize, callbacks=[callback], verbose=0)
                print(
                    f'Took {time.time()-start1} secs to fit the {b}th boostrap model')
                # Note, "predict_proba_ordered" is not used because Y_predict is already one hot-encoded, so predict returns probability for all classes.
                P_b_s[:, :, b] = model.predict(
                    np.r_[self.X_train, self.X_predict])
            else:
                model = model.fit(self.X_train[boot_samples_idx[b], :],
                                  self.Y_train[boot_samples_idx[b], ])
                P_b_s[:, :, b] = utils.predict_proba_ordered(
                    model.predict_proba(np.r_[self.X_train, self.X_predict]), model.classes_, all_classes)
            # print('converted proba')
            # print(P_b_s[:, :, b][-i:])
            in_boot_sample[b, boot_samples_idx[b]] = True
        leave_i_idx = {}  # Save {i: {b: i \notin S_b}}
        for i in range(T):
            b_keep = np.argwhere(~(in_boot_sample[:, i])).reshape(-1)
            if(len(b_keep) > 0):
                leave_i_idx[i] = b_keep
            else:
                leave_i_idx[i] = []
        return([P_b_s, leave_i_idx])

    '''B. Training LOO ensemble classifiers'''

    def get_LOO_P_i(self):
        '''
        Output:
            P_i_s: np.array(1+T1,K,T), where the FIRST row is LOO probability predictions on X_i from bootstrap models
                   Latter 1,...,T1 rows are for test observations
        Description:
            Get P_-i \in R^(1+T1)-by-K as predictions from LOO models (on X_i & X_t, t>T),
            Using P_b_s and leave_i_idx above.
            NOTE: update: get rather the aggregated prediction directly, as the 3D array is inefficient and hard to handle if T1 and T are large
        '''
        start = time.time()
        P_b_s, leave_i_idx = self.fit_bootstrap_models_with_P_b()
        print(
            f'Training Done under {type(self.regressor).__name__}, took {time.time()-start} secs.')
        start = time.time()
        T = len(self.X_train)
        K = self.K
        P_i_s = np.zeros((K, T))
        LOO_train = []
        for i in range(T):
            # element-wise aggregation of outputs from \pi_b: i \notin S_b
            # This is for Y_i
            # axis=0 because we aggregate over models, NOT labels
            # NOT axis=1 becuse P_b_s[i, :, leave_i_idx[i]] actuall has dimension |{b: i \notin S_b}-by-K|
            P_i_s[:, i] = self.agg(P_b_s[i, :, leave_i_idx[i]], axis=0)
            # Store prediction from each LOO ensemble predictor on the T1 test point. EACH entry is a matrix of size T1-by-K
            LOO_train.append(self.agg(P_b_s[T:, :, leave_i_idx[i]], axis=2))
        # LOO_train is a list of T matrices of size T1-by-K
        # Directly yield a T1-by-K matrix
        probas_test = self.agg(LOO_train, axis=0)
        # probas_test = []
        # # NOTE: This original computation was costly because it is aggregating a T-by-K matrix for each i = 1,...,T1 sequentially...
        # for i in range(T1):
        #     # Get the average of the LOO prediction
        #     probas_test.append(self.agg([LOO_train[j][i] for j in range(T)], axis=0))
        print(
            f'LOO prediction Done, took {time.time()-start} secs.')
        return [P_i_s, probas_test]

    '''C. Get scores \tau^phi_i'''

    def get_scores_tau_i(self, dataset, first_run=False, AD=False):
        '''
        Inputs (updated):
            dataset: name of what I am running
            first_run: if false, then I assume LOO predictions have been made
            AD: if true, then we are fitting on subset of all classes of data for anomaly detection
        Outputs:
            self.ERAPS_score: Scores tau_i, i>=1. An array of length T+T1
            probas_test_sorted_s: sorted test probabilities of size (T1,K), useful in the last step for getting prediction sets
            probas_to_labels: most to least likely predicted of size (T1,K), useful in the last step for getting prediction sets
        Description:
            1. For i<=T,
                Compute the INITIAL set of scores \tau^phi_i using P_i_s[0,:,i].
            2. For i=T+1,...,T+T1,
                Compute the TEST scores \tau^phi_i using quantile(P_i_s[i-T,:,:]) over the last array.
            (NOTE, if truly online prediction, then only 1. can be computed now)
        '''
        T = len(self.X_train)
        K = self.K
        T1 = len(self.X_predict)
        regr_name = self.regressor.__class__.__name__
        if first_run:
            P_i_s, probas_test = self.get_LOO_P_i()
            start = time.time()
            probas_test_sorted_s = np.zeros((T + T1, K))
            probas_labels_s = np.zeros((T + T1, K), dtype=int)
        else:
            if AD:
                with open(f'{dataset}_{regr_name}_AD_LOO_dict_ERAPS.pickle', 'rb') as handle:
                    temp_save = pickle.load(handle)
            else:
                with open(f'{dataset}_{regr_name}_LOO_dict_ERAPS.pickle', 'rb') as handle:
                    temp_save = pickle.load(handle)
            probas_test_sorted_s, probas_labels_s = temp_save.values()
        if regr_name == 'Sequential':
            # Need to transform the one-hot-encoded vectors back to integer vectors
            # NOTE: it is possible that "self.Y_train/Y_predict" have columns with 0 sum, because they are imputed due to missing column. Thus, only transform the rest columns
            train_in = np.where(self.Y_train.sum(axis=0) > 0)[0]
            self.Y_train = self.enc.inverse_transform(
                self.Y_train[:, train_in])
        for i in range(T + T1):
            # Handle BOTH scores in training (i<T) and in testing (i>=T)
            # Goal is to get conformity scores
            if i < T:
                # Initial set of scores
                Y_i = self.Y_train[i]
                if first_run:
                    probas_i = P_i_s[:, i]  # Unsorted now
            else:
                # Test scores, delete if Y_t not known now
                Y_i = self.Y_predict[i - T]
                # axis=1 because we take quantile over T LOO models, NOT labels
                # probas_i = np.percentile(P_i_s[i-T+1, :, :], 100 *
                #                          (1-self.alpha), axis=1)  # Unsorted now
                if first_run:
                    probas_i = probas_test[i - T]  # Unsorted now
            if first_run:
                probas_to_labels = np.argsort(probas_i)[::-1]
                # Sorted, largest to smallest
                probas_i_sorted = np.sort(probas_i)[::-1]
                probas_labels_s[i] = probas_to_labels
                probas_test_sorted_s[i] = probas_i_sorted
            else:
                probas_to_labels = probas_labels_s[i]
                probas_i_sorted = probas_test_sorted_s[i]
            r_i = np.where(Y_i == probas_to_labels)[0]
            if len(r_i) == 0:
                # Happens when doing anomaly detection experiment
                tau_before_slide = self.ERAPS_score[i-T]
                # We do so because this test datum is not a part of training score, so that we skip over it (but keep the score T times ago so that the overall non-conformity score distribution remains the same)
                self.ERAPS_score.append(tau_before_slide)
            else:
                r_i = r_i[0]  # Rank of i
                if r_i == 0:
                    m_i = 0
                else:
                    # Cumulative probabilities, can be zero
                    m_i = np.sum(probas_i_sorted[:r_i])
                true_rank = r_i + 1
                tau_i = m_i + probas_i_sorted[r_i] * self.Unif_RV[i] + \
                    self.lam * np.max((true_rank - self.kreg + 1, 0))
                self.ERAPS_score.append(tau_i)
        if first_run:
            temp_save = {'1': probas_test_sorted_s,
                         '2': probas_labels_s}
            if AD:
                with open(f'{dataset}_{regr_name}_AD_LOO_dict_ERAPS.pickle', 'wb') as handle:
                    pickle.dump(temp_save, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(f'{dataset}_{regr_name}_LOO_dict_ERAPS.pickle', 'wb') as handle:
                    pickle.dump(temp_save, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
            print(
                f'Conformity Scores Computed, took {time.time()-start} secs.')
        return([probas_test_sorted_s, probas_labels_s])

    '''D. Get prediction sets with sliding scores'''

    # The marginal version

    def ERAPS_prediction_sets_marginal(self, probas_test_sorted_s, probas_labels_s, alpha):
        '''
        Outputs:
            prediction_sets: A dictionary of labels: {t: C_t(X_t,U_t,\tau_t,cal)}
        Description:
            1. For each t=T+1,...,T+T1, Get the calibration score \tau_t,cal, which will be used for getting s prediction sets.
            2. Iterate sorted probabilities \pi_t(c), c=1,...,K from the most likely to least likely choices of c
               to get labels in sets
               Outputs are a dictionary of labels: {t: C_t(X_t,U_t,\tau_t,cal)}
            3. After every s predictions, slide scores forward to get new \tau_t,cal.
        '''
        prediction_sets = {}
        T = len(self.X_train)
        T1 = len(self.X_predict)
        tau_t_cal = np.percentile(self.ERAPS_score[:T], 100 * (1 - alpha))
        for t in range(T1):
            prediction_set_t = []
            if np.mod(t, self.s) == 0:
                # Sliding
                tau_t_cal = np.percentile(
                    self.ERAPS_score[t:t + T], 100 * (1 - alpha))
            # Note, these are already sorted so need not have [::-1]
            labels_t = probas_labels_s[t + T, :]
            probas_t = probas_test_sorted_s[t + T, :]
            r_t = 1  # rank of t
            for k in labels_t:
                loc_k = np.where(k == labels_t)[0][0]
                if loc_k == 0:
                    m_i = 0
                else:
                    # cumulative probabilities of labels more likely than Y_i
                    m_i = np.sum(probas_t[:loc_k])
                if m_i + probas_t[loc_k] * self.Unif_RV[t + T] + self.lam * np.max((r_t - self.kreg + 1, 0)) <= tau_t_cal:
                    prediction_set_t.append(k)
                r_t += 1  # increase rank by 1
            # print(f'Prediction set at test time {t+1} is {prediction_set_t}.')
            prediction_sets[t] = prediction_set_t
            # if t <= 5:
            #     # Quick check how the sets look like
            #     print(prediction_set_t)
        # print(
        #     f'Final set of prediction sets computed, took {time.time()-start} secs.')
        return prediction_sets

    # The class-conditional version

    def ERAPS_prediction_sets_conditional(self, probas_test_sorted_s, probas_labels_s, alpha):
        T = len(self.Y_train)
        K = self.K
        scores_train = np.array(self.ERAPS_score[:T])
        scores_predict = np.array(self.ERAPS_score[T:])
        binned_scores_train = {}
        binned_scores_predict = {}
        for k in range(K):
            idx_k = np.where(self.Y_train == k)[0]
            binned_scores_train[k] = scores_train[idx_k]
            idx_k = np.where(self.Y_predict == k)[0]
            binned_scores_predict[k] = scores_predict[idx_k]
        # We expect rare classes or those hard to predict to have VERY LARGE scores
        tau_train = {k: np.percentile(
            binned_scores_train[k], 100 * (1 - alpha)) for k in range(K)}
        print(
            f'The first set of ERAPS class-conditional non-conformity scores are: \n {tau_train}')
        prediction_sets = {}
        T1 = len(self.Y_predict)
        s = self.s
        for t in range(T1):
            prediction_set_t = []
            # This is where we will start slide in the prediction non-conformity score
            start_idx_dict = {k: 0 for k in range(K)}
            if np.mod(t, s) == 0:
                # Sliding, but now we need to know what are label of these s new response
                mul = t // s
                new_classes, new_counts = np.unique(
                    self.Y_predict[(mul - 1) * s:mul * s], return_counts=True)
                for index in range(len(new_classes)):
                    label = new_classes[index]
                    if label not in start_idx_dict.keys():
                        # This happens when doing anomaly detection
                        continue
                    start_idx = start_idx_dict[label]
                    num_occur = new_counts[index]
                    # TODO: May need to deal with missing label in training (e.g., some unseen label emerge in testing)
                    # Yet, I guess this cannot be forseen in reality, so it is less likely an issue
                    # One may always just have a "IDK" class for the classifier
                    binned_scores_train[label] = np.r_[binned_scores_train[label]
                                                       [num_occur:], binned_scores_predict[label][start_idx:start_idx + num_occur]]
                    start_idx_dict[k] += num_occur
                    tau_train[label] = np.percentile(
                        binned_scores_train[label], 100 * (1 - alpha))
            # Note, these are already sorted so need not have [::-1]
            labels_t = probas_labels_s[t + T, :]
            probas_t = probas_test_sorted_s[t + T, :]
            r_t = 1  # rank of t
            for k in labels_t:
                # # Version 1: \hat \tau^k_cal, hoping class-conditional cov. is maintained
                # # Does not work on easy-to-predict cases because \hat \tau^k_cal are too small
                # tau_t_cal = tau_train[k]
                # # Version 2: \hat \tau^max_cal, proof is union bound
                # # Too conservative, almost always producing the whole K
                # tau_t_cal = np.max([i for i in tau_train.values()])
                # Version 3: a combination of 1 and 2, where we chose \hat tau_cal depending on what the label is
                # Use \tau_cal for "easy" labels (e.g., ones with low non-conformity score based on training \tau)
                # Use \tau^max_cal for "hard" labels
                tau_t_cal = self.pi_cond_tau(t, k, tau_train, alpha)
                loc_k = np.where(k == labels_t)[0][0]
                if loc_k == 0:
                    m_i = 0
                else:
                    # cumulative probabilities of labels more likely than Y_i
                    m_i = np.sum(probas_t[:loc_k])
                if m_i + probas_t[loc_k] * self.Unif_RV[t + T] + self.lam * np.max((r_t - self.kreg + 1, 0)) <= tau_t_cal:
                    prediction_set_t.append(k)
                r_t += 1  # increase rank by 1
            # print(f'Prediction set at test time {t+1} is {prediction_set_t}.')
            prediction_sets[t] = prediction_set_t
        print(
            f'The final set of ERAPS class-conditional non-conformity scores (after sliding) are: \n {tau_train}')
        return [prediction_sets, binned_scores_train, binned_scores_predict]

    # A helper for class-conditional version
    def pi_cond_tau(self, t, k, tau_train, alpha, cutoff=2):
        '''
            Input:
                t: current prediction index
                k: the label to be possibly included
                tau_train: {k: \hat \tau^k_cal}
                cutoff: determines what classes are easy to predict
                    The cutoff now is specified for fire incidence report data, so that it is 2 (there are 5 total classes, including 0, the first 3 are easy to predict)
            Return: \hat \tau^k_cal, where
                1. \hat \tau^k_cal=\hat \tau_cal, if i^*_t <= cutoff.
                    Note that we do not use \hat \tau^k_cal because it tends to be too small (not tolerable to dist. shift in data or poor estimation)
                2. \hat \tau^k_cal=\hat \tau^max_cal, if i^*_t > cutoff.
                    This is conservative but should work for rare cases
        '''
        T = len(self.Y_train)
        if k <= cutoff:
            return np.percentile(self.ERAPS_score[t:t + T], 100 * (1 - alpha))
        else:
            return np.max([i for i in tau_train.values()])

    '''2. SRAPS: MJ ICLR'''

    def SRAPS_tau_cal(self, dataset, first_run=True):
        '''
        Description:
            Split data to training I_1 and calibration I_2 and get \tau_i on I_2
        '''
        # 1. Get predictor and predicted probabilities on I_2
        T = len(self.X_train)
        T1 = len(self.X_predict)
        Tcal = int(T * 0.5)  # 50% training data as calibration data
        Tproper = T - Tcal
        idx = np.random.choice(range(T), Tproper, replace=False)
        K = self.K
        all_classes = np.arange(K)
        model = self.regressor
        regr_name = self.regressor.__class__.__name__
        if first_run:
            if regr_name == 'Sequential':
                start1 = time.time()
                model = clone_model(self.regressor)
                model.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
                callback = keras.callbacks.EarlyStopping(
                    monitor='loss', patience=10)
                bsize = int(0.1 * Tproper)
                model.fit(self.X_train[idx, :], self.Y_train[idx],
                          epochs=25, batch_size=bsize, callbacks=[callback], verbose=0)
                print(f'Took {time.time()-start1} secs to fit SRAPS model')
                # size Tcal-by-K, where each row contains predicted probabilities for all classes
                P_Tcal = model.predict(self.X_train[np.delete(range(T), idx)])
                P_T1 = model.predict(self.X_predict)
            else:
                model = model.fit(self.X_train[idx, :],
                                  self.Y_train[idx])
                P_Tcal = utils.predict_proba_ordered(
                    model.predict_proba(self.X_train[np.delete(range(T), idx)]), model.classes_, all_classes)
                P_T1 = utils.predict_proba_ordered(
                    model.predict_proba(self.X_predict), model.classes_, all_classes)
            temp_save = {'1': P_Tcal,
                         '2': P_T1}
            with open(f'{dataset}_{regr_name}_dict_SRAPS.pickle', 'wb') as handle:
                pickle.dump(temp_save, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(f'{dataset}_{regr_name}_dict_SRAPS.pickle', 'rb') as handle:
                temp_save = pickle.load(handle)
            P_Tcal, P_T1 = temp_save.values()
        # 2. Get tau_i on I_2
        if self.regressor.__class__.__name__ == 'Sequential':
            # Need to transform the one-hot-encoded vectors back to integer vectors
            # NOTE: it is possible that "self.Y_train/Y_predict" have columns with 0 sum, because they are imputed due to missing column. Thus, only transform the rest columns
            train_in = np.where(self.Y_train.sum(axis=0) > 0)[0]
            self.Y_train = self.enc.inverse_transform(
                self.Y_train[:, train_in])
        for i in range(Tcal):
            probas_i = P_Tcal[i]  # vector of size K
            Y_i = self.Y_train[np.delete(range(T), idx)][i]
            probas_to_labels = np.argsort(probas_i)[::-1]
            probas_i_sorted = np.sort(probas_i)[::-1]
            # Rank of i-1, as Python starts at index 0
            r_i = np.where(Y_i == probas_to_labels)[0][0]
            # Cumulative probabilities, can be zero
            m_i = np.cumsum(probas_i_sorted[:r_i])
            if r_i == 0:
                m_i = 0
            else:
                # Cumulative probabilities, can be zero
                m_i = np.sum(probas_i_sorted[:r_i])
            true_rank = r_i + 1
            tau_i = m_i + probas_i_sorted[r_i] * self.Unif_RV[i] + \
                self.lam * np.max((true_rank - self.kreg + 1, 0))
            self.SRAPS_score.append(tau_i)
        return P_T1

    def SRAPS_prediction_sets_marginal(self, P_T1, alpha):
        # P_T1 contains predicted probability vector for each test point
        T = len(self.X_train)
        T1 = len(self.X_predict)
        tau_cal = np.percentile(self.SRAPS_score, 100 * (1 - alpha))
        # 3. Get prediction sets:
        prediction_sets = {}
        for t in range(T1):
            prediction_set_t = []
            labels_t = np.argsort(P_T1[t])[::-1]
            probas_t = np.sort(P_T1[t])[::-1]
            r_t = 1  # rank of t
            for k in labels_t:
                loc_k = np.where(k == labels_t)[0][0]
                if loc_k == 0:
                    m_i = 0
                else:
                    # cumulative probabilities of labels more likely than Y_i
                    m_i = np.sum(probas_t[:loc_k])
                if m_i + probas_t[loc_k] * self.Unif_RV[t + T] + self.lam * np.max((r_t - self.kreg + 1, 0)) <= tau_cal:
                    prediction_set_t.append(k)
                r_t += 1  # increase rank by 1
            prediction_sets[t] = prediction_set_t
        # print(
        #     f'Final set of prediction sets computed, took {time.time()-start} secs.')
        return prediction_sets

    def SRAPS_prediction_sets_conditional(self, P_T1, alpha):
        # P_T1 contains predicted probability vector for each test point
        T = len(self.X_train)
        T1 = len(self.X_predict)
        K = self.K
        scores_cal = np.array(self.SRAPS_score)
        Tcal = int(T * 0.5)  # 50% training data as calibration data
        Y_cal = self.Y_train[-Tcal:]
        binned_scores_cal = {}
        for k in range(K):
            idx_k = np.where(Y_cal == k)[0]
            if len(idx_k) == 0:
                binned_scores_cal[k] = [np.percentile(
                    scores_cal, 100 * (1 - alpha))]
            else:
                binned_scores_cal[k] = scores_cal[idx_k]
        tau_cal_dict = {k: np.percentile(
            binned_scores_cal[k], 100 * (1 - alpha)) for k in range(K)}
        print(
            f'The set of SRAPS class-conditional non-conformity scores are: \n {tau_cal_dict}')
        # 3. Get prediction sets:
        prediction_sets = {}
        for t in range(T1):
            prediction_set_t = []
            labels_t = np.argsort(P_T1[t])[::-1]
            probas_t = np.sort(P_T1[t])[::-1]
            r_t = 1  # rank of t
            for k in labels_t:
                tau_cal = tau_cal_dict[k]
                loc_k = np.where(k == labels_t)[0][0]
                if loc_k == 0:
                    m_i = 0
                else:
                    # cumulative probabilities of labels more likely than Y_i
                    m_i = np.sum(probas_t[:loc_k])
                if m_i + probas_t[loc_k] * self.Unif_RV[t + T] + self.lam * np.max((r_t - self.kreg + 1, 0)) <= tau_cal:
                    prediction_set_t.append(k)
                r_t += 1  # increase rank by 1
            prediction_sets[t] = prediction_set_t
        return prediction_sets
    '''3. APS: MJ NeurIPS'''

    def SAPS_tau_cal(self):
        print('Running APS by Candes')
        self.SAPS_score = []
        # 1. Get predictor and predicted probabilities on I_2
        T = len(self.X_train)
        T1 = len(self.X_predict)
        Tcal = int(T * 0.5)  # 50% training data as calibration data
        Tproper = T - Tcal
        idx = np.random.choice(range(T), Tproper, replace=False)
        K = self.K
        all_classes = np.arange(K)
        model = self.regressor
        regr_name = self.regressor.__class__.__name__
        if regr_name == 'Sequential':
            model = clone_model(self.regressor)
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            callback = keras.callbacks.EarlyStopping(
                monitor='loss', patience=10)
            bsize = int(0.1 * Tproper)
            model.fit(self.X_train[idx, :], self.Y_train[idx],
                      epochs=25, batch_size=bsize, callbacks=[callback], verbose=0)
            # size Tcal-by-K, where each row contains predicted probabilities for all classes
            P_Tcal = model.predict(self.X_train[np.delete(range(T), idx)])
            P_T1 = model.predict(self.X_predict)
        else:
            model = model.fit(self.X_train[idx, :],
                              self.Y_train[idx])
            P_Tcal = utils.predict_proba_ordered(
                model.predict_proba(self.X_train[np.delete(range(T), idx)]), model.classes_, all_classes)
            P_T1 = utils.predict_proba_ordered(
                model.predict_proba(self.X_predict), model.classes_, all_classes)
        # 2. Get tau_i on I_2
        if self.regressor.__class__.__name__ == 'Sequential':
            # Need to transform the one-hot-encoded vectors back to integer vectors
            # NOTE: it is possible that "self.Y_train/Y_predict" have columns with 0 sum, because they are imputed due to missing column. Thus, only transform the rest columns
            train_in = np.where(self.Y_train.sum(axis=0) > 0)[0]
            self.Y_train = self.enc.inverse_transform(
                self.Y_train[:, train_in])
        for i in range(Tcal):
            probas_i = P_Tcal[i]  # vector of size K
            Y_i = self.Y_train[np.delete(range(T), idx)][i]
            probas_to_labels = np.argsort(probas_i)[::-1]
            probas_i_sorted = np.sort(probas_i)[::-1]
            # Rank of i-1, as Python starts at index 0
            r_i = np.where(Y_i == probas_to_labels)[0][0]
            # Cumulative probabilities, can be zero
            m_i = np.cumsum(probas_i_sorted[:r_i])
            if r_i == 0:
                m_i = 0
            else:
                # Cumulative probabilities, can be zero
                m_i = np.sum(probas_i_sorted[:r_i])
            tau_i = m_i + probas_i_sorted[r_i] * (1-self.Unif_RV[i])
            self.SAPS_score.append(tau_i)
        return P_T1

    def SAPS_prediction_sets_marginal(self, P_T1, alpha):
        T = len(self.X_train)
        T1 = len(self.X_predict)
        tau_cal = np.percentile(self.SAPS_score, 100 * (1 - alpha))
        # 3. Get prediction sets:
        prediction_sets = {}
        for t in range(T1):
            labels_t = np.argsort(P_T1[t])[::-1]
            probas_t = np.sort(P_T1[t])[::-1]
            cum_prob_t = np.cumsum(probas_t)
            L_t = np.where(cum_prob_t >= tau_cal)[0][0]
            r_t = L_t+1  # Otherwise the sume does not include L_t
            V_t = (np.sum(probas_t[:r_t])-tau_cal)/probas_t[L_t]
            if self.Unif_RV[t + T] <= V_t:
                prediction_sets[t] = labels_t[:np.max([0, r_t-1])]
            else:
                prediction_sets[t] = labels_t[:r_t]
        return prediction_sets

    def naive_train_mod(self):
        # Include all labels where the cumulative probability exceeds 1-\alpha
        # Does not work well when the predictor is poor
        # 1. Train model on all data
        K = self.K
        all_classes = np.arange(K)
        model = self.regressor
        regr_name = self.regressor.__class__.__name__
        if regr_name == 'Sequential':
            model = clone_model(self.regressor)
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            callback = keras.callbacks.EarlyStopping(
                monitor='loss', patience=10)
            bsize = int(0.1 * self.X_train.shape[0])
            model.fit(self.X_train, self.Y_train,
                      epochs=25, batch_size=bsize, callbacks=[callback], verbose=0)
            P_T1 = model.predict(self.X_predict)
        else:
            model = model.fit(self.X_train,
                              self.Y_train)
            P_T1 = utils.predict_proba_ordered(
                model.predict_proba(self.X_predict), model.classes_, all_classes)
        return P_T1

    def naive_prediction_sets_marginal(self, P_T1, alpha):
        # 2. Get prediction sets
        T = len(self.X_train)
        T1 = len(self.X_predict)
        prediction_sets = {}
        tau_cal = 1-alpha
        for t in range(T1):
            labels_t = np.argsort(P_T1[t])[::-1]
            probas_t = np.sort(P_T1[t])[::-1]
            cum_prob_t = np.cumsum(probas_t)
            L_t = np.where(cum_prob_t >= tau_cal)[0][0]
            r_t = L_t+1  # Otherwise the sume does not include L_t
            V_t = (np.sum(probas_t[:r_t])-tau_cal)/probas_t[L_t]
            if self.Unif_RV[t + T] <= V_t:
                prediction_sets[t] = labels_t[:np.max([0, r_t-1])]
            else:
                prediction_sets[t] = labels_t[:r_t]
        return prediction_sets


def SRAPS_test(calibration_prob, Y_calibration, predicted_prob, Y_predict, penalties):
    # NOTE, this is the marginal version
    # 1. Training model and getting predictions are done already, so they are inputs and just need minor adjustments
    lam, kreg = penalties
    Tcal = calibration_prob.shape[0]
    T1 = predicted_prob.shape[0]
    K = calibration_prob.shape[1]
    all_classes = np.arange(K)
    # The two lines below are not necessary
    P_Tcal = utils.predict_proba_ordered(
        calibration_prob, all_classes, all_classes)
    P_T1 = utils.predict_proba_ordered(
        predicted_prob, all_classes, all_classes)
    # 2. Get tau_i on I_2
    start = time.time()
    Unif_RV = np.random.uniform(size=Tcal + T1)
    alpha = 0.1
    SRAPS_score = []
    for i in range(Tcal):
        probas_i = P_Tcal[i]  # vector of size K
        Y_i = Y_calibration[i]
        probas_to_labels = np.argsort(probas_i)[::-1]
        probas_i_sorted = np.sort(probas_i)[::-1]
        # Rank of i-1, as Python starts at index 0
        r_i = np.where(Y_i == probas_to_labels)[0][0]
        if r_i == 0:
            m_i = 0
        else:
            # Cumulative probabilities, can be zero
            m_i = np.sum(probas_i_sorted[:r_i])
        true_rank = r_i + 1
        tau_i = m_i + probas_i_sorted[r_i] * Unif_RV[i] + \
            lam * np.max((true_rank - kreg + 1, 0))
        SRAPS_score.append(tau_i)
    tau_cal = np.percentile(SRAPS_score, 100 * (1 - alpha))
    # lam = tau_cal
    print(f'Calibration tau is {np.round(tau_cal,2)}')
    # 3. Get prediction sets:
    prediction_sets = {}
    for t in range(T1):
        prediction_set_t = []
        labels_t = np.argsort(P_T1[t])[::-1]
        probas_t = np.sort(P_T1[t])[::-1]
        r_t = 1  # rank of t
        for k in labels_t:
            loc_k = np.where(k == labels_t)[0][0]  # It is just k
            if loc_k == 0:
                m_i = 0
            else:
                m_i = np.sum(probas_t[:loc_k])
            if m_i + probas_t[loc_k] * Unif_RV[t + Tcal] + lam * np.max((r_t - kreg + 1, 0)) <= tau_cal:
                prediction_set_t.append(k)
            r_t += 1  # increase rank by 1
        prediction_sets[t] = prediction_set_t
    # print(
    #     f'Final set of prediction sets computed, took {time.time()-start} secs.')
    return prediction_sets
