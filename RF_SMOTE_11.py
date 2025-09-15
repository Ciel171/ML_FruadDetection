import warnings

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.weightstats import ttest_ind
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
from os.path import isfile

from extra_codes import calc_vif
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import kpss

from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline





class ML_Fraud:
    __version__ = '1.0.5'

    def __init__(self, sample_start=1991, test_sample=range(2001, 2011),
                 OOS_per=1, OOS_gap=0, sampling='expanding', adjust_serial=True,
                 cv_type='kfold', temp_year=1, cv_flag=False, cv_k=10, write=True, IS_per=10):

        df = pd.read_csv('FraudDB2020.csv')
        self.df = df
        self.ss = sample_start
        self.se = np.max(df.fyear)
        self.ts = test_sample
        self.cv_t = cv_type
        self.cv = cv_flag
        self.cv_k = cv_k
        self.cv_t_y = temp_year

        sampling_set = ['expanding', 'rolling']
        if sampling in sampling_set:
            pass
        else:
            raise ValueError('Invalid sampling choice. Permitted options are "expanding" and "rolling"')

        self.sa = sampling
        self.w = write
        self.ip = IS_per
        self.op = OOS_per
        self.og = OOS_gap
        self.a_s = adjust_serial
        print('Module initiated successfully ...')
        # The dir() function returns all properties and methods of the specified object, without the values.
        list_methods = dir(self)
        # .any: It checks for any element satisfying a condition and returns a True in case it finds any one element.
        reduced_methods = [item + '()' for item in list_methods if
                           any(['analy' in item, 'compare' in item, item == 'sumstats'])]
        # string.join(iterable)
        print('Procedures are: ' + '; '.join(reduced_methods))

    def Four_Plus_FUSED(self, C_FN=30, C_FP=1):
        """
        This code uses 11 financial ratios to predict the likelihood of fraud in a financial statement.

        Parameters:
            – C_FN: Cost of a False Negative for ECM
            – C_FP: Cost of a False Positive for ECM

        Predictive models:
            – Support Vector Machine (SVM)
            – Logistic Regression (LR)
            – SGD Tree Boosting (SGD)
            – Adaptive Boosting with Logistic Regression/LogitBoost (ADA)
            – MUlti-layered Perceptron (MLP)
            – FUSED (weighted average of estimated probs of other methods)

        Outputs:
        Main results are stored in the table variable "perf_tbl_general" written into
        2 csv files: time period 2001-2010 and 2003-2008.

        Steps:
            1. Cross-validate to find optimal hyperparameters.
            2. Estimating the performance for each OOS period.

        Warnings:
            – Running this code can take up to 85 mins. The cross-validation takes up
            to 60 mins (you can skip this step) main analysis up to 15 mins.
            These figures are estimates based on a MacBook Pro 2021.

        """

        from sklearn.linear_model import LogisticRegression
        from sklearn.linear_model import SGDClassifier
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.metrics import roc_auc_score
        from sklearn.tree import DecisionTreeClassifier
        from extra_codes import ndcg_k, relogit
        from statsmodels.discrete.discrete_model import Logit
        from statsmodels.tools import add_constant

        t0 = datetime.now()
        # setting the parameters
        IS_period = self.ip
        k_fold = self.cv_k
        OOS_period = self.op  # 1 year ahead prediction
        OOS_gap = self.og  # Gap between training and testing period
        start_OOS_year = self.ts[0]  # 2001
        end_OOS_year = self.ts[-1]  # 2010
        sample_start = self.ss  # 1991
        adjust_serial = self.a_s
        cv_type = self.cv_t
        cross_val = self.cv
        temp_year = self.cv_t_y  # 1
        case_window = self.sa
        fraud_df = self.df.copy(deep=True)
        write = self.w

        reduced_tbl_1 = fraud_df.iloc[:, [0, 1, 3, 7, 8]]
        reduced_tbl_2 = fraud_df.iloc[:, -14:-3]
        reduced_tblset = [reduced_tbl_1, reduced_tbl_2]
        reduced_tbl = pd.concat(reduced_tblset, axis=1)
        reduced_tbl = reduced_tbl[reduced_tbl.fyear >= sample_start]  # 1991
        reduced_tbl = reduced_tbl[reduced_tbl.fyear <= end_OOS_year]  # 2010

        # Setting the cross-validation setting
        # IC sample: fyear 1991-2000
        tbl_year_IS_CV = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < start_OOS_year, \
                                                        reduced_tbl.fyear >= start_OOS_year - IS_period)]
        tbl_year_IS_CV = tbl_year_IS_CV.reset_index(drop=True)
        misstate_firms = np.unique(tbl_year_IS_CV.gvkey[tbl_year_IS_CV.AAER_DUMMY == 1])

        X_CV = tbl_year_IS_CV.iloc[:, -11:]
        Y_CV = tbl_year_IS_CV.AAER_DUMMY

        P_f = np.sum(Y_CV == 1) / len(Y_CV)
        P_nf = 1 - P_f

        print('prior probablity of fraud between ' + str(sample_start) + '-' +
              str(start_OOS_year - 1) + ' is ' + str(np.round(P_f * 100, 2)) + '%')

        # redo cross-validation if you wish
        if cv_type == 'kfold':
            if cross_val == True:


                #optimize random forest

                pipe_rdf = Pipeline([('scale', StandardScaler()), \
                                     ('smote', SMOTE(random_state=0)),
                                     ('base_mdl_rdf', RandomForestClassifier(max_leaf_nodes=2, max_features=11, bootstrap=True,\
                                                                             random_state=0, n_jobs=-1))])


                estimators = list(range(10,301,10))
                #max_depth = list(range(0,101,10))
                #class_weight = [{0: 1 / x, 1: 1} for x in range(10, 501, 10)]

                param_grid_rdf = {
                    'base_mdl_rdf__n_estimators': estimators,  # Number of trees
                    'base_mdl_rdf__max_depth': [None, 2, 5, 10, 15, 20, 50, 100],  # Maximum depth of trees
                    'base_mdl_rdf__min_samples_split': [2, 5, 10, 20, 50, 100],  # Minimum samples required to split a node
                    'base_mdl_rdf__criterion': ['gini', 'entropy'],  # Splitting criterion
                    'base_mdl_rdf__class_weight': [None, 'balanced'],
                }

                clf_rdf = GridSearchCV(pipe_rdf, param_grid_rdf, scoring='roc_auc', \
                                       n_jobs=-1, cv=k_fold, refit=False)

                clf_rdf.fit(X_CV, Y_CV)
                opt_params_rdf = clf_rdf.best_params_
                estimators_rdf = opt_params_rdf['base_mdl_rdf__n_estimators']
                max_depth_rdf = opt_params_rdf['base_mdl_rdf__max_depth']
                min_samples_split_opt = opt_params_rdf['base_mdl_rdf__min_samples_split']
                criterion_opt = opt_params_rdf['base_mdl_rdf__criterion']
                cw_opt = opt_params_rdf['base_mdl_rdf__class_weight']
                score_rdf = clf_rdf.best_score_


                print('Random Forest: The optimal number of estimators is ' + \
                      str(estimators_rdf) + ', max depth is' + str(max_depth_rdf) + ', min samples split is' + \
                      str(min_samples_split_opt) + \
                      ', criterion is' + str(criterion_opt) + \
                      ', class weight is' + str(cw_opt) + ', score is' + str(score_rdf))

        range_oos = range(start_OOS_year, end_OOS_year + 1, OOS_period)  # (2001,2010+1,1)


        roc_rdf = np.zeros(len(range_oos))
        roc_rdf_training = np.zeros(len(range_oos))
        sensitivity_OOS_rdf1 = np.zeros(len(range_oos))
        sensitivity_rdf1_training = np.zeros(len(range_oos))
        specificity_OOS_rdf1 = np.zeros(len(range_oos))
        specificity_rdf1_training = np.zeros(len(range_oos))
        precision_rdf1 = np.zeros(len(range_oos))
        precision_rdf1_training = np.zeros(len(range_oos))
        ndcg_rdf1 = np.zeros(len(range_oos))
        ndcg_rdf1_training = np.zeros(len(range_oos))
        ecm_rdf1 = np.zeros(len(range_oos))
        ecm_rdf1_training = np.zeros(len(range_oos))
        TP_rdf1 = np.zeros(len(range_oos))
        TN_rdf1 = np.zeros(len(range_oos))
        FP_rdf1 = np.zeros(len(range_oos))
        FN_rdf1 = np.zeros(len(range_oos))
        accuracy_training_rdf = np.zeros(len(range_oos))




        m = 0
        for yr in range_oos:  # 2001-2010
            t1 = datetime.now()
            if case_window == 'expanding':
                year_start_IS = sample_start  # 1991
            else:
                year_start_IS = yr - IS_period  # 1991
            # how many years between training and testing sample:
            # expanding: 1991-2000, 1991-2001
            # rolling: 1991-2000, 1992-2001
            tbl_year_IS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear < yr - OOS_gap, \
                                                         reduced_tbl.fyear >= year_start_IS)]
            tbl_year_IS = tbl_year_IS.reset_index(drop=True)

            misstate_firms = np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY == 1])
            # How many periods constitute the testing sample at a time: 2001, 2002
            tbl_year_OOS = reduced_tbl.loc[np.logical_and(reduced_tbl.fyear >= yr, \
                                                          reduced_tbl.fyear < yr + OOS_period)]

            print(f'before dropping the number of observations is: {len(tbl_year_OOS)}')

            if adjust_serial == True:
                ok_index = np.zeros(tbl_year_OOS.shape[0])
                for s in range(0, tbl_year_OOS.shape[0]):
                    if not tbl_year_OOS.iloc[s, 1] in misstate_firms:
                        ok_index[s] = True


            else:
                # filled with ones and keep all observations including serial frauds
                ok_index = np.ones(tbl_year_OOS.shape[0]).astype(bool)

            # deleting observations where a company appears both in IS and OOS samples
            tbl_year_OOS = tbl_year_OOS.iloc[ok_index == True, :]
            tbl_year_OOS = tbl_year_OOS.reset_index(drop=True)
            print(f'after dropping the number of observations is: {len(tbl_year_OOS)}')

            X = tbl_year_IS.iloc[:, -11:]
            mean = np.mean(X)
            std = np.std(X)
            X = (X - mean) / std
            Y = tbl_year_IS.AAER_DUMMY

            smote = SMOTE(random_state=0)
            X, Y = smote.fit_resample(X, Y)

            X_OOS = tbl_year_OOS.iloc[:, -11:]
            X_OOS = (X_OOS - mean) / std
            Y_OOS = tbl_year_OOS.AAER_DUMMY

            n_P = np.sum(Y_OOS == 1)
            n_N = np.sum(Y_OOS == 0)

            n_P_training = np.sum(Y == 1)
            n_N_training = np.sum(Y == 0)




            clf_rdf = RandomForestClassifier(
                                n_estimators=estimators_rdf,
                                criterion=criterion_opt,
                                max_depth=max_depth_rdf,
                                min_samples_split=min_samples_split_opt,
                                max_features=11,
                                bootstrap=True,
                                max_leaf_nodes=2,
                                class_weight=cw_opt,
                                random_state=0,
                                n_jobs=-1)

            clf_rdf = clf_rdf.fit(X, Y)

            #test accuracy on training sample- rdf
            probs_fraud_rdf = clf_rdf.predict_proba(X)[:, 1]
            cutoff_rdf = np.percentile(probs_fraud_rdf, 99)
            binary_predictions_rdf = (probs_fraud_rdf >= cutoff_rdf).astype(int)
            accuracy_training_rdf[m] = accuracy_score(Y, binary_predictions_rdf)
            roc_rdf_training[m] = roc_auc_score(Y, probs_fraud_rdf)
            sensitivity_rdf1_training[m] = np.sum(np.logical_and(probs_fraud_rdf >= cutoff_rdf, \
                                                            Y == 1)) / np.sum(Y)
            specificity_rdf1_training[m] = np.sum(np.logical_and(probs_fraud_rdf < cutoff_rdf, \
                                                            Y == 0)) / np.sum(Y == 0)
            precision_rdf1_training[m] = np.sum(np.logical_and(probs_fraud_rdf >= cutoff_rdf, \
                                                      Y == 1)) / np.sum(probs_fraud_rdf >= cutoff_rdf)
            ndcg_rdf1_training[m] = ndcg_k(Y, probs_fraud_rdf, 99)


            FN_rdf3 = np.sum(np.logical_and(probs_fraud_rdf < cutoff_rdf, \
                                            Y == 1))
            FP_rdf3 = np.sum(np.logical_and(probs_fraud_rdf >= cutoff_rdf, \
                                            Y == 0))

            ecm_rdf1_training[m] = C_FN * P_f * FN_rdf3 / n_P_training + C_FP * P_nf * FP_rdf3 / n_N_training





            probs_oos_fraud_rdf = clf_rdf.predict_proba(X_OOS)[:, 1]
            roc_rdf[m] = roc_auc_score(Y_OOS, probs_oos_fraud_rdf)

            cutoff_OOS_rdf = np.percentile(probs_oos_fraud_rdf, 99)
            TP_rdf1[m] = np.sum(np.logical_and(probs_oos_fraud_rdf >= cutoff_OOS_rdf, \
                                  Y_OOS == 1))
            print(str(m) + 'TP for rdf is:' + str(TP_rdf1[m]))
            TN_rdf1[m] = np.sum(np.logical_and(probs_oos_fraud_rdf < cutoff_OOS_rdf, \
                                  Y_OOS == 0))
            print(str(m) + 'TN for rdf is:' + str(TN_rdf1[m]))
            sensitivity_OOS_rdf1[m] = np.sum(np.logical_and(probs_oos_fraud_rdf >= cutoff_OOS_rdf, \
                                                            Y_OOS == 1)) / np.sum(Y_OOS)
            specificity_OOS_rdf1[m] = np.sum(np.logical_and(probs_oos_fraud_rdf < cutoff_OOS_rdf, \
                                                            Y_OOS == 0)) / np.sum(Y_OOS == 0)
            precision_rdf1[m] = np.sum(np.logical_and(probs_oos_fraud_rdf >= cutoff_OOS_rdf, \
                                                      Y_OOS == 1)) / np.sum(probs_oos_fraud_rdf >= cutoff_OOS_rdf)
            ndcg_rdf1[m] = ndcg_k(Y_OOS, probs_oos_fraud_rdf, 99)

            FN_rdf1[m] = np.sum(np.logical_and(probs_oos_fraud_rdf < cutoff_OOS_rdf, \
                                            Y_OOS == 1))
            print(str(m) + 'FN for rdf is:' + str(FN_rdf1[m]))
            FP_rdf1[m] = np.sum(np.logical_and(probs_oos_fraud_rdf >= cutoff_OOS_rdf, \
                                            Y_OOS == 0))
            print(str(m) + 'FP for rdf is:' + str(FP_rdf1[m]))

            FN_rdf2 = np.sum(np.logical_and(probs_oos_fraud_rdf < cutoff_OOS_rdf, \
                                            Y_OOS == 1))
            FP_rdf2 = np.sum(np.logical_and(probs_oos_fraud_rdf >= cutoff_OOS_rdf, \
                                            Y_OOS == 0))

            ecm_rdf1[m] = C_FN * P_f * FN_rdf2 / n_P + C_FP * P_nf * FP_rdf2 / n_N




        f1_score_rdf1_training = 2 * (precision_rdf1_training * sensitivity_rdf1_training) / \
                        (precision_rdf1_training + sensitivity_rdf1_training + 1e-8)



        f1_score_rdf1 = 2 * (precision_rdf1 * sensitivity_OOS_rdf1) / \
                        (precision_rdf1 + sensitivity_OOS_rdf1 + 1e-8)



        # create performance table now
        perf_tbl_general = pd.DataFrame()
        perf_tbl_general['models'] = ['rdf']

        perf_tbl_general['TN'] = [str(np.round(
            np.mean(TN_rdf1), 2))]

        perf_tbl_general['TP'] = [str(np.round(
            np.mean(TP_rdf1), 2))]

        perf_tbl_general['FP'] = [str(np.round(
            np.mean(FP_rdf1), 2))]

        perf_tbl_general['FN'] = [str(np.round(
            np.mean(FN_rdf1), 2))]


        perf_tbl_general['Training Roc'] = [str(np.round(
            np.mean(roc_rdf_training) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_rdf_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing Roc'] = [str(np.round(
            np.mean(roc_rdf) * 100, 2)) + '% (' + \
                                   str(np.round(np.std(roc_rdf) * 100, 2)) + '%)']

        gap_roc_rdf = roc_rdf - roc_rdf_training


        mean_gap_roc_rdf = np.round(np.mean(gap_roc_rdf) * 100, 2)


        perf_tbl_general['Gap Roc'] = [str(mean_gap_roc_rdf) + '%']


        perf_tbl_general['Training Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_rdf1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_rdf1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing Sensitivity @ 1 Prc'] = [str(np.round(
            np.mean(sensitivity_OOS_rdf1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(sensitivity_OOS_rdf1) * 100, 2)) + '%)']

        gap_sensitivity_rdf = sensitivity_OOS_rdf1 - sensitivity_rdf1_training


        mean_gap_sensitivity_rdf = np.round(np.mean(gap_sensitivity_rdf) * 100, 2)


        perf_tbl_general['Gap Sensitivity'] = [str(mean_gap_sensitivity_rdf) + '%']


        perf_tbl_general['Training Specificity @ 1 Prc'] = [str(np.round(
            np.mean(specificity_rdf1_training) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_rdf1_training) * 100, 2)) + '%)']


        perf_tbl_general['Testing Specificity @ 1 Prc'] = [str(np.round(
            np.mean(specificity_OOS_rdf1) * 100, 2)) + '% (' + \
                                                   str(np.round(np.std(specificity_OOS_rdf1) * 100, 2)) + '%)']


        gap_specificity_rdf = specificity_OOS_rdf1 - specificity_rdf1_training


        mean_gap_specificity_rdf = np.round(np.mean(gap_specificity_rdf) * 100, 2)


        perf_tbl_general['Gap Specificity'] = [str(mean_gap_specificity_rdf) + '%']


        perf_tbl_general['Training Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_rdf1_training) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_rdf1_training) * 100, 2)) + '%)']




        perf_tbl_general['Testing Precision @ 1 Prc'] = [str(np.round(
            np.mean(precision_rdf1) * 100, 2)) + '% (' + \
                                                 str(np.round(np.std(precision_rdf1) * 100, 2)) + '%)']

        gap_precision_rdf = precision_rdf1 - precision_rdf1_training


        mean_gap_precision_rdf = np.round(np.mean(gap_precision_rdf) * 100, 2)


        perf_tbl_general['Gap Precision'] = [str(mean_gap_precision_rdf) + '%']


        perf_tbl_general['Training F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_score_rdf1_training) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_rdf1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing F1 Score @ 1 Prc'] = [str(np.round(
            np.mean(f1_score_rdf1) * 100, 2)) + '% (' + \
                                                str(np.round(np.std(f1_score_rdf1) * 100, 2)) + '%)']


        gap_f1_score_rdf = f1_score_rdf1 - f1_score_rdf1_training


        mean_gap_f1_score_rdf = np.round(np.mean(gap_f1_score_rdf) * 100, 2)


        perf_tbl_general['Gap F1 Score'] = [str(mean_gap_f1_score_rdf) + '%']


        perf_tbl_general['Training NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_rdf1_training) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_rdf1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing NDCG @ 1 Prc'] = [str(np.round(
            np.mean(ndcg_rdf1) * 100, 2)) + '% (' + \
                                            str(np.round(np.std(ndcg_rdf1) * 100, 2)) + '%)']

        gap_ndcg_rdf = ndcg_rdf1 - ndcg_rdf1_training

        mean_gap_ndcg_rdf = np.round(np.mean(gap_ndcg_rdf) * 100, 2)


        perf_tbl_general['Gap NDCG'] = [str(mean_gap_ndcg_rdf) + '%']


        perf_tbl_general['Training ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_rdf1_training) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_rdf1_training) * 100, 2)) + '%)']

        perf_tbl_general['Testing ECM @ 1 Prc'] = [str(np.round(
            np.mean(ecm_rdf1) * 100, 2)) + '% (' + \
                                           str(np.round(np.std(ecm_rdf1) * 100, 2)) + '%)']

        gap_ecm_rdf = ecm_rdf1 - ecm_rdf1_training

        mean_gap_ecm_rdf = np.round(np.mean(gap_ecm_rdf) * 100, 2)


        perf_tbl_general['Gap ECM'] = [str(mean_gap_ecm_rdf) + '%']


        lbl_perf_tbl = 'perf_tbl_' + str(start_OOS_year) + '_' + str(end_OOS_year) + \
                       '_' + case_window + ',OOS=' + str(OOS_period) + ',' + \
                       str(k_fold) + 'fold' + ',serial=' + str(adjust_serial) + \
                       ',gap=' + str(OOS_gap) + '_rdf_kfold_new_standardisation_v1.csv'

        if write == True:
            perf_tbl_general.to_csv(lbl_perf_tbl, index=False)
        print(perf_tbl_general)
        t_last = datetime.now()
        dt_total = t_last - t0
        print('total run time is ' + str(dt_total.total_seconds()) + ' sec')


a = ML_Fraud(sample_start = 1991,test_sample = range (2001,2011),OOS_per = 1,OOS_gap = 0,sampling = "expanding",adjust_serial = True,
            cv_flag = True,cv_k = 10,write = True,IS_per = 10)
a.Four_Plus_FUSED()