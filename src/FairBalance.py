import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from collections import Counter
from pdb import set_trace


class FairBalance:
    def __init__(self, model, data="compas", fair_balance = True):
        models = {"SVM": SVC(kernel="linear", probability=True, class_weight="balanced"),
                  "RF": RandomForestClassifier(n_estimators=100, criterion="entropy", class_weight="balanced"),
                  "LR": LogisticRegression(class_weight="balanced"),
                  "NB": GaussianNB(),
                  "DT": DecisionTreeClassifier(criterion="entropy", class_weight="balanced")
                  }
        self.model = models[model]
        self.fair_balance = fair_balance
        self.load_data(data)

    def load_data(self, data="compas"):
        if data == "compas":
            ## Load dataset
            dataset_orig = pd.read_csv('../dataset/compas-scores-two-years.csv')

            ## Drop categorical features
            ## Removed two duplicate coumns - 'decile_score','priors_count'
            dataset_orig = dataset_orig.drop(
                ['id', 'name', 'first', 'last', 'compas_screening_date', 'dob', 'age_cat', 'juv_fel_count', 'decile_score',
                 'juv_misd_count', 'juv_other_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
                 'c_case_number', 'c_offense_date', 'c_arrest_date', 'c_days_from_compas', 'c_charge_desc', 'is_recid',
                 'r_case_number', 'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc',
                 'r_jail_in', 'r_jail_out', 'violent_recid', 'is_violent_recid', 'vr_case_number', 'vr_charge_degree',
                 'vr_offense_date', 'vr_charge_desc', 'type_of_assessment', 'decile_score', 'score_text',
                 'screening_date', 'v_type_of_assessment', 'v_decile_score', 'v_score_text', 'v_screening_date',
                 'in_custody', 'out_custody', 'start', 'end', 'event'], axis=1)

            ## Drop NULL values
            dataset_orig = dataset_orig.dropna()

            ## Change symbolics to numerics
            dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'Female', 1, 0)
            dataset_orig['race'] = np.where(dataset_orig['race'] == 'Caucasian', 1, 0)
            dataset_orig['c_charge_degree'] = np.where(dataset_orig['c_charge_degree'] == 'F', 1, 0)
            ## Rename class column
            dataset_orig.rename(index=str, columns={"two_year_recid": "label"}, inplace=True)
            dataset_orig = dataset_orig.sample(frac=1).reset_index(drop=True)
            self.y = np.array(dataset_orig["label"])
            self.X = dataset_orig.drop(["label"], axis=1)
            self.privilege = ["sex", "race"]

    def fit(self, X, y):
        if self.fair_balance:
            segments = {}
            n = len(y)
            for id in X.index:
                cub = [y[id]]+[X[key][id] for key in self.privilege]
                cub = tuple(cub)
                if cub not in segments:
                    segments[cub] = []
                segments[cub].append(id)
            if len(segments) < 2**(len(self.privilege)+1):
                raise Exception("Sorry, cannot balance training data.")
            else:
                sampled = []
                for seg in segments:
                    sampled += list(np.random.choice(segments[seg], n))
                XX = X.iloc[sampled]
                yy = y[sampled]
                self.model.fit(XX, yy)
        else:
            self.model.fit(X, y)


    def evaluate(self, preds):
        def rate(a, b):
            aa = Counter(a)[True]
            bb = Counter(b)[True]
            if aa+bb == 0:
                return 0
            else:
                return aa / float(aa+bb)

        result = {}
        pp = preds == 1
        np = preds == 0
        pg = self.y == 1
        ng = self.y == 0
        tp = pp & pg
        fp = pp & ng
        tn = np & ng
        fn = np & pg
        result["tpr"] = rate(tp, fn)
        result["fpr"] = rate(fp, tn)
        for key in self.privilege:
            result[key] = {}
            group1 = self.X[key] == 1
            group0 = self.X[key] == 0
            tp1 = tp & group1
            fp1 = fp & group1
            tn1 = tn & group1
            fn1 = fn & group1
            tp0 = tp & group0
            fp0 = fp & group0
            tn0 = tn & group0
            fn0 = fn & group0
            tpr1 = rate(tp1, fn1)
            fpr1 = rate(fp1, tn1)
            tpr0 = rate(tp0, fn0)
            fpr0 = rate(fp0, tn0)
            result[key]["eod"] = abs(tpr0 - tpr1)
            result[key]["aod"] = abs(0.5*(fpr0-fpr1+tpr0-tpr1))
        return result


    def cross_val(self, folds = 5):
        kf = KFold(n_splits = folds)
        cross_result = {"TP":0, "FP": 0, "TN": 0, "FN": 0}
        preds = []
        for train, test in kf.split(self.y):
            X_train = self.X.iloc[train]
            y_train = self.y[train]
            X_test = self.X.iloc[test]
            ss = StandardScaler().fit(X_train)
            X_train = pd.DataFrame(ss.transform(X_train), columns = self.X.columns)
            X_test = pd.DataFrame(ss.transform(X_test), columns = self.X.columns)
            self.fit(X_train, y_train)
            preds.extend(list(self.model.predict(X_test)))
        result = self.evaluate(np.array(preds))
        print(result)


