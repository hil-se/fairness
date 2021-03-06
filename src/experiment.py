import numpy
import sys
sys.path.append("../")
import numpy as np
import pandas as pd
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from collections import Counter
from fairbalance import FairBalance
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

class Experiment:
    def __init__(self, model, data="german", fair_balance = "none", target_attribute=""):
        models = {"SVM": LinearSVC(dual=False),
                  "RF": RandomForestClassifier(n_estimators=100, criterion="entropy"),
                  "LR": LogisticRegression(),
                  "DT": DecisionTreeClassifier(criterion="entropy"),
                  "NB": GaussianNB()
                  }
        data_loader = {"compas": load_preproc_data_compas, "adult": load_preproc_data_adult, "german": load_preproc_data_german}

        self.model = models[model]
        self.fair_balance = fair_balance

        # No effect on FairBalance
        self.target_attribute = target_attribute

        self.data = data_loader[data]()
        self.X = self.data.features
        self.y = self.data.labels.ravel()
        self.inject_place = "None"
        self.inject_ratio = None
        self.inject_amount = None

    def inject_bias(self, inject_place, inject_ratio):
        # inject_place = {"All": inject to both training and test data, "Train": only inject bias in training data, "None": no biased labels}
        # inject_ratio={attribute1: [ratio11, ratio12], attribute2: [ratio21, ratio22], ...}
        # ratio11 = 0.2 , ratio12 = -0.1 then
        #   1. 20% of (attribute1 =0 AND label = 0) will be changed to label = 1,
        #   2. 10% of (attribute1 =1 AND label = 1) will be changed to label = 0.
        # ratio21 = -0.1 , ratio22 = 0.2 then
        #   1. 10% of (attribute2 =0 AND label = 1) will be changed to label = 0,
        #   2. 20% of (attribute1 =1 AND label = 0) will be changed to label = 1.
        self.inject_place = inject_place
        self.inject_ratio = inject_ratio

    def inject_bias_amount(self, inject_place, inject_amount):
        # inject_place = {"All": inject to both training and test data, "Train": only inject bias in training data, "None": no biased labels}
        # inject_ratio={attribute1: amount1, attribute2: amount2, ...}
        # amount1 = 20 , amount2 = -10 then
        #   1. 20 of (attribute1 =0 AND label = 0) will be changed to label = 1,
        #       and 20 of (attribute1 =1 AND label = 1) will be changed to label = 0
        #   2. 10 of (attribute2 =1 AND label = 0) will be changed to label = 1,
        #       and 10 of (attribute2 =0 AND label = 1) will be changed to label = 0.

        self.inject_place = inject_place
        self.inject_amount = inject_amount

    def inject(self, data):
        # perform bias injection on the input data.
        if self.inject_ratio:
            for attribute in self.inject_ratio:
                y = data.labels.ravel()
                target = max(y)
                non_target = min(y)
                try:
                    ind = data.protected_attribute_names.index(attribute)
                except:
                    print("Error: Attribute %s does not exist in the protected attributes." %attribute)
                    sys.exit(1)
                groups = data.protected_attributes[:,ind]
                for group, ratio in enumerate(self.inject_ratio[attribute]):
                    to_change = non_target if ratio>0 else target
                    change_to = target if ratio>0 else non_target
                    change = numpy.where((groups == group) & (y==to_change))[0]
                    size = int(numpy.abs(ratio)*len(change))
                    selected = numpy.random.choice(change, size, replace=False)
                    for i in selected:
                        data.labels[i][0] = change_to
        elif self.inject_amount:
            for attribute in self.inject_amount:
                y = data.labels.ravel()
                target = max(y)
                non_target = min(y)
                try:
                    ind = data.protected_attribute_names.index(attribute)
                except:
                    print("Error: Attribute %s does not exist in the protected attributes." %attribute)
                    sys.exit(1)
                groups = data.protected_attributes[:,ind]
                amount = self.inject_amount[attribute]
                for group in range(2):
                    to_change = non_target if group != (amount > 0) else target
                    change_to = target if group != (amount > 0) else non_target
                    change = numpy.where((groups == group) & (y == to_change))[0]
                    selected = numpy.random.choice(change, abs(amount), replace=False)
                    for i in selected:
                        data.labels[i][0] = change_to
        return data


    def data_prepare(self):
        data_train, data_test = self.data.split([0.7], shuffle=True)
        if self.inject_place!="None":
            data_train = self.inject(data_train)
        if self.inject_place == "All":
            data_test = self.inject(data_test)
        return data_train, data_test

    def run(self):
        data_train, data_test = self.data_prepare()

        privileged_groups = [{self.target_attribute: 1}]
        unprivileged_groups = [{self.target_attribute: 0}]
        if self.fair_balance=="FairBalance":
            dataset_transf_train = FairBalance(data_train, class_balance=False)
        elif self.fair_balance=="FairBalanceClass":
            dataset_transf_train = FairBalance(data_train, class_balance=True)
        elif self.fair_balance=="Reweighing":
            RW = Reweighing(unprivileged_groups=unprivileged_groups,
                            privileged_groups=privileged_groups)
            RW.fit(data_train)
            dataset_transf_train = RW.transform(data_train)
        else:
            dataset_transf_train = data_train

        if self.fair_balance=="AdversialDebiasing":
            tf.reset_default_graph()
            sess = tf.Session()
            self.model = AdversarialDebiasing(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups,
                          scope_name='debiased_classifier',
                          debias=True,
                          sess=sess)
            self.model.fit(dataset_transf_train)
            preds = self.model.predict(data_test).labels.ravel()
            sess.close()
        else:
            scale_orig = StandardScaler()
            X_train = scale_orig.fit_transform(dataset_transf_train.features)
            y_train = dataset_transf_train.labels.ravel()

            self.model.fit(X_train, y_train, sample_weight=dataset_transf_train.instance_weights)

            X_test = scale_orig.transform(data_test.features)
            preds = self.model.predict(X_test)

        if self.fair_balance=="RejectOptionClassification":
            pos_ind = numpy.where(self.model.classes_ == dataset_transf_train.favorable_label)[0][0]
            data_train_pred = dataset_transf_train.copy(deepcopy=True)
            data_train_pred.scores = self.model.predict_proba(X_train)[:,pos_ind].reshape(-1,1)
            data_test_pred = data_test.copy(deepcopy=True)
            data_test_pred.scores = self.model.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)
            metric_name = "Statistical parity difference"
            metric_ub = 0.05
            metric_lb = -0.05
            ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups,
                                             low_class_thresh=0.01, high_class_thresh=0.99,
                                              num_class_thresh=100, num_ROC_margin=50,
                                              metric_name=metric_name,
                                              metric_ub=metric_ub, metric_lb=metric_lb)
            try:
                ROC.fit(dataset_transf_train, data_train_pred)
            except:
                return None
            preds = ROC.predict(data_test_pred).labels.ravel()


        y_test = data_test.labels.ravel()
        result = self.evaluate(numpy.array(preds), y_test, data_test)
        tmp = self.assess_unawareness(data_test)
        for key in self.data.protected_attribute_names:
            result[key]['unawareness'] = tmp[key]
        return result

    def assess_unawareness(self, X_test):
        result = {}
        test_set = X_test.convert_to_dataframe()[0].drop(X_test.label_names, axis=1)
        for key in self.data.protected_attribute_names:
            test_set2 = test_set.to_dict('list')
            m = len(test_set)
            for j in range(m):
                for k in test_set.columns:
                    if k == key:
                        test_set2[k][j] = 1.0 - test_set[k][j]
                    else:
                        test_set2[k][j] = test_set[k][j]
            test_set2 = pd.DataFrame(test_set2, columns = test_set.columns)
            pred1 = self.model.predict(test_set)
            pred2 = self.model.predict(test_set2)
            protected = test_set[key].to_numpy()
            max = np.max(protected)
            min = np.min(protected)
            protected = (protected - (max+min)/2)/((max-min)/2)
            result[key] = np.sum((pred1 - pred2)*protected)/m
        return result

    def evaluate(self, preds, truth, X_test):
        def rate(a, b):
            aa = Counter(a)[True]
            bb = Counter(b)[True]
            if aa+bb == 0:
                return 0
            else:
                return aa / float(aa+bb)

        result = {}
        # Get target label (for calculating the confusion matrix)
        target = max(set(self.y))
        pp = preds == target
        np = preds != target
        pg = truth == target
        ng = truth != target
        tp = pp & pg
        fp = pp & ng
        tn = np & ng
        fn = np & pg
        result["tpr"] = rate(tp, fn)
        result["fpr"] = rate(fp, tn)
        result["prec"] = rate(tp, fp)
        result["acc"] = rate(tp | tn, fp | fn)
        if (result["tpr"]+result["prec"]) == 0:
            result["f1"] = 0
        else:
            result["f1"] = 2*result["tpr"]*result["prec"]/(result["tpr"]+result["prec"])
        for i, key in enumerate(self.data.protected_attribute_names):
            result[key] = {}
            group1 = X_test.protected_attributes[:,i] == 1
            group0 = X_test.protected_attributes[:,i] == 0
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
            pr1 = rate(tp1 | fp1, tn1 | fn1)
            pr0 = rate(tp0 | fp0, tn0 | fn0)
            result[key]["eod"] = tpr0 - tpr1
            result[key]["aod"] = 0.5*(fpr0-fpr1+tpr0-tpr1)
            result[key]["spd"] = pr0 - pr1
        return result


