from demos import cmd
import copy
try:
   import cPickle as pickle
except:
   import pickle
from utils import *
from experiment import Experiment


def one_exp(treatment, data, fair_balance, target="", repeats=10):
    # Conduct one experiment:
    #     treatment in {"SVM", "RF", "LR", "DT"}
    #     data in {"compas", "adult", "german"}
    #     fair_balance in {"None", "FairBalance", "Reweighing", "AdversialDebiasing", "RejectOptionClassification"}
    #     target = target protected attribute, not used if fair_balance == "FairBlance" or "None"
    #     repeats = number of times repeating the experiments

    exp = Experiment(treatment, data=data, fair_balance=fair_balance, target_attribute=target)
    results = {}
    for _ in range(repeats):
        result = exp.run()
        if result:
            results = merge_dict(results, result)
    print(results)
    return results

def RQ1():
    # Perform an overall experiment on different algorithms, datasets, and FairBalance settings.
    treatments = ["LR", "SVM", "DT", "RF", "NB"]
    datasets = ["compas", "adult", "german"]
    balances = ["None", "FairBalance", "FairBalanceClass"]
    results = {}
    for treatment in treatments:
        results[treatment] = {}
        for dataset in datasets:
            results[treatment][dataset] = {}
            for balance in balances:
                results[treatment][dataset][balance] = one_exp(treatment, dataset, balance, repeats=50)
                # Print progress
                print(treatment+", "+dataset+", "+balance)
    # dump results
    with open("../dump/RQ1.pickle", "wb") as p:
        pickle.dump(results, p)
    parse_results_RQ1()

def RQ3():
    # Compare FairBalance against other soa baseline bias mitigation algorithms.
    # Classifier is fixed to logistic regression.
    treatment = "LR"
    datasets = ["compas", "adult", "german"]
    balances = ["Reweighing", "AdversialDebiasing", "RejectOptionClassification", "FairBalance", "FairBalanceClass"]
    targets = {"compas": ["sex", "race"], "adult": ["sex", "race"], "german": ["sex", "age"]}
    results = {}
    for dataset in datasets:
        results[dataset] = {}
        for balance in balances:
            if balance!="FairBalance" and balance!="FairBalanceClass":
            # Need target attribute
                for target in targets[dataset]:
                    results[dataset][balance+": "+target] = one_exp(treatment, dataset, balance, target=target)
            else:
                results[dataset][balance] = one_exp(treatment, dataset, balance)
            # Print progress
            print(dataset + ", " + balance)
    # dump results
    with open("../dump/RQ3.pickle", "wb") as p:
        pickle.dump(results, p)
    parse_results_RQ3()

def parse_results_RQ1(iqr="True"):
    # Parse results of RQ1 and save as csv files.
    with open("../dump/RQ1.pickle", "rb") as p:
        results = pickle.load(p)
    # Compare results of FairBalance against None
    compares = copy.deepcopy(results)
    for treatment in compares:
        for dataset in compares[treatment]:
            compares[treatment][dataset] = compare_dict(compares[treatment][dataset], baseline = "None")
    compare_df = dict2dfRQ1(compares)
    compare_df.to_csv("../results/RQ1_compare.csv", index=False)

    # Calculate medians and iqrs of 50 repeats
    medians = copy.deepcopy(results)
    medians = median_dict(medians, use_iqr = iqr=="True")
    median_df = dict2dfRQ1(medians)
    median_df.to_csv("../results/RQ1_median.csv", index=False)

    # Color the median csv
    colored = color(medians, compares)
    colored_df = dict2dfRQ1(colored)
    colored_df.to_csv("../results/RQ1_color.csv", index=False)


def parse_results_RQ3(iqr="True"):
    # Parse results of RQ3 and save as csv files.
    with open("../dump/RQ3.pickle", "rb") as p:
        results = pickle.load(p)
    # Compare results of other treatments against FairBalance
    compares = copy.deepcopy(results)
    for dataset in compares:
        compares[dataset] = compare_dict(compares[dataset], baseline = "FairBalance")
    compare_df = dict2dfRQ3(compares)
    compare_df.to_csv("../results/RQ3_compare.csv", index=False)

    # Calculate medians and iqrs of 10 repeats
    medians = copy.deepcopy(results)
    medians = median_dict(medians, use_iqr = iqr=="True")
    median_df = dict2dfRQ3(medians)
    median_df.to_csv("../results/RQ3_median.csv", index=False)

    # Color the median csv
    colored = color(medians, compares)
    colored_df = dict2dfRQ3(colored)
    colored_df.to_csv("../results/RQ3_color.csv", index=False)

def exp_injection1(repeats=10):
    inject_place = "Train"
    inject_ratio = {"sex":[0.2,0.0]}
    exp_injection("LR", "compas", "FairBalance", inject_place, inject_ratio, repeats)

def exp_injection(treatment, data, fair_balance, inject_place, inject_ratio, repeats=10):
    # Conduct one experiment:
    #     treatment in {"SVM", "RF", "LR", "DT"}
    #     data in {"compas", "adult", "german"}
    #     fair_balance in {"None", "FairBalance"}
    #     inject_place in {"None", "All", "Train"}
    #     inject_ratio={attribute1: [ratio11, ratio12], attribute2: [ratio21, ratio22], ...}
    #     repeats = number of times repeating the experiments

    exp = Experiment(treatment, data=data, fair_balance=fair_balance)
    exp.inject_bias(inject_place, inject_ratio)
    results = {}
    for _ in range(repeats):
        result = exp.run()
        if result:
            results = merge_dict(results, result)
    print(results)
    return results

if __name__ == "__main__":
    eval(cmd())