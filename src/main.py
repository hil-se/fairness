from FairBalance import FairBalance
import numpy as np
import pandas as pd
from demos import cmd
import copy
try:
   import cPickle as pickle
except:
   import pickle
from pdb import set_trace
from utils import *


def one_exp(treatment, data, fair_balance, repeats = 30):
    results = {}
    # Repeat 30 times for each unique setting
    for _ in range(repeats):
        fb = FairBalance(treatment,data=data,fair_balance=fair_balance=="True")
        result = fb.cross_val()
        results = merge_dict(results, result)
    # print(results)
    return results

def exps():
    # Perform an overall experiment on different algorithms, datasets, and FairBalance settings.
    treatments = ["SVM", "RF", "LR", "NB", "DT"]
    datasets = ["compas", "adult", "default", "heart", "german"]
    balances = ["True", "False"]
    results = {}
    for treatment in treatments:
        results[treatment] = {}
        for dataset in datasets:
            results[treatment][dataset] = {}
            for balance in balances:
                results[treatment][dataset][balance] = one_exp(treatment, dataset, balance)
                # dump results
                with open("../dump/results.pickle", "wb") as p:
                    pickle.dump(results, p)
                print(treatment+", "+dataset+", "+balance)


def parse_results():
    with open("../dump/results.pickle", "rb") as p:
        results = pickle.load(p)
    # Compare results between w/ and w/o FairBalance
    compares = copy.deepcopy(results)
    for treatment in compares:
        for dataset in compares[treatment]:
            compares[treatment][dataset] = compare_dict(compares[treatment][dataset])
    compare_df = dict2df(compares)
    compare_df.to_csv("../results/compare.csv", index=False)

    # Calculate medians and iqrs of 30 repeats
    medians = copy.deepcopy(results)
    medians = median_dict(medians)
    median_df = dict2df(medians)
    median_df.to_csv("../results/median.csv", index=False)



if __name__ == "__main__":
    eval(cmd())