import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
try:
   import cPickle as pickle
except:
   import pickle
from pdb import set_trace


def merge_dict(results, result):
    # Merge nested dictionaries
    for key in result:
        if type(result[key]) == dict:
            if key not in results:
                results[key] = {}
            results[key] = merge_dict(results[key], result[key])
        else:
            if key not in results:
                results[key] = []
            results[key].append(result[key])
    return results



def is_larger(x, y):
    # Check if results in x is significantly larger than those in y.
    # Return int values:
        # 0: not significantly larger
        # 1: larger with small effect size
        # 2: larger with medium effect size
        # 3: larger with large effect size

    # Mann Whitney U test
    U, pvalue = mannwhitneyu(x, y, alternative="greater")
    if pvalue>0.05:
        # If x is not greater than y in 95% confidence
        return 0
    else:
        # Calculate Cliff's delta with U
        delta = 2*U/(len(x)*len(y))-1
        # Return different levels of effect size
        if delta<0.147:
            return 0
        elif delta<0.33:
            return 1
        elif delta<0.474:
            return 2
        else:
            return 3

def compare_dict(results):
    # Compare results between w/ and w/o FairBalance
    x = results["True"]
    y = results["False"]
    for key in x:
        if type(x[key]) == dict:
            for key2 in x[key]:
                xx = x[key][key2]
                yy = y[key][key2]
                x[key][key2] = is_larger(xx, yy)
                y[key][key2] = is_larger(yy, xx)
        else:
            xx = x[key]
            yy = y[key]
            x[key] = is_larger(xx, yy)
            y[key] = is_larger(yy, xx)
    return results

def median_dict(results):
    # Compute median value of lists in the dictionary
    for key in results:
        if type(results[key]) == dict:
            results[key] = median_dict(results[key])
        else:
            med = np.median(results[key])
            iqr = np.percentile(results[key],75)-np.percentile(results[key],25)
            # results[key] = str(med)+" ("+str(iqr)+")"
            results[key] = "%.2f (%.2f)"%(med, iqr)
    return results

def dict2df(results):
    # Generate a pandas dataframe based on the dictionary
    columns = ["Algorithm", "Dataset", "FairBalance", "F1", "Accuracy", "Sex: AOD", "Sex: EOD", "Race: AOD", "Race: EOD", "Age: AOD", "Age: EOD"]
    df = {key:[] for key in columns}
    for treatment in results:
        for dataset in results[treatment]:
            # for balance in results[treatment][dataset]:
            for balance in ["False", "True"]:
                x = results[treatment][dataset][balance]
                df["Algorithm"].append(treatment)
                df["Dataset"].append(dataset)
                df["FairBalance"].append("before" if balance=="False" else "after")
                df["F1"].append(x["f1"])
                df["Accuracy"].append(x["acc"])
                if "sex" in x:
                    df["Sex: AOD"].append(x["sex"]["aod"])
                    df["Sex: EOD"].append(x["sex"]["eod"])
                else:
                    df["Sex: AOD"].append("")
                    df["Sex: EOD"].append("")
                if "race" in x:
                    df["Race: AOD"].append(x["race"]["aod"])
                    df["Race: EOD"].append(x["race"]["eod"])
                else:
                    df["Race: AOD"].append("")
                    df["Race: EOD"].append("")
                if "age" in x:
                    df["Age: AOD"].append(x["age"]["aod"])
                    df["Age: EOD"].append(x["age"]["eod"])
                else:
                    df["Age: AOD"].append("")
                    df["Age: EOD"].append("")
    df = pd.DataFrame(df, columns = columns)
    return df



