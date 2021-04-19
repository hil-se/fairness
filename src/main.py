from FairBalance import FairBalance

from demos import cmd
from pdb import set_trace

def exp():
    fb = FairBalance("LR",data="compas",fair_balance=True)
    fb.cross_val()



if __name__ == "__main__":
    eval(cmd())