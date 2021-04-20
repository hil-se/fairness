from FairBalance import FairBalance

from demos import cmd
from pdb import set_trace

def exp():
    fb = FairBalance("LR",data="german",fair_balance=False)
    fb.cross_val()



if __name__ == "__main__":
    eval(cmd())