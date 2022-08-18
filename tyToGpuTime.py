import torch
import time
import sys

gpu = torch.device("cuda")

def ver():
    print("python version:", sys.version)
    print("pytorch version:", torch.__version__)
    print("\n")

def test():
    start = time.time()
    torch.cuda.init()
    print("cuda init:", time.time()-start)
    x = torch.randn(15000,3).float()
    print("randn initialized:", time.time()-start)
    x.to(gpu)
    print("to(gpu):", time.time()-start)
    torch.cuda.synchronize()
    print("time after sync:", time.time()-start)
    print("\n")

if __name__ == "__main__":
    ver()
    test()
    test()