import torch
import numpy as np
import threading

def dummy(a, b, i):
    a[i] = b[i]

if __name__ == "__main__":
    device = torch.device("cuda:0")
    a = torch.zeros((4,3,3), requires_grad=False).to(device)
    b = torch.ones((4,3,3), requires_grad=True).to(device) + 1

    print("Before: a?", a.requires_grad)
    print("Before: b?", b.requires_grad)
    threads = []
    for k in range(4):
        x=threading.Thread(target=dummy, args=(a,b,k))
        threads.append(x)
        x.start()

    for x in threads:
        x.join()

    print("After: a", a)
    print("After: a?", a.requires_grad)
    # print("a", a)
    
