#benchmarking against Fast Weights network for MNIST

import torch, random, os
import numpy as np
from torch.autograd import Variable
from matplotlib import pyplot as plt

dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0")

#define dimensions
dIn = 1#196
H = 28

#define model variables
xInput = torch.randn(dIn+H, H, device=device, dtype=dtype)
yOutput = torch.randn(10, device=device, dtype=dtype)

w1 = torch.randn(dIn+H, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, H, device=device, dtype=dtype, requires_grad=True)
w3 = torch.randn(H, 1, device=device, dtype=dtype, requires_grad=True)

a = torch.randn(H, device=device, dtype=dtype, requires_grad=True)


sigmoid = torch.nn.Sigmoid()
tanh = torch.nn.Tanh()

#define hyperparameters
n = 0.1
lr = 0.1
epochs = 100000
batchLoss = 0.0
batch = 1
#seqLen = 10



resultsLoss = np.ndarray((epochs//100,1))
resultsCorrect = np.ndarray((epochs//100,1))
#define epoch
for epoch in range(epochs):
    for b in range(batch):
        #define batch of artificial data
        r = random.randint(0,9)
        fname = random.choice(os.listdir("C:/Users/antho/Downloads/MNIST/trainingSet.tar.gz/trainingSet.tar/trainingSet/" + str(r) + "/"))
        sample= plt.imread("C:/Users/antho/Downloads/MNIST/trainingSet.tar.gz/trainingSet.tar/trainingSet/" + str(r) + "/" + fname)
#        q1 = torch.tensor(sample[0:14, 0:14]*(1/255),dtype=dtype).view(196)
#        q2 = torch.tensor(sample[0:14, 14:28]*(1/255),dtype=dtype).view(196)
#        q3 = torch.tensor(sample[14:28,0:14]*(1/255),dtype=dtype).view(196)
#        q4 = torch.tensor(sample[14:28, 14:28]*(1/255),dtype=dtype).view(196)
#        q5 = torch.tensor(sample[7:21, 7:21]*(1/255),dtype=dtype).view(196)
        #q6 = torch.tensor(sample[4:18, 7:21]*(1/255),dtype=dtype).view(196)
        #q7 = torch.tensor(sample[10:24, 7:21]*(1/255),dtype=dtype).view(196)
#        glimpses = [q1,q2,q3,q4,q5]#, q6, q7]
        label = torch.zeros(10)
        label[r] = 1
        label = label.view(1,10)
        #zero-initialization of Hebbian weights
        h = torch.zeros(H, device=device, dtype=dtype, requires_grad=False)
        context = pre = torch.zeros(H)
        sample = sample*(1.0/255.0)
        sample = torch.tensor(sample,dtype=dtype).view(784)
        for i in range(784):
            xInput = torch.cat((sample[i].view(1), pre.view(H)))
            xInput = xInput.view(1,dIn+H)
            #define network graph
            context = tanh(torch.mm(xInput, (w1 + torch.mul(a,h))))
            pre = (torch.mm(context, w2)).clamp(min=0.001)
            yPred = ((torch.mm(pre, w3))).clamp(min=0.0)
            #update Hebbian weights
            #h = n*(torch.mm(context,pre.transpose(0,1))) + (1-n)*h
            #h = h + n*(pre.mm((context - pre.mul(h)).transpose(0,1))) #Oja's Rule
            h = n*(torch.mm(context, w1.view(H,dIn+H))[0, 0:H]) + (1-n)*h #basic Hebb's Rule
            #h = h + n*(context.view(H).mul(w1.view(H,dIn+H)[0, 0:H] - context.view(H).mul(h))) #Oja's Rule
            #h = tanh(xInput[0, 0:H]).mul(context) - (1-n)*h #BCA learning rule
            #calculate loss and gradients
        loss = torch.nn.functional.mse_loss(yPred.view(1), torch.tensor([r]).float())
        loss.backward(retain_graph=True)
        batchLoss += loss
    #apply gradient updates once per batch
    w1.data -= lr * w1.grad.data
    w2.data -= lr * w2.grad.data
    w3.data -= lr * w3.grad.data
    a.data -= lr * a.grad.data
    #clear calculated gradients from autograd
    w1.grad.data.zero_()
    w2.grad.data.zero_()
    w3.grad.data.zero_()
    a.grad.data.zero_()
    #batchLoss += loss
    if (epoch % 100 == 0):
        print("Loss for epoch #" + str(epoch) + ": " + str(batchLoss.data/100))
        print("R: " + str(r))
        #print("Label: " + str(label.data))
        print("Prediction: " + str(yPred.data))
        print("\n")
        batchLoss = 0.0

#np.save("adding_trainCorrect_10..100_bcm_n50", resultsCorrect)
#np.save("adding_trainLoss_10..100_bcm_n50", resultsLoss)
        
###############################################################################Recycling Bin
