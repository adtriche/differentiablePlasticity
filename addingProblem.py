#testing regular and hebbian networks on adding problem

import torch, random
import numpy as np
from torch.autograd import Variable

dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0")

#define dimensions
dIn = 2
H = 32

#define model variables
xInput = torch.randn(dIn+H, H, device=device, dtype=dtype)
yOutput = torch.randn(1, device=device, dtype=dtype)

w1 = torch.randn(dIn+H, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, 1, device=device, dtype=dtype, requires_grad=True)

a = torch.randn(H, device=device, dtype=dtype, requires_grad=True)

tanh = torch.nn.Tanh()

#define hyperparameters
n = 0.5
lr = 0.01
epochs = 10000
batchLoss = 0.0
#seqLen = 10

resultsLoss = np.ndarray((epochs//100,1))
resultsCorrect = np.ndarray((epochs//100,1))
correct = 0
#define epoch
for epoch in range(epochs):
    #define batch of artificial data
    seqLen = random.randint(10,100)
    testInput = np.zeros((seqLen,2))
    for i in range(seqLen):
        testInput[i,0] = random.randint(0,1)
    i1 = random.randint(0,seqLen-1)
    i2 = random.randint(0,seqLen-1)
    while (i1 == i2):
        i2 = random.randint(0,seqLen-1)
    testInput[i1, 1] = testInput[i2,1] = 1
    runningSum = 0
    
    #zero-initialization of Hebbian weights
    h = torch.zeros(H, device=device, dtype=dtype, requires_grad=False)
    context = w1[1]
    for t in range(seqLen):
        #calculate current correct in/out
        if t == i1 or t == i2:
            runningSum += torch.tensor(testInput[t,0]*testInput[t,1])
        xInput = torch.cat((torch.tensor([testInput[i,0],testInput[i,1]], dtype=dtype), context.view(H)))
        xInput = xInput.view(1,dIn+H)
        #define network graph
        context = tanh(torch.mm(xInput, (w1 + torch.mul(a,h))))
        yPred = (torch.mm(context, w2))
        #update Hebbian weights
        #h = n*(torch.mm(context, w1.view(H,dIn+H))[0, 0:H]) + (1-n)*h #basic Hebb's Rule
        #h = h + n*(context.view(H).mul(w1.view(H,dIn+H)[0, 0:H] - context.view(H).mul(h))) #Oja's Rule
        h = tanh(xInput[0, 0:H]).mul(context) - (1-n)*h
    #calculate loss and gradients
        loss = (runningSum - yPred)**(2)
        batchLoss += loss/seqLen
        loss.backward(retain_graph=True)
    
    #apply gradient updates
        w1.data -= lr * w1.grad.data
        w2.data -= lr * w2.grad.data
        a.data -= lr * a.grad.data
    #clear calculated gradients from autograd
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        a.grad.data.zero_()
    batchLoss += loss
    if (loss**(0.5) < 0.1):
        correct += 1
    if (epoch % 100 == 0):
        print("Loss for epoch #" + str(epoch) + ": " + str(batchLoss.data/100))
        print("Correct guesses: " + str(correct))
        print("Sum: " + str(runningSum))
        print("Prediction: " + str(yPred.data))
        print("\n")
        resultsCorrect[epoch//100, 0] = correct
        resultsLoss[epoch//100, 0] = batchLoss.data/100
        batchLoss = 0.0
        correct = 0

np.save("adding_trainCorrect_10..100_bcm_n50", resultsCorrect)
np.save("adding_trainLoss_10..100_bcm_n50", resultsLoss)