import os, torch, random, time
import numpy as np
import h5py as hpy

dtype = torch.float
device = torch.device("cpu")

#define dimensions
dIn = 128 #128 values per time t
H = 64
dOut = 527 #number of classes

#define model variables
w1 = torch.randn(dIn+H, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, H, device=device, dtype=dtype, requires_grad=True)
w3 = torch.randn(H, dOut, device=device, dtype=dtype, requires_grad=True)

b1 = torch.randn(1, H, device=device, dtype=dtype, requires_grad=True)
b2 = torch.randn(dIn + H, dOut, device=device, dtype=dtype, requires_grad=True)

a = torch.randn(H, device=device, dtype=dtype, requires_grad=True)

sigmoid = torch.nn.Sigmoid()
tanh = torch.nn.Tanh()

#define hyperparameters
n = 0.1
lr = 0.1
epochs = 100000
batchSize = 10

def load_test_set():
    folder = os.getcwd() + "/AudioSet Dataset/"
    data = hpy.File(folder + "eval.h5", 'r')
    vidIDs = np.array(data.get("video_id_list"))
    x = np.array(data.get("x"))
    y = np.array(data.get("y"),dtype=int)
    data.close()
    print(x.shape)
    print(y.shape)
    return x, y, vidIDs

def load_train_set():
    folder = os.getcwd() + "/AudioSet Dataset/"
    data = hpy.File(folder + "bal_train.h5", 'r')
    vidIDs = np.array(data.get("video_id_list"))
    x = np.array(data.get("x"))
    y = np.array(data.get("y"),dtype=int)
    data.close()
    return x, y, vidIDs

def train(x, y):
    batchLoss = 0.0
    context = torch.zeros(H, dtype=dtype,device=device)
    start = time.time()
    for k in range(batchSize):
        h = torch.zeros(H, device=device, dtype=dtype, requires_grad=False)
        for t in range(10):
            #prepare a training sample
            index = random.randint(0,len(x)-1)
            xInput = torch.cat((torch.tensor(x[index,t,:], dtype=dtype, device=device, requires_grad=False), context.view(H))).view(1,dIn+H)
            #network operations
            del(context)
            #context = tanh(torch.mm(xInput, w1) + b1)
            context = tanh(torch.mm(xInput, (w1 + torch.mul(a,h))) + b1)
            hidden = tanh(torch.mm(context, (w2)))
            prediction = torch.sigmoid((torch.mm(hidden, w3)))
            #Hebbian update
            h = n*(torch.mul(context.data, hidden.data)) + (1-n)*h
        label = torch.tensor(y[index,:], device=device, requires_grad=False).float()
        loss = torch.nn.functional.binary_cross_entropy(prediction.view(dOut), label)
        loss.backward(retain_graph=True)
        batchLoss += loss.data
        del(loss)
        del(h)
        del(xInput)
        del(label)
        del(prediction)
        w1.data -= lr * w1.grad.data
        w2.data -= lr * w2.grad.data
        b1.data -= lr * b1.grad.data
        w3.data -= lr * w3.grad.data
        a.data -= lr * a.grad.data
    #clear calculated gradients from autograd
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b1.grad.data.zero_()
        w3.grad.data.zero_()
        a.grad.data.zero_()
    end = time.time()
    return (end - start), batchLoss.data

def test(testX, testY):
    context = torch.zeros(H, dtype=dtype,device=device)
    results = np.ndarray(testY.shape)
    for k in range(len(x)-1):
        h = torch.zeros(H, device=device, dtype=dtype, requires_grad=False)
        for t in range(10):
            xInput = torch.cat((torch.tensor(x[k,t,:], dtype=dtype, device=device, requires_grad=False), context.view(H))).view(1,dIn+H)
            #network operations
            del(context)
            #context = tanh(torch.mm(xInput, w1) + b1)
            context = tanh(torch.mm(xInput, (w1 + torch.mul(a,h))) + b1)
            hidden = tanh(torch.mm(context, (w2)))
            prediction = torch.sigmoid((torch.mm(context, w3)))
            #Hebbian update
            h = n*(torch.mul(context.data, hidden.data)) + (1-n)*h
        results[k] = prediction.view(dOut).data
        del(h)
        del(xInput)
        del(prediction)
    return results

            
print(type((int(epochs/10))))    
    
x, y, ids = load_train_set()
epochLoss = 0.0
for epoch in range(epochs):
    timeTaken, batchLoss = train(x, y)
    epochLoss += batchLoss
    if (epoch % 10 == 0):
        print("Epoch " + str(epoch))
        print("Loss: " + str(epochLoss / 100))
        print("Time Taken: " + str(timeTaken))
        lr = lr*0.9999999
        epochLoss = 0.0

x, y, ids = load_test_set()
results = test(x, y)
np.save("testResults_100000epochs_ojarule_ce", results)


