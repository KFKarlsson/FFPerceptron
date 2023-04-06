import torch
import time

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

def one_hot_encode(img0, lab):
    img = img0.clone()
    img[:, :10] = img0.min()
    img[range(img0.shape[0]), lab] = img0.max()
    return img

#Load MNIST Data
train_loader = DataLoader(
    MNIST('./MNIST_data/', train=True,
    download=True,
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,)), Lambda(lambda x: torch.flatten(x))])),
    batch_size=60000)

test_loader = DataLoader(
    MNIST('./MNIST_data/', train=False,
    download=True,
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,)), Lambda(lambda x: torch.flatten(x))])),
    batch_size=10000)
    
dtype = torch.float

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# Training images
img0, lab = next(iter(train_loader))
img0 = img0.to(device)

# Validation images
img0_tst, lab_tst = next(iter(test_loader))
img0_tst = img0_tst.to(device)

# Forward Forward Applied to a Single Perceptron for MNIST Classification
n_input, n_out = 784, 125
batch_size, learning_rate = 10, 0.0003
g_threshold = 10
epochs = 250

perceptron = torch.nn.Sequential(torch.nn.Linear(n_input, n_out, bias = True),
                      torch.nn.ReLU())

perceptron.to(device)
optimizer = torch.optim.Adam(perceptron.parameters(), lr = learning_rate)

N_trn = img0.size(0) #Use all training images (60000)

tic = time.time()

for epoch in range(epochs):
    img = img0.clone()

    for i in range(N_trn): # Random jittering of training images up to 2 pixels
        dx, dy = torch.randint(-2, 2, (2,))
        img[i] = torch.roll(img0[i].reshape(28, 28), shifts=(dx, dy), dims=(0, 1)).flatten()

    
    perm = torch.randperm(N_trn)
    img_pos = one_hot_encode(img[perm], lab[perm]) # Good data (actual label)
    
    lab_neg = lab[perm] + torch.randint(low=1,high=10,size=(lab.size()))
    lab_neg = torch.where(lab_neg > 9, lab_neg - 10, lab_neg)
    img_neg = one_hot_encode(img[perm], lab_neg) # Bad data (random error in label)

    L_tot = 0

    for i in range(0, N_trn, batch_size):
        perceptron.zero_grad()

        # Goodness and loss for good data in batch
        img_pos_batch = img_pos[i:i+batch_size]
        g_pos = (perceptron(img_pos_batch)**2).mean(dim=1)
        loss = torch.log(1 + torch.exp(-(g_pos - g_threshold))).sum()

        # Goodness and loss for bad data in batch
        img_neg_batch = img_neg[i:i+batch_size]
        g_neg = (perceptron(img_neg_batch)**2).mean(dim=1)
        loss += torch.log(1 + torch.exp(g_neg - g_threshold)).sum()

        L_tot += loss.item()  # Accumulate total loss for epoch

        loss.backward()   # Compute gradients
        optimizer.step()  # Update parameters

    # Test model with validation set
    N_tst = img0_tst.size(0) # Use all test images (10000)
    
    #Evaluate goodness for all test images and labels 0...9
    g_tst = torch.zeros(10,N_tst).to(device)
    for n in range(10):
        img_tst = one_hot_encode(img0_tst, n)
        g_tst[n] = ((perceptron(img_tst[0:N_tst])**2).mean(dim=1)).detach()       
    predicted_label = g_tst.argmax(dim=0).cpu()

    # Count number of correctly classified images in validation set
    Ncorrect = (predicted_label == lab_tst).sum().cpu().numpy()

    print("Epoch ", epoch+1, ":\tLoss ", L_tot, " \tTime ", round(time.time() - tic), "s\tTest Error ", 100 - Ncorrect/N_tst*100, "%")