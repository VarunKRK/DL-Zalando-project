import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, no_grad
from torch import optim
import torch.nn.functional as F
import data_handler as dh


model = nn.Sequential(nn.Linear(784, 256),
                      nn.ReLU(),
                     
                      nn.Dropout(0.4),
                      nn.Linear(256, 128),
                      nn.ReLU(),                    
                      nn.Dropout(0.3),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Dropout(0.2),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim = 1)
                     )

trainset, testset, trainloader, testloader = dh.get_data()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-3)

epochs = 30

train_loss = []
test_loss = []

for e in range(epochs):
    running_loss = 0

    for images, labels in trainloader:

        images = images.resize_(images.size()[0], 784)

        optimizer.zero_grad()

        pred = model.forward(images)

        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    else:
        loss_test = 0
        accuracy = 0

        with torch.no_grad():

            model.eval()

            for images, labels in testloader:
                images = images.resize_(images.size()[0], 784)
                log_ps = model(images)
                loss_test += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

    model.train()

    train_loss.append(running_loss/len(trainloader))
    test_loss.append(loss_test/len(testloader))

    print(f'Epoch: {e + 1} | trainloss: {running_loss/len(trainloader)} | test loss: {loss_test/len(testloader)} | accuracy : {accuracy/len(testloader)}' )


plt.plot(train_loss, label = 'Trainloss')
plt.plot(test_loss, label = 'Testloss')
plt.plot()
plt.legend()
plt.show()

print("My model: \n\n", model, "\n")
print("The state dict keys: \n\n", model.state_dict().keys())


torch.save(model.state_dict(), 'checkpoint.pth')