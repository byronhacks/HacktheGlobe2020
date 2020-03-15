import pandas as pd
import numpy as np
import torch

N, D_in, H, D_out = 1989251, 1440, 100, 1

inp = pd.read_csv("coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")
# print(inp.dropna(axis=0,how="any"))
foruse = inp.dropna(axis=0,how="any")
weightedAve = foruse["Weighted_Price"]
fakeX = np.array([weightedAve.to_numpy()[m:m+1440] for m in range(len(weightedAve)-1440)])
x = torch.from_numpy(fakeX)
fakeY = np.array([weightedAve.to_numpy()[m] for m in range(1440,len(weightedAve))])
y = torch.from_numpy(fakeY)
# print(len(fakeY))
# print(len(fakeX))
# print(x)
# print(y)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4

for t in range(500):

    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    model.zero_grad()


    loss.backward()


    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

