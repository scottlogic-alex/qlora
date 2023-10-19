import torch
from torch import FloatTensor, ones, zeros
from torch.nn import Module, LayerNorm, Linear
from torch.optim import SGD
from torch.cuda.amp.autocast_mode import autocast

class Model(Module):
    ln: LayerNorm
    ln: Linear
    def __init__(self) -> None:
        super().__init__()
        self.lin = Linear(in_features=4, out_features=3)
        self.ln = LayerNorm((3,))
    
    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.lin.forward(x)
        x = self.ln.forward(x)
        return x

device = torch.device('cuda')
model = Model().to(device=device)
optim = SGD(model.parameters(), lr=2e-5)

x: FloatTensor = ones((1, 4), device=device)
true_y: FloatTensor = zeros((1, 3), device=device)

with autocast(dtype=torch.bfloat16):
    y: FloatTensor = model.forward(x)
    loss: FloatTensor = (true_y-y).mean()
loss.backward()
optim.step()

pass