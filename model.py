from torch import nn

class Mnist(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.model = nn.Sequential(
      nn.Linear(28*28, 256),
      nn.ReLU(),
      nn.Dropout(0.2),

      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Dropout(0.2),
      
      nn.Linear(256, 10)
    )
  
  def forward(self, X):
    X = self.flatten(X) # 28*28 images to (784, 1)
    logits = self.model(X)
    return logits