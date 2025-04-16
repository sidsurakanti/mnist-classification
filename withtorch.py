import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.v2 import ToImage, ToDtype, RandomAffine
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 

# %%
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

# %%
data_aug = transforms.Compose([
    ToImage(),
    RandomAffine(
        degrees=10,
        translate=(0.15, 0.15),
        scale=(0.8, 1.1),
        shear=10
    ),
    ToDtype(torch.float32, scale=True)
])

test_transforms = transforms.Compose([
    ToImage(),
    ToDtype(torch.float32, scale=True)
])

trainD = datasets.MNIST(root="data", train=True, download=True, transform=data_aug)
testD = datasets.MNIST(root="data", train=False, download=True, transform=test_transforms)

# %%
BATCH_SIZE = 128
DEVICE = torch.accelerator.current_accelerator()

train_DL = DataLoader(trainD, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
test_DL = DataLoader(testD, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

for X, y in test_DL:
  print("shape x:", X.shape) # batchsize, channels, w, h
  print("shape y:", y.shape, y.dtype), # label shape of batch
  break

# %%
loss_fn = nn.CrossEntropyLoss()

# %%
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

# %%
def accuracy(output, y):
  preds = torch.argmax(output, dim=1)
    # y (labels) == preds.argmax
  return (preds == y).float().mean()*100

# %%
def train(dataloader, model, loss_fn, optimizer=None):
  model.train()
  size = len(dataloader.dataset)
  lr = 1e-3
  loss_t = []

  for batch_i, (X, y) in enumerate(dataloader): # in batches from DL
    X, y = X.to(DEVICE), y.to(DEVICE)

    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()

    if (not optimizer):
      # -- MANUAL SGD
      with torch.no_grad():
        for param in model.parameters():
          param -= param.grad * lr # update gradients
      model.zero_grad() # reset gradients
    else:
      # optimization
      optimizer.step()
      optimizer.zero_grad() # clear gradients

    if (batch_i) % 100 < 1:
      loss_t.append(loss.item())
      print(f"Curr: {batch_i*BATCH_SIZE + len(X)}/{size}, Loss = {loss.item():.4f}")
    


# %%
def testing(dataloader, model, loss_fn):
  model.eval()
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss = correct = 0

  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(DEVICE), y.to(DEVICE)

      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

  avg_loss = test_loss / num_batches
  accuracy = correct / size

  print("Test completed,")
  print(f"Accuracy: {accuracy*100:.2f}%, Avg Loss: {avg_loss:.4f}\n")

  return accuracy, avg_loss

# %%
def fit(epochs: int):
  model = Mnist().to(DEVICE)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  acc = []
  loss = []

  print("Starting!")
  for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
    train(train_DL, model, loss_fn, optimizer)
    epoch_acc, epoch_loss = testing(test_DL, model, loss_fn)
    acc.append(round(epoch_acc*100))
    loss.append(epoch_loss)
    
  print("Done!")

  torch.save(model.state_dict(), "mnist_weights.pth")
  print("Weights saved to `mnist_weights.pth`")
  return acc, loss

# %%
if __name__ == "__main__":
  EPOCHS = 10

  acc, loss = fit(EPOCHS)
  plt.subplot(211)
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy (%)")
  plt.plot(acc, color="blue")

  plt.subplot(212)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.plot(loss, label="red")

  plt.suptitle("Model Performance")
  plt.grid()
  plt.tight_layout()
  plt.savefig("performance.png")
  plt.show()

# %%
def confusion_test():
  model = Mnist().to(DEVICE)
  dataloader = test_DL
  model.eval()
  y_true, y_pred = [], []

  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(DEVICE), y.to(DEVICE)
      preds = model(X)
      
      y_true += y.cpu().numpy().astype(int).tolist()
      pred_labels = preds.argmax(dim=1)
      y_pred += pred_labels.cpu().numpy().astype(int).tolist()

  cm = confusion_matrix(y_true, y_pred, normalize='all')

  ConfusionMatrixDisplay(cm).plot(values_format=".2f")
  plt.savefig("confusion_matrix.png")
  print("Confusion matrix saved")

# %%
if __name__ == "__main__":
  confusion_test()


