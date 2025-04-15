from withtorch import Mnist
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

DEVICE = torch.accelerator.current_accelerator()
model = Mnist().to(DEVICE)
model.load_state_dict(torch.load("mnist_weights.pth", weights_only=True))
model.eval()

test_data = datasets.MNIST(root="data", train=False, download=False, transform=ToTensor())

def inference():
  idx = torch.randint(len(test_data), size=(1,)).item()
  image, label = test_data[idx]
  
  plt.imshow(image.squeeze(), cmap="gray")

  image = image.to(DEVICE).unsqueeze(0) # readd channel
  with torch.no_grad():
    pred = model(image).argmax(dim=1).item()

  plt.title(f"Model prediction: {pred}, Correct label: {label}")
  plt.axis(False)
  plt.show()


inference()

    

