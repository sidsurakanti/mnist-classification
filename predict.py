from model import Mnist

import torch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Mnist().to(DEVICE)
model.load_state_dict(torch.load("mnist_weights.pth", weights_only=True, map_location=DEVICE))
model.eval()

class Input(BaseModel):
  data: list[list[float]]

# fastapi dev predict.py
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(input: Input):
    x = torch.tensor(input.data, dtype=torch.float32).reshape(1, 784).to(DEVICE)
    with torch.no_grad():
        pred = model(x).argmax(dim=1).item()
    return {"prediction": pred}

