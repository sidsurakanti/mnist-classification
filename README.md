## Overview
MNIST digit classifier built with just NumPy and Math. It also has a Pytorch + a realtime web implementation using FastAPI and Next.js. 

<img src="/assets/mnistdemo.gif">
<table>
  <tr>
    <td><img src="/performance.png" width="400px"></td>
    <td><img src="/confusion_matrix.png" width="400px"></td>
  </tr>
</table>

## What I learned

- How to implement a neural net from scratch  
- Forward and backward propagation using pure math
- Vectorized operations with NumPy  
- Loss functions, softmax, and gradient descent
- Data normalization, preprocessing, and augumentation
- Visualizing model performance (loss, accuracy, confusion matrix)
- Optimizing data pipeline
- Pytorch
- Full-stack integration with FastAPI and Next.js

## Stack

![Python](https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/matplotlib-2067b8?style=for-the-badge&logo=matplotlib&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-%23ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/fastapi-005571?style=for-the-badge&logo=fastapi)
![Next.js](https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=next.js&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338bdf8.svg?style=for-the-badge&logo=tailwind-css&logoColor=white)

## Getting started

Play around with the project w/ demo.py or view the demo web app locally.

### Prerequisites

- Python 3.10+ 
- pip
- Node.js 18+
- npm
- PyTorch with CUDA


```bash
# run to check
python --version
node -v
pip --version
```

### Installation
1. Clone the repo
```bash
git clone https://github.com/sidsurakanti/mnist-digit-recog.git
cd /path/to/project/
```
2. Setup the model
```bash
# backend setup
pip install -r requirements.txt
fastapi dev predict.py
```
3. Start up the website
```bash
# frontend setup
cd frontend
npm install
npm run dev
```

The app should now be live on http://localhost:3000!

## Roadmap
- [x] Build a neural net with only NumPy  
- [x] Implement backpropagation 
- [X] Add softmax + cross-entropy   
- [x] Visualize accuracy/loss  
- [X] Rebuild using Pytorch
- [X] Add GUI to draw digits and classify in real-time 
- [X] Improve digit preprocessing for the HTML canvas
- [ ] Dockerize model
- [ ] Deploy online

## Contributing

Pull requests are welcome! Feel free to open an issue or suggestion. 