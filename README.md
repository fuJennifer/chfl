# Cyclic Hierarchical Federated Learning (CHFL)

Official implementation of our IJCNN 2026 paper:

**"On the Convergence of Cyclic Hierarchical Federated Learning with Heterogeneous Data"**

---

## 🚀 Overview

This repository implements several advanced cyclic or sequential Federated Learning (FL) paradigms:

- Sequential Federated Learning (SFL) ([Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/b18e5d6a10ba57d5273871f38189f062-Paper-Conference.pdf))
- Cyclic Federated Learning (CFL) ([Paper](https://proceedings.mlr.press/v202/cho23b/cho23b.pdf))
- Cyclic Hierarchical Federated Learning (CHFL) ([Paper](https://openreview.net/pdf?id=PhLCPYsHCw))

These methods are designed to address **data heterogeneity**, **communication efficiency**, and **multi-level distributed systems**.

---

## 🧠 Key Features

- 🔁 Sequential client update (SFL)
- 🔄 Cyclic communication strategy (CFL)
- 🌐 Hierarchical aggregation across edge servers (CHFL)
- 📉 Improved convergence under heterogeneous data
- ⚙️ Flexible and configurable training pipeline

---

## 📊 Supported Datasets

- MNIST (CNN)
- CIFAR-10 (CNN)
- Shakespeare (LSTM)


---
## Key Parameters
| Parameter       | Description |
|----------------|------------|
| `--model`      | Model type (e.g., cnn, rnn) |
| `--dataset`    | Dataset name (mnist, cifar10, shakespeare) |
| `--num_users`  | Total number of clients |
| `--num_edges`  | Number of edge servers (for hierarchical FL) |
| `--epochs`     | Total number of global training rounds |
| `--edge_ep`    | Number of local training steps at edge |
| `--step`       | Cyclic update step size |
| `--pattern`    | Training mode (sfl, cfl, chfl / e2e) |
| `--gpu`        | GPU device (e.g., cuda:0) |
| `--pc`         | Number of participating clients per round |
| `--pg`         | Participation ratio / grouping factor |
| `--select_edge`| Number of selected edges per round |

## ⚙️ Environment Setup

```bash
conda create -n fl python=3.9
conda activate fl
pip install torch torchvision numpy matplotlib tqdm
```
## Try CHFL
```bash
python src/federated_main.py \
    --model=cnn \
    --dataset=mnist \
    --pattern=e2e \
    --hety=e \
    --pg=1 \
    --pc=10 \
    --gpu=cuda:0 \
    --epochs=100 \
    --edge_ep=2 \
    --num_users=50 \
    --num_edges=10 \
    --step=5 \
    --select_edge=10
```
## Try CFL
```bash
python src/federated_main.py \
    --model=cnn \
    --dataset=mnist \
    --pattern=cfl \
    --epochs=100
```
## Try SFL
```bash
python src/federated_main.py \
    --model=cnn \
    --dataset=mnist \
    --pattern=sfl \
    --epochs=100
```
