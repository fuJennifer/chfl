# Cyclic Hierarchical Federated Learning (CHFL)

Official implementation of our IJCNN 2026 paper:

**"On the Convergence of Cyclic Hierarchical Federated Learning with Heterogeneous Data"**

---

## 🚀 Overview

This repository implements several advanced cyclic or sequential Federated Learning (FL) paradigms:

- Sequential Federated Learning (SFL) ([Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/b18e5d6a10ba57d5273871f38189f062-Paper-Conference.pdf))
- Cyclic Federated Learning (CFL) ([Paper](https://proceedings.mlr.press/v202/cho23b/cho23b.pdf))

These methods are designed to address **data heterogeneity**, **communication efficiency**, and **multi-level distributed systems**.

---


## 📊 Supported Datasets

- MNIST (CNN)
- CIFAR-10 (CNN)
- Shakespeare (LSTM)


---
## Key Parameters
| Parameter       | Description |
|----------------|------------|
| `--model`      | Model type (e.g., cnn, lstm) |
| `--dataset`    | Dataset name (mnist, cifar, shakespeare) |
| `--pattern`    | pattern type (e.g., e2e(cyclic pattern), e2s) |
| `--hety`       | heterogeneity (e.g., c(IID data), e(Non-IID data)) |
| `--pc`         | intra edge heterogeneity (e.g., 1,2,5,10) |
| `--pg`         | inter edge heterogeneity (e.g., 1,2,5,10) |
| `--epochs`     | Total number of global training rounds |
| `--edge_ep`    | Number of training rounds in one edge |
| `--num_users`  | Total number of clients |
| `--num_edges`  | Number of edge servers (for hierarchical FL) |
| `--step`       | Number of training steps in one client |
| `--select_edge`| Number of selected edges per round |

## ⚙️ Environment Setup

Please read requirment.txt and recommend use conda environment.
## Try MNIST
```bash
python src/federated_main.py --model=cnn --dataset=mnist --pattern=e2e --hety=c --pc=1 --pg=10 --gpu=cuda:0 --epochs=1 --edge_ep=1 --num_users=50 --num_edges=10 --step=5 --select_edge=2
```
## Try CIFAR
```bash
python src/federated_main.py --model=cnn --dataset=cifar --pattern=e2e --hety=e --pg=1 --pc=10 --gpu=cuda:0 --epochs=1 --edge_ep=2 --num_users=50 --num_edges=10 --step=5 --select_edge=10
```
## Try Shakespeare
```bash
python src/federated_main.py --model=lstm --dataset=shakespeare --pattern=e2s --hety=e --pg=1 --pc=10 --gpu=cuda:0 --epochs=2 --edge_ep=1 --num_users=139 --num_edges=12 --step=12 --select_edge=5
```
