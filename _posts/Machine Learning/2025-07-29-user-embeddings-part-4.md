
# Part 4: Building a Recommendation Engine Using Embeddings (PyTorch Edition)

In this part, we’ll take a hands-on deep dive into building a recommendation engine using **user and item embeddings** in **PyTorch**, with a focus on:

- Learning user preferences via **contrastive learning**
- Efficient **sampling strategies**
- Evaluation using **ranking metrics** like **NDCG** and **Hit Rate**
- End-to-end PyTorch implementation

---

## Step 1: Problem Setup

We want to train a model to rank items for each user. The key idea is to learn vector representations (embeddings) of users and items such that similar users and items lie closer in the embedding space.

```python
import torch
import torch.nn as nn

class EmbeddingModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
    
    def forward(self, user_ids, item_ids):
        user_vecs = self.user_embedding(user_ids)
        item_vecs = self.item_embedding(item_ids)
        scores = (user_vecs * item_vecs).sum(dim=1)
        return scores
```

---

## Step 2: Contrastive Learning Objective

Instead of simple regression, we **contrast** a positive item with one or more negative samples:

```python
def bpr_loss(pos_score, neg_score):
    return -torch.log(torch.sigmoid(pos_score - neg_score)).mean()
```

For each user, we want the score of the positive item (clicked/viewed) to be **higher** than the negative ones.

---

## Step 3: Negative Sampling

Random negative sampling is fast but may not be effective. Use **in-batch negatives** or **popularity-based sampling**.

```python
import random

def sample_negative(user, num_items, interacted_items):
    neg = random.randint(0, num_items - 1)
    while neg in interacted_items[user]:
        neg = random.randint(0, num_items - 1)
    return neg
```

---

## Step 4: Evaluation Metrics

### Normalized Discounted Cumulative Gain (NDCG)
Measures how well the predicted ranking matches the ideal ranking.

### Hit Rate @K
Measures whether the true positive is among the top K predicted items.

```python
def hit_rate_at_k(predicted, ground_truth, k):
    return int(ground_truth in predicted[:k])

def ndcg_at_k(predicted, ground_truth, k):
    if ground_truth in predicted[:k]:
        index = predicted.index(ground_truth)
        return 1 / torch.log2(torch.tensor(index + 2.0))
    return 0.0
```

---

## Training Loop

```python
model = EmbeddingModel(num_users, num_items, embedding_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for user, pos_item in train_data:
        neg_item = sample_negative(user, num_items, interactions)
        user_tensor = torch.tensor([user])
        pos_tensor = torch.tensor([pos_item])
        neg_tensor = torch.tensor([neg_item])

        pos_score = model(user_tensor, pos_tensor)
        neg_score = model(user_tensor, neg_tensor)

        loss = bpr_loss(pos_score, neg_score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Summary

We’ve built the core of a recommender system from scratch using PyTorch embeddings. Key points:

- Embeddings represent user/item behavior
- Contrastive learning improves ranking effectiveness
- Evaluation with NDCG and Hit Rate helps measure true performance

In **Part 5**, we’ll explore using **transformers** and **pre-trained models** for next-item recommendation and cold-start problems.
