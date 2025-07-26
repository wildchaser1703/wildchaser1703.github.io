---
title: "Understanding User Embeddings: The Heart of Modern Recommendation Systems, Part - 1"
date: 2025-07-19
tags: [machine learning, recommendation systems, embeddings]
categories: [Machine Learning]
layout: post
---

## Introduction

In the world of modern machine learning and recommendation systems, **user embeddings** are the quiet yet powerful drivers behind how platforms like TikTok, Netflix, Amazon, and Spotify seem to know what you want. They are one of the foundational components that make personalization work at scale.

This post will dive into what user embeddings are, how they work, and how you can build and use them in your own machine learning workflows.

---

## What Are User Embeddings?

At its core, a **user embedding** is a fixed-length vector (e.g., 64- or 128-dimensional) that represents a user in a numerical format. The idea is to translate **complex user behavior and preferences** into a dense, continuous vector space where:

- Users with similar behaviors are placed **close** to each other.
- Users with dissimilar behaviors are placed **far apart**.

This representation captures **latent patterns**, things that may not be directly observable but can be inferred from behavior over time.

---

## Why Do We Need Embeddings?

Imagine trying to build a recommendation system by storing every user’s browsing history, clicked items, liked products, and time spent per item. This approach becomes unmanageable as your data grows.

Embeddings solve this by compressing all that information into a compact vector that still captures the **essence of the user's preferences**.

### Benefits:
- Scalable to millions of users
- Works well with deep learning models
- Fast similarity computation (dot product, cosine similarity)

---

## How Are User Embeddings Learned?

User embeddings are typically **learned** through training a model and not hardcoded.

Here’s a common high-level pipeline:

### 1. **Collect Interaction Data**
Track clicks, views, likes, ratings, dwell time, skips, etc.

### 2. **Build a Two-Tower Neural Network**
- One tower learns **user embeddings**
- Another tower learns **item embeddings**
- The model is trained so that user vectors are **close to the items** they interacted with, and far from those they didn't.

### 3. **Optimize with a Contrastive Loss**
This ensures that users are embedded near relevant items.

```python
# Pseudo PyTorch contrastive loss
positive_score = torch.dot(user_vector, pos_item_vector)
negative_scores = torch.matmul(user_vector, neg_item_vectors.T)

loss = -torch.log(torch.sigmoid(positive_score)) +        torch.sum(torch.log(torch.sigmoid(-negative_scores)))
```

---

## Real-World Use Case: TikTok Recommendations

TikTok is a masterclass in embedding-driven personalization.

- Every time you watch or skip a video, your user embedding is updated.
- Videos are also represented as embeddings.
- TikTok ranks videos by computing **dot product similarity** between your user vector and video vectors.
- The top-k most similar videos are shown to you next.

Over time, your embedding changes as your interests evolve — making the feed feel “addictively” personalized.

---

## Building Your Own: Example with TensorFlow Recommenders

Let’s say you have a dataset of users and movies they rated.

```python
import tensorflow as tf
import tensorflow_recommenders as tfrs

# Sample vocab
user_ids = ["alice", "bob", "carol"]
movie_titles = ["Inception", "Avengers", "Toy Story"]

embedding_dim = 32

# User and item towers
user_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(vocabulary=user_ids),
    tf.keras.layers.Embedding(len(user_ids)+1, embedding_dim)
])

movie_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(vocabulary=movie_titles),
    tf.keras.layers.Embedding(len(movie_titles)+1, embedding_dim)
])

# Recommendation model
model = tfrs.models.Model(user_model=user_model, item_model=movie_model)
```

You can train this using a retrieval task, evaluate it, and then use the trained user embeddings to serve recommendations in production.

---

## Other Use Cases of User Embeddings

- **Spotify**: Learn musical preferences based on skip behavior, time of day, etc.
- **YouTube**: Represent viewers based on watch history, duration, and device.
- **Amazon**: Show product recommendations based on item-user vector similarity.
- **Search Engines**: Rank results by matching query/user embeddings with document embeddings.

---

## Embeddings and Time

It’s important to remember that **preferences change**.

### Strategies to adapt:
- Use **time-decayed weighting** for recent interactions.
- Periodically re-train embeddings.
- Combine short-term (session-based) and long-term embeddings.

---

## Potential Pitfalls

- **Bias reinforcement**: Embeddings can reinforce stereotypes or lock users into echo chambers.
- **Cold start**: New users (no history) have poor embeddings until interactions occur.
- **Privacy**: Embeddings must be securely stored and ethically used.

---

## Final Thoughts

User embeddings are more than just numerical tricks. They're **compact representations of human behavior** which are dynamic, evolving, and personal. Used wisely, they power some of the most magical experiences on the internet. Used carelessly, they can be intrusive or misleading.

Understanding how to build and use them is a powerful skill in any ML engineer or data scientist's toolkit.

---

## Further Reading

- [YouTube’s Deep Neural Networks for Recommendations](https://research.google/pubs/pub45530/)
- [DLRM: Facebook’s Deep Learning Recommendation Model](https://arxiv.org/abs/1906.00091)
- [TensorFlow Recommenders](https://www.tensorflow.org/recommenders)
- [Hugging Face Embedding Notebooks](https://huggingface.co/blog/recommendations)