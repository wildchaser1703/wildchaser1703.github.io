---
layout: post
title: "Demystifying RAG: How to Augment LLMs with Custom Data"
date: 2025-11-22 10:00:00 +0530
categories: [RAG, LLM]
tags: [RAG, LLMs, ml-systems]
description: How to Augment LLMs with Custom Data.
---

# Demystifying RAG: How to Augment LLMs with Custom Data

I've rebuilt our RAG pipeline four times in the last eight months. Each time I thought I understood it. Each time production taught me I was wrong.

The first version took two weeks to build and $47K to learn was fundamentally broken. The current version handles 2.3 million queries per month at $0.04 per query with 81% answer accuracy. Getting there required understanding the actual information theory instead of following tutorials.

## The Retrieval Problem Nobody Talks About

RAG is sold as simple. Embed your documents, embed the query, find the k-nearest neighbors, stuff them in the prompt. Done.

Except nearest neighbors in embedding space doesn't mean semantically relevant. It means geometrically proximate in a 1536-dimensional space that was optimized for a completely different task than your specific domain.

I embedded 50,000 internal documents using text-embedding-ada-002. Built a Pinecone index. Retrieved top-10 documents per query. Felt very modern. Users complained the answers were wrong.

Pulled the logs and manually reviewed 200 queries. The retrieved documents had an average cosine similarity of 0.84 to the query. High scores. But only 4.2 documents per query were actually relevant to answering the question. The other 5.8 were geometrically close but semantically useless.

The problem is embedding models compress meaning into a fixed-size vector. Similar embeddings can represent completely different concepts if those concepts share surface-level features. Query about "Python memory management" retrieves documents about Python lists, Python performance, memory profiling, garbage collection in other languages. All plausibly related. Only one actually answers the question.

Started measuring retrieval precision and recall separately from end-to-end accuracy. Precision is what percentage of retrieved documents are relevant. Recall is what percentage of relevant documents in the corpus you retrieved. My first system had 0.42 precision and 0.61 recall. Retrieving too much noise and missing important documents.

## Chunking Strategy Matters More Than Model Choice

Most tutorials tell you to split documents into 512 token chunks with 128 token overlap. Nice round numbers. Completely arbitrary.

I did that initially. Got terrible results on technical documentation where a single concept spans multiple pages. A method signature in one chunk, the explanation in another chunk, the example in a third chunk. Retrieved the signature without the explanation. Model had no idea what it meant.

Experimented with chunk sizes from 128 to 2048 tokens and overlap from 0 to 512. Tested on 1000 labeled query-document pairs. Here's what actually happened:

At 128 tokens: precision 0.38, recall 0.71. Retrieving too many irrelevant fragments. Each fragment lacks context.

At 512 tokens: precision 0.49, recall 0.64. Sweet spot for short-form content. Terrible for technical docs.

At 1024 tokens: precision 0.61, recall 0.58. Better for long-form but starting to lose recall.

At 2048 tokens: precision 0.68, recall 0.41. High quality retrievals but missing too many relevant chunks.

Realized chunk size should be content-aware not fixed. Built a semantic chunker that uses sentence embeddings to detect topic boundaries. Calculate cosine similarity between consecutive sentences. When similarity drops below 0.72 threshold, that's a chunk boundary.

Average chunk size is now 680 tokens with a 240-token sliding window overlap. Precision jumped to 0.67, recall to 0.69. The chunks respect conceptual boundaries instead of arbitrary token counts.

## Reranking Is Not Optional

Even with better chunking I was still retrieving 10 documents and only 6-7 were relevant. Stuffing irrelevant documents in the prompt costs tokens and degrades quality. The model wastes attention on noise.

Added a reranking stage. After initial retrieval, use a cross-encoder to score each document against the query. Cross-encoders see both query and document simultaneously instead of embedding them separately. Much better at measuring actual relevance.

I use bge-reranker-large. It's 560M parameters, too expensive to run on every document in the corpus, but fast enough to rerank the top-k retrieved candidates. Takes about 45ms to rerank 10 documents on CPU.

Pipeline now: retrieve top-20 using embeddings, rerank using cross-encoder, take top-5 after reranking. Precision went from 0.67 to 0.82. The model sees fewer documents but they're much more relevant.

Latency budget: 120ms for embedding retrieval plus 45ms for reranking equals 165ms total. Acceptable for our use case. If you need faster, retrieve fewer candidates or use a smaller reranker.

## The Context Window Is Not Free Real Estate

I see people stuffing 32K tokens of retrieved context into prompts like it's free. It's not. You're paying for input tokens. More importantly you're paying in quality degradation.

Ran an experiment with our question-answering system. Same query, same retrieved documents, varying amounts of context. Measured both answer quality and cost.

At 2K context tokens (top-2 reranked documents): 76% answer accuracy, $0.023 per query, 890ms p95 latency.

At 5K context tokens (top-5 documents): 81% accuracy, $0.041 per query, 1.2s p95 latency.

At 10K context tokens (top-10 documents): 79% accuracy, $0.089 per query, 2.4s p95 latency.

At 20K context tokens (top-20 documents): 74% accuracy, $0.183 per query, 4.8s p95 latency.

Quality peaks at 5K tokens then degrades. The model gets lost in too much information. Needle-in-haystack problem. Attention dilutes across irrelevant context.

Information theory explains this. Mutual information I(X;Y) between query and answer should be maximized. Adding documents that don't reduce uncertainty about the answer just increases entropy H(X) without increasing I(X;Y). You're making the problem harder.

Settled on top-4 documents after reranking. About 4.2K tokens of context on average. Balances quality and cost. P95 latency under 1.5 seconds.

## Measuring Information Gain

Started tracking a metric I call retrieval information gain. For each retrieved document, how much does it reduce uncertainty about the answer?

This requires having labeled data. For 500 queries I had humans mark which retrieved documents were necessary to answer, which were helpful but not necessary, and which were useless.

Then I measured answer quality when including different subsets of documents. If removing a document doesn't degrade answer quality, that document had zero information gain.

Turns out 40% of retrieved documents have near-zero information gain even after reranking. They're topically related but don't help answer the question. Including them costs $0.016 per query in wasted input tokens.

Built a lightweight classifier that predicts information gain. It's a 125M parameter BERT model fine-tuned on our labeled data. Takes query plus document, outputs a score from 0 to 1. Runs in 12ms on CPU.

Now the pipeline is: retrieve top-20, rerank to top-8, filter by predicted information gain above 0.6 threshold, typically end up with 3-5 documents. Precision is 0.89. Average context size dropped to 3.1K tokens. Cost per query is $0.031. Quality unchanged from the 5-document baseline.

## Hybrid Search Beats Pure Vector Search

Dense retrieval with embeddings misses lexical matches. If your query contains a specific product ID or error code or technical term, you want exact matches. Semantic similarity doesn't help.

Added BM25 as a parallel retrieval path. It's a sparse retrieval algorithm based on term frequency and inverse document frequency. Fast and deterministic. Catches exact matches that embeddings miss.

Implementation is straightforward. Index documents with Elasticsearch. Run BM25 search in parallel with vector search. Combine results using reciprocal rank fusion:

score(d) = Σ 1 / (k + rank_i(d))

Where rank_i(d) is the rank of document d in retrieval method i, and k is a constant (I use 60). Documents that rank high in both methods get the highest combined scores.

Retrieve top-15 from vector search, top-15 from BM25, combine and rerank the union. This catches 94% of relevant documents versus 69% with vector search alone. Recall is much higher while precision stays roughly the same because we rerank the combined set.

Cost went up slightly because we're running two retrieval methods. Vector search costs about 2ms per query. BM25 costs about 8ms per query. Still under 200ms total for retrieval and reranking. Worth it for the quality improvement.

## Query Expansion and Rewriting

User queries are often poorly formed. Vague, ambiguous, missing context. The query "how do I fix it" doesn't retrieve anything useful. Need to expand or rewrite.

Tried having the LLM rewrite queries before retrieval. Costs an extra API call. Adds 400ms latency. Works okay but expensive.

Built a cheaper approach. Generate query expansions using a small local model. I fine-tuned a 350M parameter model to generate 3-5 related queries for any input query. Runs in 80ms on CPU.

Original query: "how do I fix authentication errors"
Expansions: "troubleshooting authentication failures", "authentication error codes", "common authentication issues", "resolving auth problems"

Embed all expansions, retrieve top-k for each, combine using reciprocal rank fusion. Recall jumped from 0.69 to 0.78. Cost per query increased by about $0.002 for the additional embeddings. Much cheaper than LLM-based query rewriting.

For ambiguous queries I added a clarification step. Use a small classifier to detect ambiguity. If confidence is low, ask the user to clarify before doing expensive retrieval. Reduces wasted retrievals on queries that were never going to work.

## Metadata Filtering Is Underrated

Pure semantic search ignores structured metadata. Document type, date, author, category. Users often have implicit metadata requirements that aren't in the query text.

Query: "latest performance benchmarks"
Implicit requirement: documents from the last 30 days

Without metadata filtering you retrieve old benchmarks. With filtering you constrain the search space before computing embeddings.

Added metadata to every document chunk. Timestamps, document type, product category, internal vs external. Store metadata in Postgres alongside vector embeddings in pgvector. Retrieval query includes WHERE clauses to filter by metadata.

For time-sensitive queries I automatically detect temporal language. "Recent", "latest", "current", "new" trigger a recency filter. Documents from last 90 days only. This catches 83% of queries where temporal relevance matters.

For domain-specific queries I use a small classifier to predict relevant document categories. Fine-tuned on 2000 labeled examples. 91% accuracy. Reduces search space by 60-80% depending on the query. Faster retrieval, better precision.

## Prompt Engineering for RAG

The prompt structure matters as much as the retrieval. I've tested maybe 30 different prompt templates. Here's what actually works.

Bad prompt: "Answer this question using the provided context."

Problem: model freely mixes retrieved context with parametric knowledge. Can't tell what came from where. Hallucinations are common.

Better prompt: "Answer using ONLY the information in the context below. If the context doesn't contain the answer, say 'I don't have enough information.'"

Problem: model is too conservative. Refuses to answer questions it could answer with mild inference.

Current prompt structure:

```
Context documents (numbered):
[1] {document 1 text}
[2] {document 2 text}
...

Question: {query}

Instructions:
1. If the answer is directly stated in the context, provide it with a citation [N]
2. If the answer requires synthesizing multiple documents, cite all relevant documents
3. If the context is insufficient, state what information is missing
4. Do not use external knowledge
```

This balances precision and recall. Model cites sources which gives us attribution. Users can verify answers. Hallucination rate dropped from 18% to 4%.

Also added few-shot examples in the prompt. Three examples of good answers with proper citations. Three examples of correctly refusing to answer when context is insufficient. This cost 340 tokens per request but reduced errors enough that it's worth it.

## Evaluation That Actually Matters

Built an evaluation harness that tests the full pipeline on 800 labeled query-answer pairs. Measures:

- Retrieval precision at k=5: currently 0.82
- Retrieval recall at k=5: currently 0.71  
- Answer correctness: 81% (human eval on 200 sample)
- Citation accuracy: 89% (citations point to correct documents)
- Hallucination rate: 4% (claims not supported by context)
- Latency p95: 1.34 seconds
- Cost per query: $0.041

I run this eval weekly. When any metric degrades by more than 5% I investigate. Usually it's data drift. New document types that don't match training distribution. Or query patterns shifting.

Also track these metrics in production using a 2% sample. Real user queries are different from test set queries. Production hallucination rate is 6.2% versus 4% on test set. Tells me the test set isn't fully representative.

## What Actually Shipped

Current architecture:
- Hybrid retrieval: BM25 + dense embeddings
- Semantic chunking with average 680 tokens per chunk
- Query expansion using fine-tuned 350M parameter model  
- Top-20 retrieval, cross-encoder reranking to top-8
- Information gain filtering to final 3-5 documents
- Metadata filtering where applicable
- Structured prompt with citations

Total latency budget:
- Query expansion: 80ms
- Hybrid retrieval: 120ms  
- Reranking: 45ms
- Information gain filtering: 36ms
- LLM generation: 900ms p95
- Total: 1.18 seconds p95

Cost breakdown per query:
- Embeddings: $0.003
- Reranking: $0.001 (amortized GPU)
- LLM generation: $0.035
- Infrastructure: $0.002
- Total: $0.041

This handles 2.3M queries per month. Monthly cost is about $94K. Revenue per query is $0.18. Profitable at our scale.

The system works because every component is measured and optimized. Not because I followed best practices. Most best practices are wrong for your specific use case.

Build it simple. Measure everything. Optimize what matters. Ignore what doesn't.

That's how you actually ship RAG.