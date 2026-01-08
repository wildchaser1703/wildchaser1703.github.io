---
layout: post
title: "LLM Evaluation Metrics Beyond Accuracy"
date: 2025-01-28 10:00:00 +0530
categories: [Machine Learning, MLOps]
tags: [llm-evaluation, production-ml, ml-systems, evaluation-metrics]
description: A practitioner's guide to evaluating LLMs in production using metrics that actually reflect user impact.
---

# LLM Evaluation Metrics Beyond Accuracy

Accuracy is a lie we tell ourselves to feel better about shipping LLMs to production.

Not that it's wrong. It's just incomplete in ways that will cost you money and users if you're not careful. I shipped an LLM-powered support agent last year that had 89% accuracy on our test set. Felt great. Users hated it.

The model was right 89% of the time but wrong in the most expensive possible ways. It would confidently hallucinate refund policies. Refuse to escalate when it should. Generate responses that were technically accurate but completely missed what the user was asking for.

Accuracy told me the model worked. It didn't tell me the model was useful.

## The Calibration Problem

First thing I started tracking after that disaster was calibration. This is the gap between how confident your model says it is and how often it's actually right.

A well-calibrated model that says it's 80% confident should be correct 80% of the time. Most LLMs are terribly calibrated. They'll output tokens with high probability while being completely wrong.

Measuring this requires bucketing predictions by confidence and calculating accuracy within each bucket. I extract logprobs from the API, average them across the output tokens, and use that as a confidence score. Then I bucket predictions into deciles and measure actual accuracy in each bucket.

Our support agent was outputting refund amounts with 0.95 average logprobs. Felt very sure of itself. Actual accuracy on those predictions was 62%. The model was confidently wrong and we had no way to catch it without manual review.

Started using Expected Calibration Error as a metric. ECE measures the weighted average of the gap between confidence and accuracy across all buckets. Formula is:

ECE = Σ (|confidence_i - accuracy_i| × n_i / N)

Where n_i is the number of samples in bucket i and N is total samples. Lower is better. Well-calibrated model should have ECE under 0.05.

Ours was at 0.33. Completely uncalibrated. Added a calibration layer that adjusts confidence scores based on historical accuracy. Not perfect but it got ECE down to 0.12. More importantly, we could finally trust the model when it said it was unsure and route those cases to humans.

## Semantic Similarity Isn't Enough

Second metric everyone uses is semantic similarity between generated output and reference output. BLEU score, ROUGE score, cosine similarity of embeddings. All measuring roughly the same thing: how close is the generated text to the expected text.

Problem is semantic similarity doesn't capture what actually matters. I can generate text that's semantically similar but factually wrong. Or semantically different but functionally equivalent.

Had a model generating product descriptions. High ROUGE scores, looked great. Descriptions were fluent and on-topic. Also wrong about product specifications 30% of the time. ROUGE doesn't care if you said the battery lasts 8 hours when it actually lasts 10 hours. Semantically those sentences are nearly identical.

Started tracking factual consistency separately. For any claim the model makes about a verifiable fact, check if that fact is true in the source material. This requires another LLM call unfortunately. I use a smaller model to extract claims from the generated text, then verify each claim against the source documents.

Built a simple claim extraction prompt: "List all factual claims in this text as bullet points." Then for each claim: "Is this claim supported by the source material? Answer yes, no, or unknown."

Aggregate this into a factual consistency score: number of supported claims divided by total claims. Our product description model had 0.71 factual consistency despite 0.89 ROUGE-L. That gap is where users lose trust.

## Relevance Over Similarity

Semantic similarity also fails on relevance. A response can be about the right topic but completely miss what the user was asking for.

User asks: "How do I return a defective product?"
Model responds: "Our products are manufactured with the highest quality standards and undergo rigorous testing."

Semantically related to product quality. Completely useless as an answer.

Started using pointwise relevance metrics instead. For each generated response, have a separate model rate relevance on a scale: completely irrelevant, somewhat relevant, mostly relevant, fully relevant. Map these to 0, 0.33, 0.67, 1.0.

Average relevance score across test set gives you a better signal than cosine similarity ever will. Our support agent had 0.84 semantic similarity but only 0.58 relevance. Explained why users kept asking followup questions. The model was talking about the right things but not answering the actual question.

This requires labeled data obviously. We had support agents rate a sample of 500 responses. Takes time but it's worth it. The relevance signal correlates much better with user satisfaction than any automatic metric.

## Latency as a Quality Metric

Nobody thinks of latency as a quality metric but it should be. A perfect response that takes 12 seconds is a bad response.

We tracked p50, p95, and p99 latency separately. Mean latency is useless because it hides the outliers that ruin user experience. Our mean was 2.3 seconds which seemed fine. P99 was 11 seconds which explained why we were getting complaints.

Profiled the slow requests. Most of them were hitting the maximum token generation limit. Model would ramble for 800 tokens when 150 would have done the job. High semantic similarity, high relevance, completely useless because nobody reads past the first paragraph and it took forever to generate.

Added token budgets as a hard constraint. Route classification gets 50 tokens. Question answering gets 200 tokens. Explanation requests get 400 tokens. Enforce this at the API level with max_tokens parameter.

P99 dropped to 3.2 seconds. User satisfaction went up even though we were generating shorter responses. Turns out fast and concise beats slow and comprehensive every time.

Also started tracking time-to-first-token separately. This is how long before the user sees anything start to appear. For streaming responses this matters more than total latency. Users will wait for a response if they see it's coming. They won't wait through three seconds of blank screen.

TTFT is dominated by prompt processing time. We had 2400 token system prompts. That's a lot of prefill compute before you generate a single output token. Cut the prompt to 800 tokens. TTFT dropped from 1.8 seconds to 0.4 seconds. Made the product feel dramatically faster even though total latency barely changed.

## Cost Per Query as Quality

Cost is a quality metric whether you want it to be or not. An expensive model you can't afford to run is a bad model.

Track cost per query broken down by component. Prompt tokens, completion tokens, any additional API calls for retrieval or verification or chain-of-thought steps. Our support agent averaged $0.23 per query. Seemed reasonable until we projected to full user volume. Would have been $180K per month.

Profiled where the cost was going. 40% was in the system prompt which we were sending with every request. Implemented prompt caching. Cost dropped to $0.14 per query. Another 30% was in chain-of-thought reasoning tokens we were generating and then throwing away. The model would "think through" the problem for 200 tokens before generating the actual response.

Useful for quality but the user never sees it. Tried disabling CoT. Quality dropped by 15% measured by relevance. Tried compressing the CoT to 50 tokens. Quality dropped by 3%. Kept the compressed version. Cost per query is now $0.09.

At scale these optimizations are the difference between a viable product and burning investor money. If you're not tracking cost per query broken down by component you're flying blind.

## Consistency Across Rephrases

LLMs are stochastic and that's fine until it's not. User asks the same question two different ways and gets contradictory answers. Destroys trust immediately.

Started testing consistency explicitly. Take queries from our test set and generate paraphrases. "How do I return a defective product?" becomes "What's your return policy for broken items?" or "Can I send back something that doesn't work?"

Generate responses for the original and all paraphrases. Measure consistency by checking if the factual claims are the same. Don't need identical text. Just need the same information.

Our model had 0.67 consistency. Meaning a third of the time it would give different answers to the same question phrased differently. This is with temperature at 0.0. The variation comes from tokenization and context framing.

Fixed it by adding examples of paraphrased queries in the prompt. "Users may ask about returns in various ways. Always provide consistent information regardless of phrasing." Got consistency up to 0.84. Not perfect but good enough that users stopped complaining about contradictory answers.

## Instruction Following Precision

Final metric I track is instruction following. Did the model actually do what I told it to do?

This sounds obvious but LLMs are remarkably creative at ignoring instructions. Especially complex multi-step instructions. I'll say "First extract the user's intent. Then retrieve relevant documents. Then generate a response using only information from those documents."

Model will skip the retrieval step and hallucinate. Or it'll retrieve documents but then ignore them and use training data. Or it'll do everything right on 80% of queries and completely ignore instructions on the other 20%.

Built a rubric for instruction following. Break down the prompt into discrete steps. For each step, manually check if the model followed it on a sample of outputs. Score each step as yes or no. Instruction following score is the percentage of steps followed correctly.

Our agent was at 0.73 instruction following. Most common failure was using information not in the retrieved context. Second most common was wrong output format. We'd specify JSON and get markdown. Or specify a maximum of three bullet points and get seven.

Tightened the prompt structure. Added explicit markers for each step. "Step 1: Extract intent. Output format: single sentence. Step 2: Retrieve documents. Output format: list of document IDs." More rigid but instruction following went to 0.91.

## What Actually Matters

You need a dashboard with all of these. Accuracy alone will lie to you. I look at:

- Calibration (ECE under 0.10)
- Factual consistency (above 0.85)
- Relevance (above 0.75)
- P95 latency (under 2 seconds)
- Cost per query (within budget)
- Consistency across rephrases (above 0.80)
- Instruction following (above 0.85)

When any of these degrades I investigate before accuracy drops enough to notice. Usually catches problems two weeks earlier than accuracy alone would.

The metrics that matter are the ones that measure what users actually care about. Fast, cheap, relevant, factually correct, consistent responses. Accuracy is necessary but not sufficient.

Build evaluation that measures the things that break in production. Because they will break. And when they do you need metrics that tell you why.

---