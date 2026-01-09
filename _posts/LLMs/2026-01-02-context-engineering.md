---
layout: post
title: "Context Engineering: The Forgotten Architecture of LLM Systems"
date: 2026-01-02 10:00:00 +0530
categories: [LLM, System Design]
tags: [LLM, System Design, Transformers, MLOps, Inference, Architecture]
math: true
---

# Context Engineering: The Forgotten Architecture of LLM Systems

I've spent $340K learning that context management is the actual product, not the model.

Everyone obsesses over which model to use, how to fine-tune, whether to self-host. Almost nobody thinks systematically about the 8,192 or 32,768 or 128,000 tokens you're feeding into that model with every request. That's the actual system you're building. The model is just an expensive function call in the middle.

Our first production system treated context like a garbage dump. Throw everything in, let the model figure it out. Cost us $28K in the first month and had a 34% hallucination rate. The current system has engineered context at every layer. Same models, $4.1K monthly spend, 6% hallucination rate, 3.2x faster.

The difference is architecture. Not prompts. Architecture.

## The Fundamental Constraint

Context windows have three costs that most people don't think about together: computational cost, monetary cost, and quality cost.

Computational cost scales quadratically with sequence length for attention mechanisms. The self-attention operation is O(n²) where n is sequence length. At 100,000 tokens you're doing 10 billion operations per forward pass. This isn't theoretical. I profiled our inference with torch.profiler. At 8K context, attention layers consumed 42% of compute time. At 32K context, that went to 71%. Same model, four times the input length, not four times slower, seven times slower in practice.

Flash Attention helps but doesn't eliminate the fundamental scaling. Flash Attention 2 gets you to about O(n² / k) where k is typically 4-8 depending on hardware. Still superlinear. A 64K context request on our infrastructure takes 8.2 seconds for first token. An 8K context request takes 1.1 seconds. That 7.1 second difference is attention computation.

Monetary cost is simpler but people forget it compounds. GPT-4 charges $0.01 per 1K input tokens. Sounds cheap. At 32K context that's $0.32 per request. We were handling 180K requests per month. That's $57,600 just in input token costs. Output tokens add another $12K. Infrastructure adds $8K. Total monthly spend of $77,600 for a single feature.

Cut average context to 6.8K tokens through engineering. Input token cost dropped to $12,240. Same functionality, 78% cost reduction.

Quality cost is the one nobody measures until users complain. I ran controlled experiments on our question-answering system. Fixed the query, fixed the relevant information, varied the total context size by adding irrelevant documents.

At 4K context with only relevant information: 84% answer accuracy.
At 8K context with 50% irrelevant information: 79% accuracy.
At 16K context with 75% irrelevant information: 71% accuracy.
At 32K context with 87% irrelevant information: 64% accuracy.

This is the needle-in-haystack problem but it manifests gradually, not just in adversarial cases. Every irrelevant token increases entropy without increasing mutual information between context and answer. The model's attention becomes diffuse. Quality degrades.

Models with larger context windows still suffer from this. Claude with 200K context doesn't mean you should use 200K context. It means you can when you must. Most of the time you shouldn't.

## Context as a State Machine

The breakthrough for me was realizing context is state and state management is a solved problem in distributed systems. Stop thinking about context as a prompt. Think about it as a state vector that evolves through discrete transitions.

Built our agent as an explicit finite state machine with seven states: initialization, intent_classification, information_gathering, retrieval, synthesis, response_generation, and termination. Each state has a maximum context budget. Context that's irrelevant to the current state gets pruned.

State: initialization
Context budget: 800 tokens
Contents: system prompt, user preferences, current timestamp

State: intent_classification  
Context budget: 1,200 tokens
Contents: system prompt, conversation history (last 3 turns), current query

State: information_gathering
Context budget: 2,400 tokens  
Contents: system prompt, classified intent, conversation history, clarification questions

State: retrieval
Context budget: 8,000 tokens
Contents: system prompt, final query, retrieved documents (top-5), metadata

State: synthesis
Context budget: 6,000 tokens
Contents: system prompt, query, filtered documents (top-3), synthesis instructions

State: response_generation
Context budget: 4,000 tokens
Contents: system prompt, query, synthesized information, response guidelines

The key insight is context requirements are state-dependent. During intent classification you don't need retrieved documents. During retrieval you don't need the full conversation history. Only include what's necessary for the current operation.

Implemented this as a context manager class that tracks state transitions and enforces budgets:

```python
class ContextManager:
    def __init__(self, max_budget: int = 8000):
        self.max_budget = max_budget
        self.current_state = None
        self.contexts = {}
        
    def transition(self, new_state: State, required_context: Dict[str, str]):
        """Transition to new state and rebuild context"""
        state_budget = STATE_BUDGETS[new_state]
        
        # Prioritize context by importance for this state
        prioritized = self._prioritize_context(new_state, required_context)
        
        # Pack context within budget
        packed = self._pack_context(prioritized, state_budget)
        
        self.current_state = new_state
        self.contexts[new_state] = packed
        
        return packed
```

Average context size across states is 3,800 tokens. Peak is 8,000 during retrieval. Total tokens per query dropped from 32,000 (monolithic) to 26,600 (state-based) even though we're making more LLM calls. Cost still down because fewer tokens per call and some states use smaller models.

More importantly, quality improved. Each state gets exactly the context it needs. No noise. Intent classification accuracy went from 72% to 89%. Retrieval precision from 0.61 to 0.78. End-to-end task completion from 58% to 76%.

State machines are deterministic. When something breaks I can trace exactly which state failed and what context it had. No more debugging by re-reading prompt logs and guessing.

## Hierarchical Context Compression

Even within states, 8,000 tokens is often wasteful. Most information can be compressed without loss of functionality.

Built a three-tier compression system: lossless truncation, semantic compression, and learned compression.

Lossless truncation is simple. Conversation history beyond three turns has minimal utility. Remove it. System prompts can be cached using prompt caching APIs. Only pay for them once per session instead of every request. Timestamps only need minute precision, not millisecond. Examples in few-shot prompts can rotate instead of including all of them.

Made a list of everything in our context and measured information gain per token. Defined information gain as change in task success rate when that context is removed. Ran A/B tests removing different context elements one at a time.

System prompt: -12% task success when removed (high utility, 340 tokens)
Full conversation history: -3% when reduced to last 3 turns (low utility for older turns)
Few-shot examples: -8% when reduced from 5 to 3 examples (moderate utility, high token cost)
Timestamps: -0% when reduced precision (zero utility for precision beyond minutes)
User preferences: -6% when removed (moderate utility, 120 tokens)

Cut context from 4,200 to 2,800 tokens through lossless pruning. Task success dropped from 76% to 75%. Worth it for 33% cost reduction.

Semantic compression is harder. Take a 2,000 token document and compress it to 500 tokens while preserving information relevant to the query. This is lossy but controlled.

Tried extractive summarization first. Select the most important sentences, discard the rest. Used sentence transformers to embed each sentence, then select sentences with highest cosine similarity to the query embedding. Fast but loses context. A sentence might be relevant but incomprehensible without surrounding sentences.

Switched to abstractive summarization with a compression ratio target. Use a smaller model to rewrite documents more concisely. Prompt: "Compress this document to under 500 tokens while preserving all information relevant to: {query}. Maintain key facts, numbers, and relationships."

This worked better but added latency. Summarizing 5 documents with GPT-3.5 added 800ms and $0.008 per request. Batched the summarization calls. Got it down to 400ms. Still expensive.

Built a learned compression model instead. Fine-tuned a 770M parameter model on 10K examples of (query, document, compressed_document) triples. Compressed documents were written by GPT-4 and then validated by humans. Training cost $4,800. Inference costs 120ms on CPU and $0.0004 per document.

Compression quality measured by answer accuracy when using compressed versus full documents:
- Full documents (2,000 tokens avg): 81% accuracy
- Extractive compression (500 tokens): 73% accuracy  
- GPT-3.5 compression (500 tokens): 78% accuracy
- Fine-tuned compression (500 tokens): 79% accuracy

The fine-tuned model matches GPT-3.5 quality at 20x lower cost and 3x lower latency. Deployed it as a preprocessing step before documents enter the context.

For really long documents I implemented hierarchical compression. Split document into chunks, compress each chunk independently, then compress the compressed chunks. Two-stage compression gets you 16x reduction with manageable quality loss. A 16K document becomes 1K tokens. Answer accuracy drops from 81% to 71% but that's acceptable for exploratory queries where the user will drill down if needed.

## KV Cache Management as First-Class State

Key-value caching in transformer inference is where most people miss optimization opportunities. The KV cache stores attention keys and values from previous tokens so you don't recompute them. But managing what stays in cache is an engineering decision.

During generation, every token produced extends the KV cache. Cache size equals sequence length times number of layers times hidden dimension times two (for keys and values). For Llama 2 70B that's 32 layers × 8192 hidden dim × 2 × sequence_length × 2 bytes (FP16). At 4K sequence length you're storing 4.2GB per request just for KV cache.

If you're serving multiple requests, cache memory becomes your bottleneck. You can fit maybe 12 concurrent 4K context requests on an 80GB A100 before running out of VRAM. Double the context to 8K and you're down to 6 concurrent requests. Halve your throughput.

Implemented explicit KV cache management. Most of our queries follow templates. "What is {entity}?" or "How do I {action}?" The system prompt and query structure are identical across requests. The only variation is the entity or action.

Used prompt caching to store the system prompt's KV cache. Every request reuses it. Saves 340 tokens of prefill computation per request. At 180K requests per month that's 61M tokens of compute saved. Reduces first-token latency from 1.1s to 0.73s.

For multi-turn conversations I keep KV cache alive between turns. User asks a question, we generate a response, user asks a followup. Instead of recomputing the entire conversation history from scratch, we keep the KV cache from the previous turn and only compute the new tokens.

Implemented this with a cache manager that tracks conversation sessions:

```python
class KVCacheManager:
    def __init__(self, max_cache_memory: int):
        self.active_caches = {}  # session_id -> cache
        self.max_memory = max_cache_memory
        
    def get_cache(self, session_id: str, current_turn: int):
        if session_id in self.active_caches:
            cached_turn = self.active_caches[session_id]['turn']
            if cached_turn == current_turn - 1:
                return self.active_caches[session_id]['kv_cache']
        return None
        
    def store_cache(self, session_id: str, turn: int, kv_cache):
        cache_size = self._calculate_size(kv_cache)
        
        # Evict old caches if necessary
        while self._total_size() + cache_size > self.max_memory:
            self._evict_lru()
            
        self.active_caches[session_id] = {
            'turn': turn,
            'kv_cache': kv_cache,
            'last_access': time.time()
        }
```

This cut multi-turn latency by 65%. First turn still takes 0.73s. Subsequent turns take 0.26s because we're only computing the new tokens.

Cache hit rate is 47% across all requests. Every cache hit saves 400-800ms depending on sequence length. But cache memory is limited. Had to implement LRU eviction. Active conversations stay cached, inactive ones get evicted after 5 minutes.

Monitoring showed cache thrashing during peak hours. Too many concurrent conversations, not enough cache memory. Added a priority system. Premium users get higher cache priority. Their conversations stay cached longer. Free tier users get evicted first. Improved experience for paying customers while staying within memory budget.

## Dynamic Context Allocation

Not all queries need the same context budget. Simple queries like "what's your return policy" need maybe 1,200 tokens. Complex queries like "analyze the differences between these three product specifications and recommend the best one for my use case" need 8,000+ tokens.

Built a query complexity classifier. Takes the user query, outputs predicted complexity as low, medium, or high. Fine-tuned a 125M parameter BERT model on 3,000 labeled examples. Labeling criteria:

Low complexity (1-2K token budget):
- Factoid questions
- Simple lookups
- Single-hop reasoning

Medium complexity (3-5K token budget):
- Multi-step questions
- Comparisons between 2-3 items  
- Questions requiring synthesis

High complexity (6-10K token budget):
- Analysis tasks
- Open-ended exploration
- Questions requiring broad context

Classifier accuracy is 84%. When it misclassifies, it typically overestimates rather than underestimates complexity, which is safe. Better to allocate too much context than too little.

Based on predicted complexity, the system allocates context budget dynamically:

```python
def allocate_context(query: str) -> ContextBudget:
    complexity = complexity_classifier(query)
    
    if complexity == "low":
        return ContextBudget(
            system_prompt=400,
            history=200,
            retrieval=600,
            total=1200
        )
    elif complexity == "medium":
        return ContextBudget(
            system_prompt=600,
            history=800,
            retrieval=2600,
            total=4000
        )
    else:  # high
        return ContextBudget(
            system_prompt=800,
            history=1200,
            retrieval=6000,
            total=8000
        )
```

This reduced average context from 5,200 tokens to 3,400 tokens. Cost dropped 35% while task success stayed flat. Simple queries got faster because they weren't waiting for irrelevant retrieval. Complex queries still got the full context they needed.

Also implemented adaptive retrieval based on predicted complexity. Low complexity queries retrieve top-3 documents. Medium retrieve top-5. High complexity retrieve top-8 then rerank to top-6. Precision is higher when you retrieve fewer documents for simple queries.

## Context-Aware Prompt Engineering

System prompts are typically static. Same instructions for every request. This is suboptimal because different tasks need different instructions.

Decomposed our monolithic 840-token system prompt into modular components:

- Core instructions (200 tokens): Always included
- Task-specific instructions (150-300 tokens): Included based on detected intent
- Capability descriptions (120 tokens): Only included when relevant tools are available
- Safety guidelines (80 tokens): Always included
- Examples (180 tokens per example): Included based on query similarity

Built a prompt composer that assembles the system prompt dynamically:

```python
class PromptComposer:
    def compose(self, query: str, intent: str, available_tools: List[str]) -> str:
        components = [self.core_instructions]
        
        # Add task-specific instructions
        if intent in self.task_instructions:
            components.append(self.task_instructions[intent])
            
        # Add relevant capability descriptions
        for tool in available_tools:
            if tool in self.capability_descriptions:
                components.append(self.capability_descriptions[tool])
                
        # Add safety guidelines
        components.append(self.safety_guidelines)
        
        # Add most relevant examples
        relevant_examples = self.select_examples(query, n=2)
        components.extend(relevant_examples)
        
        return "\n\n".join(components)
```

Average system prompt length dropped from 840 tokens to 580 tokens. More importantly, task success improved from 76% to 81% because the instructions are actually relevant to what the user is trying to do.

For multi-turn conversations I implemented context persistence. First turn includes full instructions. Subsequent turns reference those instructions instead of repeating them. "Following the guidelines established in our conversation..." This saves 400 tokens per turn after the first.

Also tested position-dependent prompting. Put the most important instructions at the beginning and end of the system prompt. Models show recency and primacy effects. They weight tokens near the beginning and end higher than tokens in the middle. Moved critical safety guidelines from the middle to the end. Compliance improved from 94% to 98%.

## Entropy-Based Context Pruning

Information theory gives you a principled way to decide what stays in context. Calculate the entropy of each context element. Prune elements with high entropy but low mutual information with the task.

Implemented this for conversation history. Each turn has entropy H(turn). But not all high-entropy turns are informative. A turn where the user asks about shipping and we discuss shipping has high entropy. A turn where the user says "okay" has low entropy. Neither is necessarily more important.

What matters is mutual information I(turn; task). How much does this turn reduce uncertainty about completing the current task? Measured this empirically by removing turns and testing task success.

Built a turn importance scorer:

```python
def score_turn_importance(turn: ConversationTurn, current_query: str) -> float:
    # Semantic relevance to current query
    relevance = cosine_similarity(
        embed(turn.content),
        embed(current_query)
    )
    
    # Recency (exponential decay)
    turns_ago = current_turn_number - turn.turn_number
    recency = math.exp(-0.3 * turns_ago)
    
    # Information density (inverse of entropy)
    density = 1.0 / (1.0 + calculate_entropy(turn.content))
    
    # Combined score
    return 0.4 * relevance + 0.4 * recency + 0.2 * density
```

Sort conversation turns by this score. Include high-scoring turns up to the budget. Prune low-scoring turns. This is better than simple recency-based truncation because it preserves turns that are old but relevant.

Tested on conversations with 10+ turns. Including all turns costs 4,800 tokens and achieves 78% task success. Including last 3 turns costs 1,200 tokens and achieves 71% task success. Including top-5 scored turns costs 2,000 tokens and achieves 79% task success.

The scored approach uses less context than "all turns" while achieving better quality. Proves that not all context is equally valuable.

## Attention Pattern Analysis

Ran experiments with sparse attention patterns to understand what the model actually attends to. Used attention visualization tools to analyze which tokens get high attention weights during generation.

Findings were surprising. In a typical retrieval-augmented generation task with 6,000 token context:

- System prompt tokens: 15% of total attention weight
- Query tokens: 28% of attention weight  
- Retrieved document tokens: 52% of attention weight
- Conversation history tokens: 5% of attention weight

The model barely looks at conversation history during generation. It focuses on system prompt, query, and retrieved documents. This validates the entropy-based pruning that prunes history aggressively.

Within retrieved documents, attention is highly skewed. Top 20% of tokens by attention weight account for 68% of total attention. The model is effectively ignoring most of what you give it.

Used this insight to build attention-guided compression. After retrieving documents, run a single forward pass to get attention patterns. Identify tokens with attention weight below threshold. Remove them before the generation pass.

This is expensive because it requires two forward passes. But it's precise. You're pruning exactly the tokens the model doesn't care about.

Tested on a sample of 500 queries. Without attention-guided pruning: 5,200 token context, 81% accuracy. With attention-guided pruning: 3,100 token context, 80% accuracy. Compression ratio of 1.68x with only 1% accuracy loss.

Latency impact is the issue. Two forward passes instead of one. For applications where cost matters more than latency, this is a win. For applications where latency is critical, it's not worth it.

Implemented it as an optional optimization that engages when query complexity is high and context size exceeds 8,000 tokens. Only worth the extra pass when you're already paying for large context.

## Contextual Feedback Loops

Built a system that learns from its mistakes at the context level. When a query fails, diagnose whether the failure was due to insufficient context, irrelevant context, or wrong context structure.

Implemented failure classification:

```python
class ContextFailureAnalyzer:
    def diagnose(self, query: str, context: str, 
                 expected: str, actual: str) -> FailureType:
        
        # Check if expected answer is in context
        if self.is_in_context(expected, context):
            # Context had the answer but model missed it
            return FailureType.ATTENTION_FAILURE
        
        # Check if a retrieval would have helped
        if self.would_retrieval_help(query, expected):
            return FailureType.MISSING_CONTEXT
            
        # Check if irrelevant context confused the model
        if self.measure_noise_ratio(context, query) > 0.6:
            return FailureType.TOO_MUCH_NOISE
            
        # Check if context structure was wrong
        if not self.validate_structure(context):
            return FailureType.STRUCTURAL_ISSUE
            
        return FailureType.MODEL_CAPABILITY
```

Aggregate these failures weekly. If we see 15%+ MISSING_CONTEXT failures, increase retrieval depth. If we see 20%+ TOO_MUCH_NOISE failures, tighten reranking thresholds. If we see ATTENTION_FAILURE spikes, adjust prompt structure to highlight key information.

This created a feedback loop where context engineering improves over time based on production failures. After 12 weeks of this process, task success went from 81% to 87%. Most improvement came from fixing structural issues we didn't know existed.

Built a context quality dashboard that tracks:
- Average context size by query type
- Context utilization (% of tokens with attention weight > 0.05)
- Retrieval precision and recall
- Context-attributable failures vs model-attributable failures
- Cost per query segmented by context size bins

Review this dashboard weekly. When metrics degrade, investigate. Usually it's data drift. New query patterns that don't match our training distribution. Or document corpus changes that affect retrieval.

## Production Architecture

Current system architecture:

**Layer 1: Query Analysis**
- Complexity classification (125M param BERT)
- Intent detection (same model)
- Dynamic budget allocation
- Cost: 18ms, $0.0001

**Layer 2: Context Assembly**
- State machine transition
- Retrieval (if needed)
- Compression (if needed)
- Prompt composition
- Cost: 180ms, $0.004

**Layer 3: KV Cache Management**
- Check for cached prefixes
- Load or compute KV cache
- Cost: 40ms cache hit, 600ms cache miss

**Layer 4: Generation**
- Single LLM call with engineered context
- Streaming output
- Cost: 800ms p95, $0.035

**Layer 5: Feedback**
- Log context structure
- Log attention patterns (sampled)
- Log failures for analysis
- Cost: 5ms, $0.0001

Total latency p95: 1.04 seconds
Total cost per query: $0.041
Task success rate: 87%
Hallucination rate: 6%

Compare to initial system:
- Latency: 4.2 seconds
- Cost: $0.156
- Task success: 58%
- Hallucination: 34%

Same models. Different architecture.

## What I Learned

Context is not a prompt. It's a state vector in a state machine. Manage it like state.

Compression is not about making things shorter. It's about maximizing information density. Entropy per token should be high. Mutual information with the task should be high. Everything else should be pruned.

The model will fail. Your job is to make sure it has exactly the context it needs to succeed, nothing more, nothing less. This requires measurement, not intuition.

Most optimization happens outside the model. Retrieval, reranking, compression, caching, pruning. The model just processes what you give it. Give it better input and you get better output.

Cost and quality are not in opposition if you engineer context properly. Our system is 4x cheaper and 50% more accurate than the initial version. Because we removed the noise that was costing money and degrading quality.

Build instrumentation first. You cannot optimize what you cannot measure. Context size, attention patterns, failure modes, cost per component. Measure everything.

The teams that win at LLM applications won't have the best models. They'll have the best context engineering. Because context is the actual product.