# Featured project source evidence

Reviewed 2026-11-16. Selected **Documentation RAG Pipeline**, **Agentic Insight Analyst**, and **Retail Demand Forecasting** for distinct retrieval, agent orchestration, and time-series ML emphasis. These contain substantive implementation and test source, with limitations stated in the portfolio data.

## Method and evidence boundary

Public repository metadata, complete recursive file trees, and `commits/main` were read through `https://api.github.com/repos/wildchaser1703/<repo>`. File contents were fetched from `https://raw.githubusercontent.com/wildchaser1703/<repo>/<commit>/<path>` using TLS verification and `curl --max-time 20`. Every file link below pins the exact commit returned by the commit endpoint. Trees were not truncated.

Remote source was read as text/JSON only. No repository code, notebook cells, tests, dependency installations, or model calls were executed. Included tests show intended coverage, not passing results. No deployment, runtime reliability, benchmark, latency, accuracy, or business outcome was independently established. The JSON result fields explicitly describe source-reviewed implementation.

## production-rag-system

Commit: [`65eb63dcc132fa772de9c06215311677bc105b4a`](https://github.com/wildchaser1703/production-rag-system/commit/65eb63dcc132fa772de9c06215311677bc105b4a).

- [README.md](https://github.com/wildchaser1703/production-rag-system/blob/65eb63dcc132fa772de9c06215311677bc105b4a/README.md) — Architecture and evaluation claims; treated as documentation.
- [src/rag_pipeline/ingestion/loader.py](https://github.com/wildchaser1703/production-rag-system/blob/65eb63dcc132fa772de9c06215311677bc105b4a/src/rag_pipeline/ingestion/loader.py) — WebBaseLoader and recursive overlapping text chunks; load failures return an empty list.
- [src/rag_pipeline/retrieval/vector_store.py](https://github.com/wildchaser1703/production-rag-system/blob/65eb63dcc132fa772de9c06215311677bc105b4a/src/rag_pipeline/retrieval/vector_store.py) — OpenAI embeddings, persistent Chroma collection, configurable retrieval with default k=4.
- [src/rag_pipeline/generation/generator.py](https://github.com/wildchaser1703/production-rag-system/blob/65eb63dcc132fa772de9c06215311677bc105b4a/src/rag_pipeline/generation/generator.py) — LCEL retrieval/prompt/model/parser composition. Only page_content enters context; output is a string without citations.
- [src/rag_pipeline/main.py](https://github.com/wildchaser1703/production-rag-system/blob/65eb63dcc132fa772de9c06215311677bc105b4a/src/rag_pipeline/main.py) — Wires ingestion, indexing, retrieval, and generation; does not invoke RAGEvaluator.
- [src/rag_pipeline/evaluation/evaluator.py](https://github.com/wildchaser1703/production-rag-system/blob/65eb63dcc132fa772de9c06215311677bc105b4a/src/rag_pipeline/evaluation/evaluator.py) — RAGAS wrapper for supplied questions, answers, contexts, and ground truths; catches evaluation failures and returns an empty dictionary. Dependency compatibility was not checked.
- [src/rag_pipeline/config.py](https://github.com/wildchaser1703/production-rag-system/blob/65eb63dcc132fa772de9c06215311677bc105b4a/src/rag_pipeline/config.py) — Pydantic settings and default embedding/model identifiers.
- [tests/test_ingestion.py](https://github.com/wildchaser1703/production-rag-system/blob/65eb63dcc132fa772de9c06215311677bc105b4a/tests/test_ingestion.py) — Two tests for chunk splitting and defaults; no retrieval/generation quality tests in the listed tree.
- [.github/workflows/ci.yml](https://github.com/wildchaser1703/production-rag-system/blob/65eb63dcc132fa772de9c06215311677bc105b4a/.github/workflows/ci.yml) — Configured lint, type-check, and pytest jobs; workflow success was not verified.

Copy boundary: title deliberately describes a documentation pipeline; the repository name is not evidence of production deployment. The evaluator is separate from the runner, and the grounding instruction is a prompt rather than an enforced factuality guarantee.

## agentic-insight-analyst

Commit: [`3ed7e28095bfab7f1de102d2807c2c63bb43cfe5`](https://github.com/wildchaser1703/agentic-insight-analyst/commit/3ed7e28095bfab7f1de102d2807c2c63bb43cfe5).

- [README.md](https://github.com/wildchaser1703/agentic-insight-analyst/blob/3ed7e28095bfab7f1de102d2807c2c63bb43cfe5/README.md) — Agent architecture claims and example trace; neither production readiness nor the trace was accepted as measured evidence.
- [src/agent.py](https://github.com/wildchaser1703/agentic-insight-analyst/blob/3ed7e28095bfab7f1de102d2807c2c63bb43cfe5/src/agent.py) — LLM planning and tool selection, registry-name fallback, argument construction, observation, evaluation, and synthesis.
- [src/tools.py](https://github.com/wildchaser1703/agentic-insight-analyst/blob/3ed7e28095bfab7f1de102d2807c2c63bb43cfe5/src/tools.py) — TF-IDF/KMeans clustering, TF-IDF phrases, numeric score summaries, word-frequency summaries, and templated recommendations. No retrieval tool in the registry despite README wording; a registry is not an execution sandbox.
- [src/memory.py](https://github.com/wildchaser1703/agentic-insight-analyst/blob/3ed7e28095bfab7f1de102d2807c2c63bb43cfe5/src/memory.py) — Session-local plans, entries, tool results, insights, serialization, and execution trace.
- [src/evaluator.py](https://github.com/wildchaser1703/agentic-insight-analyst/blob/3ed7e28095bfab7f1de102d2807c2c63bb43cfe5/src/evaluator.py) — Iteration, diminishing-return, and repetition stops. Progress weights insight count, tool diversity, and success rate; success_criteria is not used in the estimate. Output validation checks fields, not factual quality.
- [api/main.py](https://github.com/wildchaser1703/agentic-insight-analyst/blob/3ed7e28095bfab7f1de102d2807c2c63bb43cfe5/api/main.py) — FastAPI routes, process-local jobs, and CSV path handling. raw_text is accepted but not forwarded into agent.run.
- [tests/test_agent_flow.py](https://github.com/wildchaser1703/agentic-insight-analyst/blob/3ed7e28095bfab7f1de102d2807c2c63bb43cfe5/tests/test_agent_flow.py) — Memory, tool, evaluator, parser, tool-composition, and mocked-LLM tests. The mocked run checks result shape, not report accuracy or live-model behavior.

Copy boundary: describe bounded orchestration and inspectable intermediate work. `_call_llm` catches failures and returns `{}`; the enclosing run can still report `success=True` after default synthesis. No claim of proven goal satisfaction, effective recommendations, durable jobs, or production readiness.

## retail-demand-forecasting-system

Commit: [`9fd10d476d6188444efe2b73c74551a5d52505e7`](https://github.com/wildchaser1703/retail-demand-forecasting-system/commit/9fd10d476d6188444efe2b73c74551a5d52505e7).

- [README.md](https://github.com/wildchaser1703/retail-demand-forecasting-system/blob/9fd10d476d6188444efe2b73c74551a5d52505e7/README.md) — Synthetic-data context and reported performance table. All numerical results and production-readiness claims omitted from portfolio copy.
- [scripts/run_pipeline.py](https://github.com/wildchaser1703/retail-demand-forecasting-system/blob/9fd10d476d6188444efe2b73c74551a5d52505e7/scripts/run_pipeline.py) — Validation, chronological split, combined feature creation, baseline/ML training, ensemble, and CSV metrics output.
- [src/data_loader.py](https://github.com/wildchaser1703/retail-demand-forecasting-system/blob/9fd10d476d6188444efe2b73c74551a5d52505e7/src/data_loader.py) — CSV/Parquet loading and date-boundary train/validation/test splits.
- [src/data_validation.py](https://github.com/wildchaser1703/retail-demand-forecasting-system/blob/9fd10d476d6188444efe2b73c74551a5d52505e7/src/data_validation.py) — Schema/type/missing-value/range checks, duplicate-key checks, and statistics.
- [src/feature_engineering.py](https://github.com/wildchaser1703/retail-demand-forecasting-system/blob/9fd10d476d6188444efe2b73c74551a5d52505e7/src/feature_engineering.py) — Calendar, group lag, rolling, difference, and promotion features. Rolling windows include current sales; differences also include current sales.
- [src/ml_models.py](https://github.com/wildchaser1703/retail-demand-forecasting-system/blob/9fd10d476d6188444efe2b73c74551a5d52505e7/src/ml_models.py) — XGBoost, LightGBM, and RandomForest wrappers, feature importance and persistence. TimeSeriesMLForecaster.predict raises NotImplementedError; its rolling features also include the current target.
- [src/evaluation.py](https://github.com/wildchaser1703/retail-demand-forecasting-system/blob/9fd10d476d6188444efe2b73c74551a5d52505e7/src/evaluation.py) — Metric functions, time-series split utilities, model comparison, and rolling forecast evaluation.
- [src/mlflow_utils.py](https://github.com/wildchaser1703/retail-demand-forecasting-system/blob/9fd10d476d6188444efe2b73c74551a5d52505e7/src/mlflow_utils.py) — Explicit logging wrappers and log_forecast_experiment helper; existence does not prove runs were tracked.
- [tests/test_features.py](https://github.com/wildchaser1703/retail-demand-forecasting-system/blob/9fd10d476d6188444efe2b73c74551a5d52505e7/tests/test_features.py) — Feature-column and value checks. The rolling-feature test explicitly expects an unshifted rolling mean, so it does not guard against leakage.
- [tests/test_models.py](https://github.com/wildchaser1703/retail-demand-forecasting-system/blob/9fd10d476d6188444efe2b73c74551a5d52505e7/tests/test_models.py) — Baseline values, lengths, and invalid-input checks; these are not ML accuracy benchmarks.
- [tests/test_evaluation.py](https://github.com/wildchaser1703/retail-demand-forecasting-system/blob/9fd10d476d6188444efe2b73c74551a5d52505e7/tests/test_evaluation.py) — Metric calculations and some edge cases; source reviewed, not run.

Copy boundary: substantial forecasting components justify selection, but benchmark claims are unsuitable. `engineer_features` combines all date splits before generating features, and unshifted rolling/difference features expose the current target. Held-out lags can also use actual earlier held-out sales, which is not a fixed-origin multi-step evaluation. Baselines predict daily totals while ML predicts store rows; their errors are not directly comparable, and their ensemble arrays can differ in length. `--use-mlflow` constructs a tracker without starting/logging training runs. The README's performance table is reported documentation, not independently measured evidence, and is omitted entirely.

## Other candidates

- **IntelliQuery** — commit `d8ad692faa885209e5c13b1065e1c9c0c3b54a89`; [README.md](https://github.com/wildchaser1703/IntelliQuery/blob/d8ad692faa885209e5c13b1065e1c9c0c3b54a89/README.md), [IntelliQuery_Model.ipynb](https://github.com/wildchaser1703/IntelliQuery/blob/d8ad692faa885209e5c13b1065e1c9c0c3b54a89/IntelliQuery_Model.ipynb). Real multimodal notebook work, but less suitable for this engineering shortlist. The main notebook concatenates question and image vectors and trains a dense answer classifier. Its fit call has no held-out validation data; it requires external vector/annotation files. The README credits base implementations. No automated test files appear in the tree. Notebook outputs were not treated as independent evaluation. [Pinned tree](https://api.github.com/repos/wildchaser1703/IntelliQuery/git/trees/d8ad692faa885209e5c13b1065e1c9c0c3b54a89?recursive=1).
- **indic-speech-ondevice** — commit `9c17b948ea7676b3a947768cbdde8f5d20572fdd`; [README.md](https://github.com/wildchaser1703/indic-speech-ondevice/blob/9c17b948ea7676b3a947768cbdde8f5d20572fdd/README.md). In-progress benchmark proposal. The complete tree contains documentation, packaging/lock files, and data/results placeholders, but no speech pipeline implementation or tests. Named components and latency targets are plans, not measured results. [Pinned tree](https://api.github.com/repos/wildchaser1703/indic-speech-ondevice/git/trees/9c17b948ea7676b3a947768cbdde8f5d20572fdd?recursive=1).
- **llm-tokenizer-from-scratch** — commit `8663b926eedfeeaf258470ac91881c00d240a743`; [README.md](https://github.com/wildchaser1703/llm-tokenizer-from-scratch/blob/8663b926eedfeeaf258470ac91881c00d240a743/README.md). The tree contains only README.md, LICENSE, and .gitignore; the README is a title. No tokenizer implementation or tests to feature. [Pinned tree](https://api.github.com/repos/wildchaser1703/llm-tokenizer-from-scratch/git/trees/8663b926eedfeeaf258470ac91881c00d240a743?recursive=1).

## Deliverable

`/workspace/wildchaser1703.github.io/data/engineering-projects.json` contains exactly three objects in the requested schema, with full commit hashes and commit-pinned evidence links. Diagrams summarize implementation paths; they do not imply validated runtime outcomes. Only this evidence file and the project JSON were authored for this task.
