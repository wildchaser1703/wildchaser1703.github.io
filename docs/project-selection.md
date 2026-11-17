# Project publication — 17 November 2026

The current selection contains five projects: Agentic Insight Analyst, Depth Estimation & Segmentation, Documentation RAG Pipeline, Retail Demand Forecasting and IntelliQuery. The homepage leads with the agent and attributed depth study. Earlier experiments remain in the archive.

The reviewed source revisions are published on each repository's `fix/portfolio-readiness` branch. Website source/evidence links are pinned to those exact commits so the documented behavior matches the linked code. Default project branches have not been merged as part of this website release.

The isolated offline suites pass: agent 35, depth 7, RAG 10, forecasting 36 and IntelliQuery 5 (93 total). These test execution and evaluation plumbing, not live model quality. Synthetic demonstrations are not presented as real-data benchmark scores. Real VQA performance, historical depth metrics and live agent/RAG answer quality remain unverified.

The directory layout, typography, colors and interaction design are retained. Project descriptions, stacks, limitations, source links and existing SVG system sketches now match the reviewed implementations. IntelliQuery is presented as a multimodal baseline with explicit data/evaluation limits.

## Original pre-remediation assessment

The following is historical evidence from before the fixes; it does not describe the newly linked source revisions.

# Project readiness review — 17 November 2026

## Decision

Lead with **Agentic Insight Analyst** as an agent prototype and **Depth Estimation & Segmentation** as an attributed research study. Keep **Documentation RAG** as a supporting prototype and **Retail Demand Forecasting** visible as an experimental workbench, respecting the request to retain it. Move **IntelliQuery** to the archive until its evaluation and setup improve. None has sufficient evidence here to claim production readiness or independently verified model quality.

| Project | Verified evidence | Readiness and placement |
| --- | --- | --- |
| Agentic Insight Analyst | All 23 supplied offline tests pass. Tool registry, bounded loop, memory and API exist. A separate failure-injection check forced every LLM call to fail and still received `success=True`. | Homepage / main directory, explicitly **Agent prototype**. Fix failure propagation, raw-text handling and persistence before presenting a dependable service. |
| Depth Estimation & Segmentation | Source includes paired-input encoders, separate depth/mask heads, loss implementations, notebooks, curves and qualitative results. Setup docs explicitly attribute the included library to TensorNet v0.9. | Homepage / main directory, explicitly **Research study**, with TensorNet attribution. The documented requirements file is absent and code imports an unpackaged `tensornet` path. Not certified runnable; training/results were not reproduced. |
| Documentation RAG Pipeline | Both supplied ingestion tests pass. Chunking, Chroma retrieval and generation are implemented; RAGAS is a separate wrapper. | Supporting main-directory **RAG prototype**. Add reproducible dependencies, retrieval/answer evaluations, failure tests and citations before claiming reliable or production-quality answers. These two tests do not exercise a live model or vector database. |
| Retail Demand Forecasting | All 29 supplied tests pass in the audit environment; coverage reports 27% overall. A counterfactual probe changes only the current target from 30 to 3000: its rolling input changes from 25 to 1510. | Keep in main directory as **Experimental workbench**, not a homepage flagship or an accuracy showcase. Fix leakage, daily/store granularity alignment and multi-step forecasting, then rerun temporal evaluation. Passing existing unit tests does not remove these issues. |
| IntelliQuery | Source contains image and BERT feature notebooks and a 500-class answer classifier. The main notebook trains without held-out validation; its saved training accuracy is not generalization evidence. It relies on external feature files and older Keras APIs, and cites upstream VQA demos. | **Archive** as a notebook experiment. Needs an environment lock/setup, accessible artifacts, correct held-out evaluation and explicit baseline attribution before promotion. No model run was attempted. |

## Evidence and scope

Reviewed public repository heads:

- Agent: `3ed7e28095bfab7f1de102d2807c2c63bb43cfe5`
- Depth: `e12c803f451378d65f0accad3f761b597303826d`
- RAG: `65eb63dcc132fa772de9c06215311677bc105b4a`
- Forecasting: `9fd10d476d6188444efe2b73c74551a5d52505e7`
- IntelliQuery: `d8ad692faa885209e5c13b1065e1c9c0c3b54a89`

Commit-pinned source links are in `data/engineering-projects.json`. Depth attribution/setup: `Depth-Estimation/docs/project_setup.md`; package import: `model/base_model.py`. Agent failure handling: `src/agent.py` (`_call_llm` fallback and final success flag). Forecast leakage: `src/feature_engineering.py` (unshifted rolling/difference features). VQA evaluation: `IntelliQuery_Model.ipynb` (`model.fit(x, y, ...)` without validation).

The tests ran against unmodified source checkouts in a separate Python 3.12 audit venv with inherited scientific packages and the dependencies needed for the offline suites. This was **not** a clean install of each complete pinned dependency manifest. Notable versions: pytest 9.1.1, pandas 2.2.3, NumPy 1.26.4, scikit-learn 1.8.0, OpenAI client 1.109.1, LangChain community 0.0.38/core 0.1.53. Network connections were blocked during pytest; no paid APIs, remote model calls, pretrained pickle loading or training jobs were used. No independent benchmark scores are claimed.

Audit artifacts live outside the site repository in `/workspace/project-readiness/`: test logs, a small counterfactual leakage probe, failure-injection log and exact environment versions. No project implementation was modified to make tests pass. Website tests validate presentation and links, not ML correctness.

## Display changes

The homepage leads with the agent prototype and the TensorNet-based depth study. The project index shows four clearly labelled entries; IntelliQuery and the earlier experiments remain accessible in the archive. Forecasting stays visible as requested. Existing routes and published articles are retained. Readiness labels and source-backed limitations replace any implied blanket endorsement of all five as finished systems.

## Future promotion

Quant Fertility and Indic Speech On-device are promising LLM-infrastructure directions but currently have setup scaffolds rather than implemented benchmarks/results. Engineering Gate 1 and LLM Tokenizer from Scratch were README-only in the earlier inventory; Chess Coach was empty. These should earn promotion through code, reproducible measurements and evaluation evidence rather than their titles.
