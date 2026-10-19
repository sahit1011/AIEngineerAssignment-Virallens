# Resources referred to

Raw dump of everything this work relied on, split by how it was used.

## Fetched or read during the work

| Resource | Used for |
|---|---|
| `data/*.pdf` (the five filings) | Corpus; every eval question and expected page was located in them |
| https://console.groq.com/docs/rate-limits | Rate-limit headers, 429 behaviour, published per-model limits |
| https://console.groq.com/docs/models | Groq production model list |
| `GET https://api.groq.com/openai/v1/models` (this account, 2026-10-18) | Models actually served: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b` |
| `x-ratelimit-limit-*` response headers (this account) | Measured quota: 8000 tokens/min, 1000 requests/day per model |
| https://openrouter.ai/docs/api-reference/limits | Free-model limits: 20 requests/min |
| `GET https://openrouter.ai/api/v1/models` (2026-10-18) | Free models supporting tool calling and structured outputs |
| PyPI JSON API for each dependency | Current versions before pinning |
| Installed library source (`.venv/lib/python3.12/site-packages/`) | Signatures and behaviour of the library APIs used |
| — `deepagents/graph.py`, `deepagents/middleware/subagents.py`, `deepagents/middleware/filesystem.py` | `create_deep_agent`, the auto-added `general-purpose` sub-agent, middleware merge by name |
| — `langchain/agents/middleware/{model_fallback,model_retry,model_call_limit,types}.py` | Fallback/retry/limit semantics; why a custom fallback middleware was needed |
| — `langchain_groq/chat_models.py`, `groq/_base_client.py`, `groq/_exceptions.py` | Structured-output methods, retry and `retry-after` handling, exception classes |
| — `langchain_openrouter/chat_models.py`, `openrouter/errors/` | OpenRouter structured output and errors |
| — `langchain_text_splitters/base.py`, `character.py` | `start_index` semantics used for page mapping |
| — `fastembed/text/text_embedding.py`, `fastembed/common/utils.py` | Query/passage prefixes, cache location |
| — `sentence_transformers/sentence_transformer/{losses,modules}`, `SentenceTransformer` | MultipleNegativesRankingLoss inputs (labels unused), `Dense`, `preprocess`, `fit` defaults |
| — `rank_bm25.py`, scikit-learn 1.9.1 signatures | BM25Okapi, TF-IDF/SVD, KMeans (`n_init` default), t-SNE, metrics |

## Cited from prior knowledge (not retrieved during the work)

| Work | Used for |
|---|---|
| Cormack, Clarke & Büttcher, "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods", SIGIR 2009 | RRF and its constant k = 60 |
| Lee, Chang & Toutanova, "Latent Retrieval for Weakly Supervised Open Domain Question Answering", ACL 2019 | Inverse Cloze Task pairs |
| Henderson et al., "Efficient Natural Language Response Suggestion for Smart Reply", 2017 | Multiple-negatives ranking loss (in-batch negatives) |
| Reimers & Gurevych, "Sentence-BERT", EMNLP 2019; model card `sentence-transformers/all-MiniLM-L6-v2` | Baseline encoder (384-d, 256-token input) |
| Xiao et al., "C-Pack: Packaged Resources To Advance General Chinese Embedding" (BGE), 2023; `BAAI/bge-small-en-v1.5` | Q1 dense retriever |
| Deerwester et al., "Indexing by Latent Semantic Analysis", JASIS 1990 | LSA baseline |
| Manning, Raghavan & Schütze, *Introduction to Information Retrieval*, 2008, §6.4.1 | Sublinear tf scaling |
| Rousseeuw, "Silhouettes", J. Comput. Appl. Math. 1987; Hubert & Arabie 1985 (ARI); Strehl & Ghosh 2002 (NMI) | Clustering metrics |
| van der Maaten & Hinton, "Visualizing Data using t-SNE", JMLR 2008 | Embedding maps |
