# Design notes

The decisions behind the code, each with the evidence that settled it, and the known limitations.
Measurements were taken on the five filings in `data/` and on the Groq free tier.

## Preprocessing (`src/virallens/common/`)

pypdf's text carries page furniture that would otherwise end up in every embedding. Cleaning removes
what the extracted text actually contained:

| Noise | Rule | Evidence for the threshold |
|---|---|---|
| ECF stamps, page numbers, running heads (`Opinion of the Court`) | An edge line (first or last 3 of a page) recurring on ≥ 3 pages and ≥ 25% of a filing's pages, digits normalised | Macquarie's running heads recur on 5+ of 11 pages (threshold 3); the most frequent non-furniture edge line anywhere recurs on 5 of 55 pages (threshold 14) |
| Tables of contents and authorities | A page whose lines are ≥ 30% dot leaders | Index pages are 45–75% leader lines; body pages have none |
| Underline rules above running heads | Dropped before the edge check | They pushed `Opinion of the Court` out of the 3-line window |
| Words broken across lines (≈ 700 in the two briefs) | Rejoined; the hyphen is kept only if the filing spells the word hyphenated elsewhere and never solid, or the word is already a compound | `risk-factor` keeps its hyphen in all 47 occurrences; `sec-tion` becomes `section` |
| `10b–5` / `10b-5` | An en dash between word characters becomes a hyphen | The same term appeared 40 times one way and 106 the other |

**Chunking is structure-aware.** Filings are split first at their headings — ALL-CAPS, Roman
numeral, and Title Case point headings — and labelled with the heading path
(`ARGUMENT > C. The Ninth Circuit's Approach Is Erroneous > 1. …`). Each section is then split
recursively at paragraph, sentence and clause boundaries (≤ 1000 characters, 150 overlap), and every
chunk keeps its page range. The petitioners' ARGUMENT, one 44,827-character section before point
headings were detected, becomes eight, the largest with 12,213 characters of body text. Result:
370 chunks, a median of 175 word-piece tokens (at most 400).

| Threshold | Evidence |
|---|---|
| A "heading" with < 64 characters of body is a fragment, folded into the previous section | Name-only signature lines and caption fragments carry ≤ 58; the shortest real sections (the briefs' CONCLUSIONs) 80–81. Longer signature and address blocks (67–126) overlap that range and remain small sections |
| Lettered point headings: ≥ 65% of words of 4+ letters capitalised | Real headings score ≥ 0.67; sentence-case paragraph openings ≤ 0.60. A miss only mislabels a section; it never drops text |

## Retrieval (`src/virallens/rag/retrieval.py`)

Hybrid BM25 + dense (`BAAI/bge-small-en-v1.5`, fastembed/ONNX), fused by Reciprocal Rank Fusion
(k = 60). Each chunk is indexed with its filing title and section path prepended. BM25 proposes only
chunks that contain a query term: zero-score ties would otherwise be fused as candidates.

On the 15 evaluation questions (`python main.py eval-retrieval`, k = 6):

| Retriever | hit@1 | hit@6 | MRR |
|---|---|---|---|
| BM25 | 0.67 | 0.87 | 0.767 |
| Dense | 0.47 | 0.87 | 0.624 |
| **Hybrid** | **0.73** | **0.93** | **0.800** |

Each single retriever misses a question the other finds (`us-requested-judgment` for BM25,
`fb-question-presented` for dense). The index manifest records a fingerprint of the PDFs, the
chunking parameters and the embedding model; an index built from anything else is refused.

## Models and the call policy (`src/virallens/rag/llm.py`)

**Measured limits** (response headers and 429 responses on this account): 8,000 tokens/min,
200,000 tokens/day and 1,000 requests/day per Groq model, plus 30 requests/min per Groq's
documentation; OpenRouter free models allow 20 requests/min and 50/day.
Because each Groq model is metered separately, the roles are spread over three models.

**Model per role**, chosen by a comparison on the same prompts and inputs (120 calls per model:
router and grader on the 15 questions, verifier on 15 genuine answers and 15 deliberately mismatched
ones, two trials each):

| Model | Invalid outputs | Router | Grader keeps an expected passage | Verifier accepts genuine | Verifier rejects mismatched |
|---|---|---|---|---|---|
| **qwen/qwen3.8-27b** | **0 / 120** | **30 / 30** | 28 / 30 | 28 / 30 | 30 / 30 |
| openai/gpt-oss-120b | 1 / 120 | 28 / 30 | 28 / 30 | 27 / 30 | 30 / 30 |
| openai/gpt-oss-20b | 2 / 120 | 30 / 30 | 28 / 30 | 28 / 30 | 30 / 30 |

Qwen runs the classification roles (router, grader, rewriter, verifier); gpt-oss-120b answers and
plans research; gpt-oss-20b runs the research analysts (deep agents need tool calling, which was
validated on gpt-oss). Groq lists qwen3.8-27b as a preview model, so the model ids are settings.

**Structured output uses strict JSON-schema mode**, so the provider enforces the schema. With the
looser modes, gpt-oss sometimes returned the JSON Schema itself instead of an instance, or (with
function calling) no tool call at all.

**Call policy** — one function, `call_with_policy`, shared by graph nodes and deep agents:

| Failure | Handling |
|---|---|
| Invalid model output (`json_validate_failed`, `tool_use_failed`) | Retry once on Groq |
| Rate limit (429) or server error (5xx) | The Groq SDK retries twice, waiting the reported `retry-after` when it is at most 60 s, so per-minute limits are absorbed on Groq |
| Still failing after those retries, rejected key (401), unreachable | Move to OpenRouter |
| Invalid output that persists after the retry | Move to OpenRouter |
| Any other error | Raise |
| No OpenRouter key | Errors that would move to OpenRouter raise instead |

Invalid output was observed in the end-to-end evaluation on both gpt-oss models: an empty
generation under strict JSON, and a call to `repo_browser.search`, a tool gpt-oss was trained with
that does not exist here. Repeating the request succeeded.

Graph nodes cannot use LangChain's `with_fallbacks` (it filters by exception type only, and an
invalid-output 400 is the same type as any other 400), and deep agents cannot use it either
(`create_deep_agent` binds tools to its model). LangChain's `ModelFallbackMiddleware` falls back on
every exception, so the policy runs as a small middleware instead.

**The fallback was exercised against the live API** by running `main.py ask` with an invalid Groq
key, so that every Groq call failed with 401 and moved to OpenRouter (`nemotron-3-super-120b:free`):
a direct question was answered correctly and verified (3,728 tokens, 29 s); a research question ran
the planner and analysts end to end on OpenRouter, with tool calls (17 passages, 50,830 tokens, about
4.5 minutes), and was returned `NOT VERIFIED` with its unsupported claim listed.

## LangGraph flow (`src/virallens/rag/graph.py`)

- **Routing.** The router defaults to `direct`; `research`, several times costlier, is reserved for
  questions that ask to compare parties or connect cases. The router prompt was written against the
  same 15 questions it is evaluated on, so its accuracy there is not a held-out figure.
- **Verification** runs a code check before the model: the answer must cite something, and only
  passages it was given. The model verifier then sees only the cited passages.
- **Revision** gives the answerer its draft and the unsupported claims, with the draft's cited
  passages first and at most 12 passages (about 3k tokens), since the research path can collect
  dozens. An answer still unsupported after one revision is returned flagged with its claims.

## End-to-end evaluation

`python main.py eval-e2e --out eval/results.json`, Groq free tier, no OpenRouter key:

| | |
|---|---|
| Answered | 15 / 15 (no failures) |
| Route matches the label | 15 / 15 |
| Verified by the citation verifier | 15 / 15 |
| Cites a passage on an expected page | 14 / 15 |
| Tokens | 89,669 in total: 2.6k–5.3k per direct question, 17.0k and 25.4k for the two research questions |

`eval/results.json` records the passages each answer cites. The miss is the research question
comparing the two Facebook briefs: it cites their argument pages (United States pp. 18, 24, 25–26;
petitioners pp. 22–23, 37, 49) rather than the narrower pages listed as expected, which were left
unchanged rather than widened to fit the output. The other research question, on how *Macquarie*
bears on the Facebook case, counts as a hit but cites only *Macquarie* (pp. 4, 11), not the
briefs — the comparison is one-sided.

## DeepAgents research (`src/virallens/rag/research.py`)

- `create_deep_agent` adds its own `general-purpose` sub-agent (no call limit, free to answer from
  model memory) unless a spec takes that name; the filing analyst takes it, so the planner's only
  delegate is the analyst.
- Only the analyst has `search_filings`, so every passage behind the research answer is recorded and
  the verifier checks the answer against exactly those passages.
- Planning uses `TodoListMiddleware` (deepagents 0.7.28 does not add it by default). The filesystem
  tools are cut to `read_file`, the minimum deepagents accepts.
- `ModelCallLimitMiddleware` caps the planner at 14 model calls and each analyst at 6, and raises.

## Known limitations

- **RRF can demote a confident single-retriever hit.** For the terse query `Rule 56(f) affidavit
  additional discovery`, BM25 ranks *Last Atlantis* first but dense search reads `Rule 56(f)` as
  near `Rule 10b-5`; fused, *Last Atlantis* lands at rank 5 — inside the 6 passages the answerer
  sees. The natural-language question ranks it 1st.
- `sec-remedies` ("What remedies does the SEC seek…") is a retrieval miss in every mode: the
  complaint says "relief", "enjoining" and "disgorge". The grader → rewriter loop exists for this.
- Footnotes are extracted in reading position and can land mid-sentence (*Last Atlantis* p. 5).
- Section labels are imperfect where headings are adjacent and for signature or address blocks;
  page citations are unaffected. Sentence-case point headings stay in the body text.
- pypdf leaves some intra-word splits (`an y`, `issu er`) that cleaning cannot safely repair.
- deepagents' built-in summarisation middleware makes its own model calls, outside the call policy.
- The research path is not deterministic: the same comparison question was verified in the
  evaluation run and returned `NOT VERIFIED`, with its unsupported claim listed, on another.
- **Groq's daily token cap bounds evaluation.** A full `eval-e2e` run uses about 60k tokens on
  gpt-oss-120b (answers and research planning), so one account supports roughly three runs a day.
- **The OpenRouter free tier is a thin fallback**: free models allow 50 requests a day (under 10
  credits purchased) and one research question can take 25. Per-minute Groq limits are therefore
  waited out on Groq; the fallback covers Groq's daily cap, a rejected key or an outage, for as long
  as its own allowance lasts.
- In Q2, 17 of the 370 chunks exceed all-MiniLM-L6-v2's 256-token input and are truncated.
- 15 hand-written questions are enough to compare retrievers and catch regressions, not to estimate
  accuracy precisely.
