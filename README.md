# Automated Question–Answer Generation with Mistral-7B

A notebook-based NLP project that generates question–answer pairs from news articles and automobile reviews using **Mistral-7B**, **LangChain**, **Sentence Transformer embeddings** and **FAISS**.

Unlike an interactive chatbot, this project creates questions automatically, answers them using retrieved source passages, and exports the pairs to a CSV file.

## Workflow

1. Read news and/or review CSV files.
2. Normalize whitespace, remove missing and duplicate text, and filter rows shorter than 10 words.
3. Combine sources into a consistent `text` column.
4. Split source documents into chunks.
5. Generate and refine factual questions using Mistral.
6. Remove duplicate questions, statements and headings.
7. Retrieve relevant passages through FAISS and generate source-based answers using the same model instance.
8. Export `Question` and `Answer` columns to `output/QA.csv`.

## Technology

| Component | Implementation |
|---|---|
| Language model | `TheBloke/Mistral-7B-Instruct-v0.1-GGUF` |
| Quantized model file | `mistral-7b-instruct-v0.1.Q4_K_M.gguf` |
| Inference wrapper | CTransformers |
| Question generation | LangChain Classic refine chain |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` |
| Retrieval | FAISS; 2 chunks per question by default |
| Answer generation | LangChain Classic RetrievalQA |
| Data processing | Pandas |
| Interface | Jupyter notebook |

The updated imports use LangChain's split packages. The embedding wrapper is `HuggingFaceEmbeddings`, matching the general-purpose Sentence Transformer model.

## Setup

Use a fresh Python environment compatible with the listed dependencies; Python 3.11 is a suggested starting point. Exact versions have not been locked by a full model run.

```bash
git clone https://github.com/komal25kwah/QA-Generation.git
cd QA-Generation
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

Open **Mistral 7B.ipynb** and run cells in order. Keep `qa_pipeline.py` beside the notebook.

The notebook defaults to the small **fictional sample inputs** in `examples/`. Replace the paths with your own CSVs for real experiments. Initial execution downloads the quantized model and embedding model; allow several GB of disk/memory capacity. CPU inference can be slow. FAISS uses its CPU package by default, so a GPU is not required for indexing.

In Colab, clone the repository and change into its directory before importing `qa_pipeline`. Mount Google Drive only if your input files are stored there.

## Input and output

| Input | Required column |
|---|---|
| News CSV | `content` |
| Reviews CSV | `Review Text` |

Provide either source or both. In notebook configuration, set an unused source to `None`. Files must have headers and UTF-8 text.

```python
from qa_pipeline import prepare_corpus, get_csv

corpus = prepare_corpus(
    news_path="examples/news_sample.csv",
    reviews_path="examples/reviews_sample.csv",
    output_dir="output",
)
qa_csv = get_csv(corpus, output_dir="output", max_questions=10)
```

Generated files:

- `output/corpus.csv`: cleaned source text, one named column.
- `output/QA.csv`: generated question–answer pairs.

The sample CSVs contain fictional input text, not measured model outputs or a real collected dataset. Generated answers should be reviewed against their sources before reuse.

## Repository contents

| File | Purpose |
|---|---|
| `Mistral 7B.ipynb` | Step-by-step workflow and configuration |
| `qa_pipeline.py` | Preprocessing, question generation, retrieval and export |
| `examples/` | Small fictional source inputs |
| `tests/test_pipeline.py` | Data-flow and export checks |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes local datasets, output and downloaded model weights |

## Validation

```bash
python -m unittest discover -s tests -v
```

Four automated tests cover source merging without losing the first row, schema/empty-input validation, question filtering and CSV export with punctuation/Unicode. Export tests mock the answer chain; they do not assess model quality.

The Python module passed syntax checking and the four tests during cleanup. **A complete Mistral download/inference run has not been performed in this cleanup environment.** No generation accuracy, speed or dataset-size claim is made.

Question generation uses a bounded refine list, so it does not guarantee exhaustive coverage of a large corpus. FAISS retrieval and source-constrained prompting reduce unsupported answers but do not eliminate them.

## Author

**Komal Wahid**  
Data Science · NLP · Generative AI

## Implementation references

- [LangChain CTransformers reference](https://reference.langchain.com/python/langchain-community/llms/ctransformers/CTransformers)
- [LangChain refine-chain loader](https://reference.langchain.com/python/langchain-classic/chains/summarize/chain/load_summarize_chain)
- [Sentence Transformer embedding integration](https://docs.langchain.com/oss/python/integrations/embeddings/sentence_transformers)
