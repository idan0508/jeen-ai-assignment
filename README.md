# Document Indexing & Semantic Retrieval

Index PDF and DOCX files into PostgreSQL (pgvector) using Gemini embeddings, then search them by meaning from the command line.

```
file --> extract text --> split into chunks --> embed (gemini-embedding-001) --> store (pgvector) --> semantic search
```

## Prerequisites

- Python 3.11 or newer (developed and tested on 3.11)
- A [Gemini API key](https://aistudio.google.com/apikey)
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension (Docker setup included)

## Installation

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in the two variables. Secrets are never stored in code and never printed.

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Your Gemini API key |
| `POSTGRES_URL` | PostgreSQL connection string, e.g. `postgresql://USER:PASSWORD@HOST:5432/DATABASE` |

## Database Setup

### Option 1 (Recommended): Docker

```bash
docker compose up -d
```

This starts PostgreSQL 17 with pgvector preinstalled on port 5432. Use the local development credentials from `docker-compose.yml` in your `.env`:

```
POSTGRES_URL=postgresql://docindexer:docindexer@localhost:5432/docindexer
```

The table and index are created automatically on the first indexing run.

### Option 2: Existing PostgreSQL

Install pgvector on the server, create a database, and put its connection string in `POSTGRES_URL`.
The tool enables the extension itself when it is missing, which requires permission to run `CREATE EXTENSION vector;`.
If your database user lacks that permission, ask an admin to run the statement once; the tool detects the existing extension and skips it.

## Usage & Sample Output

### Index a document

```bash
python index_documents.py --file ./docs/example.pdf --strategy paragraph
```

| Option | Description | Default |
|---|---|---|
| `--file` | Path to a `.pdf` or `.docx` file (required) | - |
| `--strategy` | `fixed`, `sentence` or `paragraph` | `paragraph` |
| `--chunk-size` | Maximum characters per chunk | `500` |
| `--overlap` | Characters shared by neighbouring chunks (see [Chunking](#chunking)) | `100` |
| `--verbose` | Show debug output (never includes secrets) | off |

Re-indexing the same file with the same strategy replaces its previous chunks.

Example indexing run (both example files, all strategies, default parameters):

```
$ python index_documents.py --file ./docs/example.pdf --strategy fixed
Indexed 'example.pdf' with strategy 'fixed' (chunk_size=500, overlap=100): 1823 characters -> 5 chunks stored.

$ python index_documents.py --file ./docs/example.pdf --strategy sentence
Indexed 'example.pdf' with strategy 'sentence' (chunk_size=500, overlap=100): 1823 characters -> 4 chunks stored.

$ python index_documents.py --file ./docs/example.pdf --strategy paragraph
Indexed 'example.pdf' with strategy 'paragraph' (chunk_size=500, overlap=100): 1823 characters -> 6 chunks stored.

$ python index_documents.py --file ./docs/example.docx --strategy fixed
Indexed 'example.docx' with strategy 'fixed' (chunk_size=500, overlap=100): 1823 characters -> 5 chunks stored.

$ python index_documents.py --file ./docs/example.docx --strategy sentence
Indexed 'example.docx' with strategy 'sentence' (chunk_size=500, overlap=100): 1823 characters -> 4 chunks stored.

$ python index_documents.py --file ./docs/example.docx --strategy paragraph
Indexed 'example.docx' with strategy 'paragraph' (chunk_size=500, overlap=100): 1823 characters -> 6 chunks stored.
```

### Search

```bash
python search.py --query "login issue"
```

| Option | Description | Default |
|---|---|---|
| `--query` | Natural-language search text (required) | - |
| `--top-k` | Number of results | `5` |
| `--strategy` | Only search chunks created with this strategy | all |
| `--verbose` | Show debug output | off |

Each result shows its rank, cosine similarity score, file, strategy, chunk number and the full chunk text.

Example search results returned from the database:

```
$ python search.py --query "login issue" --top-k 3
Top 3 results for "login issue":
------------------------------------------------------------
Rank:      1
Score:     0.7091
File:      example.pdf
Strategy:  paragraph
Chunk:     #0
Text:
    Acme Support Center - Frequently Asked Questions

    Login issues. If you cannot log in to your account, first check that Caps Lock is off and that you are using the email address you registered with. After five failed attempts the account is locked for 15 minutes for security reasons. If the problem continues, clear your browser cookies or try a private window.
------------------------------------------------------------
Rank:      2
Score:     0.7091
File:      example.docx
Strategy:  paragraph
Chunk:     #0
Text:
    Acme Support Center - Frequently Asked Questions

    Login issues. If you cannot log in to your account, first check that Caps Lock is off and that you are using the email address you registered with. After five failed attempts the account is locked for 15 minutes for security reasons. If the problem continues, clear your browser cookies or try a private window.
------------------------------------------------------------
Rank:      3
Score:     0.6999
File:      example.pdf
Strategy:  sentence
Chunk:     #0
Text:
    Acme Support Center - Frequently Asked Questions Login issues. If you cannot log in to your account, first check that Caps Lock is off and that you are using the email address you registered with. After five failed attempts the account is locked for 15 minutes for security reasons. If the problem continues, clear your browser cookies or try a private window. Password reset. Select the Forgot password link on the login page and enter your email address.
------------------------------------------------------------
```

The search is semantic: the query below shares no keywords with the best match ("refund", "cancel"), and results can be limited to one strategy.
The PDF and DOCX examples contain identical text, so each result appears once per file.

```
$ python search.py --query "how do I get my money back" --strategy paragraph --top-k 3
Top 3 results for "how do I get my money back":
------------------------------------------------------------
Rank:      1
Score:     0.6134
File:      example.pdf
Strategy:  paragraph
Chunk:     #4
Text:
    Cancelling a subscription. You may cancel at any time from the Subscription page. Your plan stays active until the end of the paid period, and your data is kept for 90 days afterwards in case you decide to return. Refunds are available within 14 days of the first payment.
------------------------------------------------------------
Rank:      2
Score:     0.6134
File:      example.docx
Strategy:  paragraph
Chunk:     #4
Text:
    Cancelling a subscription. You may cancel at any time from the Subscription page. Your plan stays active until the end of the paid period, and your data is kept for 90 days afterwards in case you decide to return. Refunds are available within 14 days of the first payment.
------------------------------------------------------------
Rank:      3
Score:     0.5829
File:      example.pdf
Strategy:  paragraph
Chunk:     #2
Text:
    Two-factor authentication. When two-factor authentication is enabled you must enter a six digit code from your authenticator app at every login. If you lost your phone, use one of the recovery codes you saved during setup. Without a recovery code, our team must verify your identity before disabling the feature.
------------------------------------------------------------
```

If nothing matches, the tool prints `No results found. Have you indexed any documents yet?` and exits with code 0.

### Inspect the stored rows

```bash
docker compose exec db psql -U docindexer -d docindexer -P pager=off -c "SELECT id, filename, split_strategy, chunk_index, vector_dims(embedding) AS dims, left(chunk_text, 40) AS chunk_preview, created_at FROM document_chunks WHERE filename = 'example.pdf' AND split_strategy = 'paragraph' ORDER BY chunk_index"
```

```
 id |  filename   | split_strategy | chunk_index | dims |              chunk_preview               |          created_at
----+-------------+----------------+-------------+------+------------------------------------------+-------------------------------
 82 | example.pdf | paragraph      |           0 |  768 | Acme Support Center - Frequently Asked Q | 2026-11-25 18:56:42.508088+00
 83 | example.pdf | paragraph      |           1 |  768 | Password reset. Select the Forgot passwo | 2026-11-25 18:56:42.508088+00
 84 | example.pdf | paragraph      |           2 |  768 | Two-factor authentication. When two-fact | 2026-11-25 18:56:42.508088+00
 85 | example.pdf | paragraph      |           3 |  768 | Billing and invoices. Invoices are gener | 2026-11-25 18:56:42.508088+00
 86 | example.pdf | paragraph      |           4 |  768 | Cancelling a subscription. You may cance | 2026-11-25 18:56:42.508088+00
 87 | example.pdf | paragraph      |           5 |  768 | Contacting support. Our team answers ema | 2026-11-25 18:56:42.508088+00
(6 rows)
```

## Project Structure

```
.
├── index_documents.py          # CLI: index a PDF/DOCX file
├── search.py                   # CLI: semantic search
├── doc_indexer/
│   ├── config.py               # loads and validates .env settings
│   ├── errors.py               # domain exceptions with exit codes
│   ├── extraction.py           # PDF/DOCX -> clean text
│   ├── chunking.py             # the three chunking strategies
│   ├── embeddings.py           # Gemini client: batching, retry, L2 normalization
│   ├── db.py                   # VectorRepository (schema, storage, similarity search)
│   ├── cli.py                  # shared CLI error handling and logging
│   └── services/
│       ├── indexing_service.py # extract -> chunk -> embed -> store
│       └── search_service.py   # embed query -> vector search
├── tests/                      # unit and integration tests
├── docs/                       # example.pdf and example.docx
├── docker-compose.yml          # PostgreSQL 17 with pgvector
├── .env.example
├── requirements.txt
└── requirements-dev.txt
```

## Requirements Checklist

| Requirement | Where |
|---|---|
| `index_documents.py` with a simple terminal command | `index_documents.py` |
| Clean text extraction; unsupported files and documents without text handled | `doc_indexer/extraction.py` |
| Fixed-size with overlap, sentence-based and paragraph-based chunking | `doc_indexer/chunking.py` |
| Embeddings with `gemini-embedding-001` | `doc_indexer/embeddings.py` |
| PostgreSQL + pgvector storage with the required columns | `doc_indexer/db.py` |
| Semantic search via vector similarity | `search.py`, `doc_indexer/services/search_service.py` |
| `.env` for `GEMINI_API_KEY` and `POSTGRES_URL`, no secrets in code or output | `doc_indexer/config.py` |
| Error handling for all listed cases | [Error Handling](#error-handling) |

## Chunking

| Strategy | How it splits |
|---|---|
| `fixed` | Windows of `--chunk-size` characters; neighbouring chunks share `--overlap` characters |
| `sentence` | Whole sentences packed into chunks of up to `--chunk-size` characters; common abbreviations such as "Dr." do not end a sentence |
| `paragraph` | Whole paragraphs (blank-line separated) packed into chunks of up to `--chunk-size` characters |

`--chunk-size` applies to every strategy. `--overlap` only affects `fixed`, and the fallback used when a single sentence or paragraph is longer than a whole chunk.
`--overlap` must be smaller than `--chunk-size`.

## Database Schema

```sql
CREATE TABLE document_chunks (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    filename        TEXT        NOT NULL,
    split_strategy  TEXT        NOT NULL CHECK (split_strategy IN ('fixed', 'sentence', 'paragraph')),
    chunk_index     INT         NOT NULL,
    chunk_text      TEXT        NOT NULL,
    embedding       vector(768) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (filename, split_strategy, chunk_index)
);

CREATE INDEX document_chunks_embedding_hnsw
    ON document_chunks USING hnsw (embedding vector_cosine_ops);
```

## Error Handling

Errors are printed to stderr as a short message and the process exits with a category-specific code. Messages are fixed text and never contain API keys, connection strings or stack traces.

| Situation | Exit code | Message |
|---|---|---|
| Missing configuration | 2 | `Missing required environment variable(s): GEMINI_API_KEY. Copy .env.example to .env and fill them in.` |
| Missing file | 3 | `File not found: <path>` |
| Unsupported file type | 3 | `Unsupported file type '.txt'. Supported types: .pdf, .docx` |
| Document without text | 3 | `No extractable text found. The file may be empty or a scanned document (images only), which would require OCR.` |
| Embedding failure | 4 | `Gemini rejected the request (HTTP 400). Check that GEMINI_API_KEY is valid and has access to gemini-embedding-001.` |
| Database connection failure | 5 | `Could not connect to PostgreSQL. Check POSTGRES_URL in your .env file and make sure the database is running and reachable.` |
| pgvector not enabled, no permission | 5 | `The pgvector extension is not enabled and the database user is not allowed to create it. Ask a database admin to run: CREATE EXTENSION vector;` |
| Empty search results | 0 | `No results found. Have you indexed any documents yet?` |
| Unexpected error | 1 | `Unexpected error (re-run with --verbose to see the error type).` |

## Design Decisions

- **768-dimension embeddings, L2-normalized:** the model's default 3072 dimensions exceed pgvector's 2000-dimension limit for indexes, and truncated vectors are not unit length.
- **Separate task types:** documents use `RETRIEVAL_DOCUMENT` and queries use `RETRIEVAL_QUERY`, as recommended for retrieval.
- **Selective retry:** only rate limits (429), server errors (5xx) and network failures are retried with backoff; invalid keys and bad requests fail immediately.
- **Cheapest checks first:** file, parameter and schema checks run before the paid embedding call, so bad input never costs an API request.
- **Atomic re-indexing:** old chunks are deleted and new ones inserted in one transaction, so a failure never leaves partial data.
- **Extension check before creation:** the tool looks in `pg_extension` first, so it works for database users without `CREATE EXTENSION` rights.
- **Cosine similarity with an HNSW index:** scores are `1 - cosine distance`, so higher means more similar.
- **Services layer:** the indexing and search workflows receive their embedder and repository as arguments, which keeps them testable without network or database.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The unit tests need no network or database and cover extraction, chunking, embeddings (normalization, retry rules), the services and both CLIs (exit codes, messages, secret hygiene).
The integration tests run against a real PostgreSQL with pgvector and are skipped unless `TEST_POSTGRES_URL` is set (they only touch rows whose filename starts with `itest_`):

```bash
# macOS / Linux
TEST_POSTGRES_URL="postgresql://docindexer:docindexer@localhost:5432/docindexer" pytest tests/test_db_integration.py
# Windows (PowerShell)
$env:TEST_POSTGRES_URL = "postgresql://docindexer:docindexer@localhost:5432/docindexer"; pytest tests/test_db_integration.py
```

## Known Limitations

- Text inside DOCX tables is not extracted (body paragraphs only).
- Scanned PDFs without a text layer are rejected; there is no OCR.
- Chunking parameters are not stored in the database, and re-indexing a file with the same strategy replaces its chunks.
- `filename` stores the file name only, so two files with the same name in different folders replace each other.
- Sentence splitting is rule-based and knows only a small list of abbreviations.
- Paragraph detection in PDFs depends on the file's layout; PDFs without visible paragraph gaps fall back to fixed-size splitting.
- Filtering by strategy is applied after the HNSW index lookup, so it can return fewer than `--top-k` results on large tables.
- There is no command to delete an indexed document.
