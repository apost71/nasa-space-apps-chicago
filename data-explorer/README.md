# Data Explorer

A LangChain and LangGraph-based agent for exploring NASA AppEEARS data and performing RAG operations using Elasticsearch.

## Features

- Search for data in NASA AppEEARS
- Download data from AppEEARS
- Ingest data into Elasticsearch
- Perform RAG queries on the ingested data
- Agent-based workflow using LangChain and LangGraph

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nasa-space-apps-chicago/data-explorer.git
cd data-explorer
```

2. Install uv (if not already installed):
```bash
pip install uv
```

3. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. Install the package in development mode:
```bash
uv pip install -e .
```

5. Copy the example environment file and fill in your credentials:
```bash
cp .env.example .env
```

## Configuration

Edit the `.env` file with your credentials:

- `OPENAI_API_KEY`: Your OpenAI API key for LangChain
- `OPENAI_BASE_URL`: Base URL for the OpenAI-compatible API (default: https://api.openai.com/v1)
- `ELASTICSEARCH_HOST`: Elasticsearch host (default: localhost)
- `ELASTICSEARCH_PORT`: Elasticsearch port (default: 9200)
- `ELASTICSEARCH_USERNAME`: Elasticsearch username
- `ELASTICSEARCH_PASSWORD`: Elasticsearch password
- `APPEARS_USERNAME`: NASA AppEEARS username
- `APPEARS_PASSWORD`: NASA AppEEARS password
- `MCP_SERVER_URL`: URL of the MCP server (default: http://localhost:8000/mcp)

## Usage

Run a data exploration query:

```bash
data-explorer explore "Find temperature data for Chicago from 2020 to 2023"
```

The agent will:
1. Search for relevant data in AppEEARS
2. Download the data
3. Ingest it into Elasticsearch
4. Perform RAG queries to answer your question

## Development

Install development dependencies:

```bash
uv pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Format code:

```bash
black .
isort .
```

## License

MIT 