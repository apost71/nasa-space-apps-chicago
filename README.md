# NASA Space Apps Chicago - Data Explorer

A LangChain and LangGraph-based agent for exploring NASA AppEEARS data and performing RAG operations using Elasticsearch.

> [!WARNING]
> This is an active work in progress and subject to change significantly without notice.

## Features

- Search for data in NASA AppEEARS
- Download data from AppEEARS
- Ingest data into Elasticsearch
- Perform RAG queries on the ingested data
- Agent-based workflow using LangChain and LangGraph

## Installation

### Local Development

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



### Docker Compose Setup

1. Copy the example environment file and fill in your credentials:
```bash
cp .env.example .env
```



2. Build and start the services:
```bash
docker compose up --build
```

The services will be available at:
- MCP Server: http://localhost:8000/mcp
- Elasticsearch: http://localhost:9200
- Data Explorer: Available through the CLI

## Configuration

Edit the `.env` file with your credentials:

- `OPENAI_API_KEY`: Your OpenAI API key for LangChain
- `OPENAI_BASE_URL`: Base URL for the OpenAI-compatible API (default: https://api.openai.com/v1)
- `OPENAI_MODEL`: The OpenAI model to use.  ex: gpt-4o-mini-2024-07-18
- `ELASTICSEARCH_USERNAME`: Elasticsearch username (default: elastic)
- `ELASTICSEARCH_PASSWORD`: Elasticsearch password
- `APPEARS_USERNAME`: NASA AppEEARS username
- `APPEARS_PASSWORD`: NASA AppEEARS password

## Usage

### Docker Compose

You can use Docker Compose to run the MCP server and the data explorer.

```bash
docker compose up -d
```

This will start two services:

  - Data Explorer: http://localhost:7860
  - MCP Server: http://localhost:8000


MCP Inspector can be used to experiment with the MCP server.

```
fastmcp dev mcp/src/elastic_appears_mcp/server.py
```

The above command will start the inspector at localhost:6274.  Assuming the docker compose is up and running, the mcp server can be accessed using the streamable-http transport at `http://localhost:8000/mcp`.

The agent can:
1. Search for relevant data in AppEEARS
2. Download the data
3. Ingest it into Elasticsearch
4. Perform RAG queries to answer your question


## License

MIT