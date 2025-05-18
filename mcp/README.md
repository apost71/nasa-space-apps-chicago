# Space Apps MCP Server

A Message Control Protocol (MCP) server implementation for NASA Space Apps Chicago data exploration tools. This server provides a standardized interface for interacting with Elasticsearch and NASA AppEEARS services.

## Features

- **Elasticsearch Integration**
  - List available indices
  - Search indices with custom queries
  - Ingest single documents
  - Bulk ingest multiple documents

- **NASA AppEEARS Integration**
  - List available products
  - Get layers for specific products
  - Submit point requests with multiple layers
  - Check task status
  - Download task results

## Installation

### Using Docker (Recommended)

1. Build the Docker image:
```bash
docker build -t space-apps-mcp .
```

2. Run the container:
```bash
docker run -p 8000:8000 \
  -e ELASTICSEARCH_URL=your_elasticsearch_url \
  -e APPEARS_USERNAME=your_appears_username \
  -e APPEARS_PASSWORD=your_appears_password \
  space-apps-mcp
```

### Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/nasa-space-apps-chicago/mcp-server.git
cd mcp-server
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

4. Set up environment variables:
```bash
export ELASTICSEARCH_URL=your_elasticsearch_url
export APPEARS_USERNAME=your_appears_username
export APPEARS_PASSWORD=your_appears_password
```

5. Run the server:
```bash
python -m space_apps_mcp.server
```

## Environment Variables

- `ELASTICSEARCH_URL`: URL of your Elasticsearch instance
- `APPEARS_USERNAME`: NASA AppEEARS username
- `APPEARS_PASSWORD`: NASA AppEEARS password

## API Endpoints

The server exposes the following MCP tools:

### Elasticsearch Tools

- `list_elastic_indices()`: List all available Elasticsearch indices
- `search_elastic_index(index, query)`: Search an Elasticsearch index
- `ingest_elastic_document(index, document)`: Ingest a single document
- `bulk_ingest_elastic(index, documents)`: Bulk ingest multiple documents

### AppEEARS Tools

- `list_appears_products()`: List available AppEEARS products
- `get_appears_layers(product_and_version)`: Get layers for a product
- `submit_appears_point_request(layers, locations, start_date, end_date, task_name)`: Submit a point request
- `get_appears_task_status(task_id)`: Check task status
- `download_appears_task(task_id, output_path)`: Download task results

## Development

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Run tests:
```bash
pytest
```

3. Format code:
```bash
black .
isort .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 