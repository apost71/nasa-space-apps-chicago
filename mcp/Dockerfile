FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY src/ src/

# Install Python dependencies
RUN pip install --no-cache-dir uv && \
    uv pip install --system -e .

# Expose the MCP server port
EXPOSE 8000

# Run the MCP server
CMD ["fastmcp", "run", "src/space_apps_mcp/server.py", "--transport", "streamable-http", "--port", "8000", "--host", "0.0.0.0"] 
