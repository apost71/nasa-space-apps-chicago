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

# Expose the Gradio port
EXPOSE 7860

# Run the web interface
CMD ["python", "-m", "data_explorer.web"] 