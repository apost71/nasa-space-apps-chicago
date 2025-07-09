"""
Script to ingest structured CSV data using LlamaIndex's VectorStoreIndex with Elasticsearch backend.
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
)
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.elasticsearch import ElasticsearchEmbedding
from elasticsearch import AsyncElasticsearch
from elasticsearch.client import MlClient
from dotenv import load_dotenv
import os
import glob
import argparse
import json
import logging
import sys
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """
    Configure logging with both file and console handlers.
    
    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, logs only to console.
    """
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Set logging level for specific modules
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

# Configure logging before anything else
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", f"logs/ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
)

logger = logging.getLogger(__name__)

load_dotenv()

def setup_elasticsearch() -> ElasticsearchStore:
    """
    Set up Elasticsearch vector store for LlamaIndex using credentials from env.
    """
    try:
        # Get Elasticsearch configuration from environment variables
        es_host = os.getenv("ELASTICSEARCH_HOST", "localhost")
        es_port = os.getenv("ELASTICSEARCH_PORT", "9200")
        es_user = os.getenv("ELASTICSEARCH_USER")
        es_password = os.getenv("ELASTICSEARCH_PASSWORD")
        
        # Create AsyncElasticsearch client with authentication
        es_client = AsyncElasticsearch(
            f"http://{es_host}:{es_port}",
            basic_auth=(es_user, es_password),
            verify_certs=False  # Set to True in production with proper certificates
        )
        
        # Create index with mapping for both vector and structured fields
        index_name = "light_emissivity_index"
        mapping = {
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": 768,  # BGE base model dimensions
                        "index": True,
                        "similarity": "cosine"
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "file_name": {"type": "keyword"},
                            "row_id": {"type": "keyword"},
                            "numeric_fields": {"type": "object"},
                            "text_fields": {"type": "object"},
                            "date_fields": {"type": "object"},
                            "all_fields": {"type": "keyword"},
                            "location": {"type": "geo_point"}
                        }
                    }
                }
            }
        }
        
        # Create index if it doesn't exist
        if not es_client.indices.exists(index=index_name):
            es_client.indices.create(index=index_name, body=mapping)
        
        vector_store = ElasticsearchStore(
            es_client=es_client,
            index_name=index_name,
            distance_strategy="COSINE",

        )
        
        logger.info("Connected to Elasticsearch vector store")
        return vector_store
    except Exception as e:
        logger.error(f"Error setting up Elasticsearch: {e}")
        raise

def load_csv_data(csv_file: Path) -> pd.DataFrame:
    """
    Load and preprocess CSV data into a pandas DataFrame.
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Convert numeric columns appropriately
        for col in df.columns:
            # Try to convert to numeric if possible
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                # If conversion fails, keep as is
                pass
        
        logger.info(f"Loaded DataFrame with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        raise

def sanitize_field_name(field_name: str) -> str:
    """
    Sanitize field names to be Elasticsearch-friendly by:
    1. Converting to lowercase
    2. Replacing spaces and special characters with underscores
    3. Removing any remaining special characters
    """
    # Convert to lowercase
    field_name = field_name.lower()
    # Replace spaces and special characters with underscores
    field_name = field_name.replace(' ', '_').replace('-', '_').replace('.', '_')
    # Remove any remaining special characters
    field_name = ''.join(c for c in field_name if c.isalnum() or c == '_')
    return field_name

def create_documents(df: pd.DataFrame, file_name: str) -> List[Document]:
    """
    Create LlamaIndex documents from DataFrame with structured metadata.
    """
    documents = []
    
    # Get field metadata from DataFrame attributes
    field_metadata = df.attrs.get('field_metadata', {})
    
    # Determine column types for better metadata organization
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    text_cols = df.select_dtypes(include=['object', 'string']).columns
    date_cols = df.select_dtypes(include=['datetime64']).columns
    
    # Create mapping of original column names to sanitized names
    column_mapping = {col: sanitize_field_name(col) for col in df.columns}
    
    for _, row in df.iterrows():
        # Create text representation for semantic search
        text_parts = []
        for col in text_cols:
            if pd.notna(row[col]):
                text_parts.append(f"{col}: {row[col]}")
        
        text = " | ".join(text_parts)
        
        # Create location field if latitude and longitude exist
        # location = None
        # if 'Latitude' in df.columns and 'Longitude' in df.columns:
        #     if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
        #         location = {
        #             "lat": float(row['Latitude']),
        #             "lon": float(row['Longitude'])
        #         }
        
        # Create structured metadata according to index mapping
        metadata = {
            "file_name": file_name,
            "row_id": str(row.name),
            "numeric_fields": {},
            "text_fields": {},
            "date_fields": {},
            "all_fields": [],
        }

        # if location:
        #     metadata["location"] = location
        
        # Add numeric fields with sanitized names
        for col in numeric_cols:
            if pd.notna(row[col]):
                sanitized_name = column_mapping[col]
                metadata["numeric_fields"][sanitized_name] = float(row[col])
                metadata["all_fields"].append(f"{col}: {row[col]}")
        
        # Add text fields with sanitized names
        for col in text_cols:
            if pd.notna(row[col]):
                sanitized_name = column_mapping[col]
                metadata["text_fields"][sanitized_name] = str(row[col])
                metadata["all_fields"].append(f"{col}: {row[col]}")
        
        # Add date fields with sanitized names
        for col in date_cols:
            if pd.notna(row[col]):
                sanitized_name = column_mapping[col]
                metadata["date_fields"][sanitized_name] = str(row[col])
                metadata["all_fields"].append(f"{col}: {row[col]}")
        
        # Create document with structured metadata
        doc_kwargs = {"text": text, "metadata": metadata}

        doc = Document(**doc_kwargs)
        documents.append(doc)
    
    return documents

def create_index(
    df: pd.DataFrame,
    vector_store: ElasticsearchStore,
    file_name: str
) -> VectorStoreIndex:
    """
    Create a VectorStoreIndex from the DataFrame with Elasticsearch backend.
    """
    try:
        # Create Elasticsearch embedding model with async support
        embeddings = ElasticsearchEmbedding.from_credentials(
            model_id="baai__bge-base-en",
            es_url=f"http://{os.getenv('ELASTICSEARCH_HOST', 'localhost')}:{os.getenv('ELASTICSEARCH_PORT', '9200')}",
            es_username=os.getenv('ELASTICSEARCH_USER'),
            es_password=os.getenv('ELASTICSEARCH_PASSWORD')
        )
        
        # Configure global settings with increased chunk size
        Settings.embed_model = embeddings
        Settings.chunk_size = 2048  # Increased from 512 to handle larger metadata
        
        # Create storage context with Elasticsearch
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        # Create documents with structured metadata
        documents = create_documents(df, file_name)
        
        # Create index from documents
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
            response_mode="tree_summarize"
        )
        
        logger.info(f"Successfully created index from DataFrame")
        return index
    except Exception as e:
        logger.error(f"Error creating index: {e}", exc_info=True)
        raise e

def process_csv_file(csv_file: Path, vector_store: ElasticsearchStore) -> None:
    """
    Process a single CSV file and add it to the index.
    """
    try:
        logger.info(f"\nProcessing file: {csv_file.name}")
        
        # Load CSV data
        df = load_csv_data(csv_file)
        
        # Create index
        index = create_index(df, vector_store, csv_file.name)
        
        # Create query engine
        query_engine = index.as_query_engine(response_mode="tree_summarize")
        
        # Optional: Test a query
        response = query_engine.query(
            f"Give me a summary of the data in {csv_file.name}"
        )
        logger.info(f"\nTest Query Response for {csv_file.name}:")
        logger.info(response)
        
    except Exception as e:
        logger.error(f"Error processing {csv_file.name}: {e}", exc_info=True)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process CSV files into Elasticsearch index.')
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Path to directory containing CSV files (default: ./data)',
        default=None
    )
    args = parser.parse_args()
    
    # Set up Elasticsearch vector store
    vector_store = setup_elasticsearch()
    
    # Get the data directory path
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent / "data"
    
    # Ensure the data directory exists
    if not data_dir.exists():
        logger.info(f"Creating data directory at {data_dir}")
        data_dir.mkdir(parents=True)
        logger.info("Please add your CSV files to the data directory and run the script again.")
        return
    
    # Find all CSV files in the data directory
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        logger.info(f"No CSV files found in {data_dir}")
        logger.info("Please add your CSV files to the data directory and run the script again.")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to process in {data_dir}")
    
    # Process each CSV file
    for csv_file in csv_files:
        process_csv_file(csv_file, vector_store)
    
    logger.info("\nAll files processed successfully!")

if __name__ == "__main__":
    main() 