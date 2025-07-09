"""
Agent module that implements a RAG-based agent using LlamaIndex.
"""
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.elasticsearch import ElasticsearchEmbedding
from elasticsearch import AsyncElasticsearch
import logging
from llama_index.core.tools.types import ToolMetadata
# Configure logging
logger = logging.getLogger(__name__)

load_dotenv()

def create_query_engine() -> Any:
    """
    Create a query engine for RAG using the Elasticsearch index.
    """
    try:
        # Get Elasticsearch configuration
        es_host = os.getenv("ELASTICSEARCH_HOST", "localhost")
        es_port = os.getenv("ELASTICSEARCH_PORT", "9200")
        es_user = os.getenv("ELASTICSEARCH_USER")
        es_password = os.getenv("ELASTICSEARCH_PASSWORD")
        
        # Create Elasticsearch client
        es_client = AsyncElasticsearch(
            f"http://{es_host}:{es_port}",
            basic_auth=(es_user, es_password),
            verify_certs=False
        )
        
        # Create vector store
        vector_store = ElasticsearchStore(
            es_client=es_client,
            index_name="light_emissivity_index",
            distance_strategy="COSINE"
        )
        
        # Create embedding model
        embeddings = ElasticsearchEmbedding.from_credentials(
            model_id="baai__bge-base-en",
            es_url=f"http://{es_host}:{es_port}",
            es_username=es_user,
            es_password=es_password
        )
        
        # Configure settings
        Settings.embed_model = embeddings
        Settings.chunk_size = 2048
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        # Create index
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )
        
        # Create query engine
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="tree_summarize"
        )
        
        return query_engine
    except Exception as e:
        logger.error(f"Error creating query engine: {e}")
        raise

def process_user_message(message: str, history: list) -> str:
    """
    Process a user message using the RAG-based agent.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize the LLM
    llm = OpenAI(model="gpt-4o-mini-2024-07-18", temperature=0, api_key=api_key)
    
    # Create memory
    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    
    # Create the query engine
    query_engine = create_query_engine()
    
    # Create the query engine tool
    query_engine_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="emissivity_data_query",
            description="""Use this tool to answer questions about emissivity and land surface temperature data.
            This tool can help with:
            - Analyzing land surface temperature patterns
            - Understanding emissivity values and their significance
            - Interpreting quality control data
            - Finding location-specific information
            
            The tool will use the indexed data to provide accurate, contextual answers.
            """
        )
    )
    
    # Create the agent with the query engine tool
    agent = ReActAgent(
        tools=[query_engine_tool],
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # Get response from agent
    response = agent.chat(message)
    return str(response) 