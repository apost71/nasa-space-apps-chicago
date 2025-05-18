"""Data Explorer agent implementation using LangChain and LangGraph with MCP integration."""

import logging
from typing import Dict, List, Any, AsyncGenerator
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

# Configure logging
logger = logging.getLogger(__name__)

load_dotenv()

def get_host_url(url: str) -> str:
    """Convert localhost URLs to use host.docker.internal when running in Docker."""
    if os.getenv('DOCKER_HOST_IP'):
        return url.replace('localhost', os.getenv('DOCKER_HOST_IP'))
    return url

async def create_agent():
    """Create a LangGraph agent with MCP tools."""
    logger.info("Initializing MCP client...")
    # Initialize MCP client
    client = MultiServerMCPClient(
        {
            "elastic_appears": {
                "url": get_host_url(os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")),
                "transport": "streamable_http",
            }
        }
    )

    # Get tools from MCP server
    logger.info("Fetching tools from MCP server...")
    tools = await client.get_tools()
    logger.info(f"Retrieved {len(tools)} tools from MCP server")

    logger.info("Initializing LLM...")
    base_url = get_host_url(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    logger.info(f"OpenAI Base URL: {base_url}")
    logger.info(f"OpenAI API Key: {os.getenv('OPENAI_API_KEY')}")
    llm = ChatOpenAI(
        base_url=base_url,
        model=os.getenv("OPENAI_MODEL", "/models/Qwen_Qwen3-8B-Q6_K_L.gguf"),
        temperature=0,
        streaming=True  # Enable streaming
    )
    logger.info("LLM initialized successfully")

    # Create agent with tools
    logger.info("Creating agent with tools...")
    agent = create_react_agent(
        model=llm,
        tools=tools
    )
    logger.info("Agent created successfully")

    return agent

async def run_agent(query: str) -> AsyncGenerator[Dict[str, Any], None]:
    """Run the data explorer agent with streaming responses."""
    logger.info(f"Running agent with query: {query}")
    
    try:
        agent = await create_agent()
        logger.info("Agent created, invoking with query...")
        
        # Run the agent with streaming
        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": query}]}
        ):
            logger.debug(f"Received chunk: {chunk}")
            yield chunk
            
        logger.info("Agent completed successfully")
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}", exc_info=True)
        yield {"error": str(e)} 