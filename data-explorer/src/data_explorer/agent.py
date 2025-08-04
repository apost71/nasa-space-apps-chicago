"""Data Explorer agent implementation using LangChain and LangGraph with MCP integration."""

import logging
from typing import Dict, List, Any, AsyncGenerator, Optional
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState

# Configure logging
logger = logging.getLogger(__name__)

load_dotenv()


def get_host_url(url: str) -> str:
    """Convert localhost URLs to use host.docker.internal when running in Docker."""
    if os.getenv("DOCKER_HOST_IP"):
        return url.replace("localhost", os.getenv("DOCKER_HOST_IP"))
    return url


async def create_agent():
    """Create a LangGraph agent with MCP tools and checkpoint memory."""
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
        streaming=True,  # Enable streaming
        max_tokens=4000,  # Limit response length to prevent infinite loops
        timeout=120,  # Add timeout to prevent hanging
    )
    logger.info("LLM initialized successfully")

    # Create checkpointer for memory management
    checkpointer = InMemorySaver()

    # Create agent with tools and checkpoint memory
    logger.info("Creating agent with tools and checkpoint memory...")
    agent = create_react_agent(model=llm, tools=tools, checkpointer=checkpointer)
    logger.info("Agent created successfully with checkpoint memory")

    # Store checkpointer with agent for memory management
    agent._checkpointer = checkpointer

    return agent


async def run_agent(
    query: str, agent: Optional[Any] = None, thread_id: Optional[str] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """Run the data explorer agent with streaming responses and checkpoint memory."""
    logger.info(f"Running agent with query: {query}")

    try:
        # Use existing agent or create new one
        if agent is None:
            agent = await create_agent()
            logger.info("Created new agent with checkpoint memory")
        else:
            logger.info("Using existing agent with checkpoint memory")

        # Generate thread ID if none provided
        if thread_id is None:
            import uuid

            thread_id = str(uuid.uuid4())
            logger.info(f"Generated thread ID: {thread_id}")

        # Get current messages for context
        messages = get_messages(agent, thread_id)
        logger.info(
            f"Running agent with {len(messages)} messages in history for thread: {thread_id}"
        )

        # Prepare input with new message
        input_messages = messages + [HumanMessage(content=query)]

        # Run the agent with streaming and checkpoint memory
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50,  # Increase recursion limit to handle complex workflows
        }

        step_count = 0
        consecutive_tool_calls = 0
        max_consecutive_tool_calls = 10  # Prevent infinite tool calling loops

        async for chunk in agent.astream(
            {"messages": input_messages}, config, stream_mode="values"
        ):
            step_count += 1
            logger.debug(f"Step {step_count}: Received chunk type: {type(chunk)}")

            # Log tool calls to help debug recursion issues
            if isinstance(chunk, dict) and "tools" in chunk:
                consecutive_tool_calls += 1
                logger.info(
                    f"Step {step_count}: Tool call detected (consecutive: {consecutive_tool_calls})"
                )

                # Check for potential infinite loop
                if consecutive_tool_calls > max_consecutive_tool_calls:
                    logger.warning(
                        f"Step {step_count}: Too many consecutive tool calls ({consecutive_tool_calls}), potential infinite loop detected"
                    )
                    yield {
                        "error": f"Agent appears to be stuck in a loop after {consecutive_tool_calls} consecutive tool calls. Please try rephrasing your query."
                    }
                    return

                if "messages" in chunk["tools"]:
                    for msg in chunk["tools"]["messages"]:
                        if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                            logger.info(f"Step {step_count}: Tool call ID: {msg.tool_call_id}")
            else:
                # Reset consecutive tool call counter when we get a non-tool response
                consecutive_tool_calls = 0

            yield chunk

        logger.info("Agent completed successfully")

    except Exception as e:
        logger.error(f"Error running agent: {str(e)}", exc_info=True)
        yield {"error": str(e)}


# Memory management functions
def get_messages(agent: Any, thread_id: str) -> List[BaseMessage]:
    """Get all messages in the conversation history for the given thread."""
    try:
        checkpointer = agent._checkpointer
        checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})

        if (
            checkpoint
            and "channel_values" in checkpoint
            and "messages" in checkpoint["channel_values"]
        ):
            return checkpoint["channel_values"]["messages"]

        logger.debug(f"No messages found in checkpoint for thread {thread_id}")
    except Exception as e:
        logger.warning(f"Could not retrieve messages for thread {thread_id}: {e}")

    return []


def clear_history(agent: Any, thread_id: str):
    """Clear the conversation history for the given thread."""
    try:
        checkpointer = agent._checkpointer
        checkpointer.delete_thread(thread_id)
        logger.info(f"Cleared conversation history for thread: {thread_id}")
    except Exception as e:
        logger.warning(f"Could not clear history for thread {thread_id}: {e}")


def get_history_summary(agent: Any, thread_id: str) -> str:
    """Get a summary of the conversation history for the given thread."""
    messages = get_messages(agent, thread_id)
    if not messages:
        return "No conversation history."

    summary = f"Conversation has {len(messages)} messages:\n"
    for i, msg in enumerate(messages[-5:], 1):  # Show last 5 messages
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        summary += f"{i}. {role}: {content_preview}\n"

    return summary


def get_thread_checkpoints(agent: Any, thread_id: str) -> List[Dict[str, Any]]:
    """Get all checkpoints for the given thread."""
    try:
        checkpointer = agent._checkpointer
        # Use the correct list() method from InMemorySaver
        checkpoints = list(checkpointer.list({"configurable": {"thread_id": thread_id}}))

        result = []
        for cp in checkpoints:
            try:
                # Each checkpoint is a CheckpointTuple with a checkpoint attribute
                checkpoint_data = cp.checkpoint

                checkpoint_id = checkpoint_data.get("id", "unknown")
                timestamp = checkpoint_data.get("ts", "unknown")

                # Get message count from channel_values
                message_count = 0
                if "channel_values" in checkpoint_data:
                    messages = checkpoint_data["channel_values"].get("messages", [])
                    message_count = len(messages)

                result.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "timestamp": timestamp,
                        "message_count": message_count,
                    }
                )
            except Exception as e:
                logger.warning(f"Error processing checkpoint: {e}")
                continue

        return result
    except Exception as e:
        logger.warning(f"Could not retrieve checkpoints for thread {thread_id}: {e}")
        return []
