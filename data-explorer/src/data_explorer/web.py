import json
import logging
import os
from typing import Any, Dict, List, Optional, Generator, AsyncGenerator, Tuple
import asyncio
import gradio as gr
from gradio import ChatMessage
from dotenv import load_dotenv

from .agent import (
    create_agent,
    run_agent,
    get_history_summary,
    clear_history,
    get_messages,
    get_thread_checkpoints,
)

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Global agent
agent: Optional[Any] = None
# Default thread ID for web interface
DEFAULT_THREAD_ID = "web_interface"


def _is_tool_message(message) -> bool:
    """Check if a message is a tool message that should be filtered out."""
    # Check if it's a tool message by role or type
    if hasattr(message, "type") and message.type == "tool":
        return True
    if hasattr(message, "role") and message.role == "tool":
        return True
    if hasattr(message, "tool_call_id"):
        return True
    return False


def _is_assistant_message(message) -> bool:
    """Check if a message is an assistant message that should be included."""
    # Check if it's an assistant message by role or type
    if hasattr(message, "type") and message.type == "ai":
        return True
    if hasattr(message, "role") and message.role == "assistant":
        return True
    if hasattr(message, "type") and message.type == "assistant":
        return True
    # If no role/type is specified, assume it's an assistant message (for backward compatibility)
    if not hasattr(message, "role") and not hasattr(message, "type"):
        return True
    return False


async def explore_data(
    query: str, history: Optional[List[ChatMessage]] = None
) -> AsyncGenerator[Tuple[str, List[ChatMessage]], None]:
    """Run a data exploration query and stream the results with checkpoint memory using ChatMessage."""
    global agent

    try:
        logger.info(f"Processing query: {query}")

        # Initialize agent if not already done
        if agent is None:
            logger.info("Initializing agent with checkpoint memory...")
            agent = await create_agent()

        # Convert history to list of ChatMessage objects
        messages = history or []
        logger.info(f"Starting with {len(messages)} existing messages")

        # Add user message immediately
        messages.append(ChatMessage(role="user", content=query))
        yield "", messages

        # Stream directly from the agent like the LangChain example
        logger.info("Invoking agent with query...")

        # Get current messages for context
        from .agent import get_messages

        current_messages = get_messages(agent, DEFAULT_THREAD_ID)
        logger.info(f"Running agent with {len(current_messages)} messages in history")

        # Prepare input with new message
        from langchain_core.messages import HumanMessage

        input_messages = current_messages + [HumanMessage(content=query)]

        # Run the agent with streaming and checkpoint memory
        config = {"configurable": {"thread_id": DEFAULT_THREAD_ID}, "recursion_limit": 50}

        step_count = 0
        consecutive_tool_calls = 0
        max_consecutive_tool_calls = 10

        async for chunk in agent.astream(
            {"messages": input_messages},
            config,
            # stream_mode="values"
        ):
            step_count += 1
            logger.info(f"Step {step_count}: Received chunk type: {type(chunk)}")
            logger.info(f"Step {step_count}: Chunk: {chunk}")
            if isinstance(chunk, dict):
                logger.info(f"Step {step_count}: Chunk keys: {list(chunk.keys())}")
                if "messages" in chunk:
                    logger.info(f"Step {step_count}: Messages count: {len(chunk['messages'])}")
                if "output" in chunk:
                    logger.info(f"Step {step_count}: Output: {chunk['output'][:100]}...")

            # Handle tool usage - LangGraph format
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
                    messages.append(
                        ChatMessage(
                            role="assistant",
                            content=f"Agent appears to be stuck in a loop after {consecutive_tool_calls} consecutive tool calls. Please try rephrasing your query.",
                            metadata={"title": "❌ Loop Detected", "status": "done"},
                        )
                    )
                    yield "", messages
                    return

                # Process tool messages in LangGraph format
                if "messages" in chunk["tools"]:
                    for msg in chunk["tools"]["messages"]:
                        if hasattr(msg, "name") and hasattr(msg, "content"):
                            tool_name = msg.name
                            tool_content = msg.content
                            # Try to parse JSON content for better display
                            try:
                                tool_data = json.loads(tool_content)
                                if isinstance(tool_data, dict):
                                    # Show a summary of the tool result
                                    if "status" in tool_data:
                                        summary = f"Tool {tool_name} completed with status: {tool_data['status']}"
                                        if "jobs" in tool_data:
                                            summary += f" (found {len(tool_data['jobs'])} jobs)"
                                        elif "total_jobs" in tool_data:
                                            summary += f" (total: {tool_data['total_jobs']} jobs)"
                                    else:
                                        summary = f"Tool {tool_name} returned data"
                                else:
                                    summary = (
                                        f"Tool {tool_name} returned: {str(tool_data)[:100]}..."
                                    )
                            except json.JSONDecodeError:
                                # If not JSON, show raw content
                                summary = f"Tool {tool_name}: {tool_content[:100]}..."

                            messages.append(
                                ChatMessage(
                                    role="assistant",
                                    content=summary,
                                    metadata={"title": f"🛠️ Used tool {tool_name}"},
                                )
                            )
                            yield "", messages
            else:
                # Reset consecutive tool call counter when we get a non-tool response
                consecutive_tool_calls = 0

            # Handle final output - LangGraph uses different structures
            if isinstance(chunk, dict):
                # Check for various output formats
                if "output" in chunk:
                    messages.append(ChatMessage(role="assistant", content=chunk["output"]))
                    yield "", messages
                elif "messages" in chunk:
                    # Extract assistant messages from the messages list
                    for message in chunk["messages"]:
                        if hasattr(message, "content") and message.content:
                            # Skip tool messages and only include assistant messages
                            if not _is_tool_message(message) and _is_assistant_message(message):
                                messages.append(
                                    ChatMessage(role="assistant", content=message.content)
                                )
                                yield "", messages
                elif "agent" in chunk and "messages" in chunk["agent"]:
                    # Old format - extract assistant messages
                    for message in chunk["agent"]["messages"]:
                        if hasattr(message, "content") and message.content:
                            if not _is_tool_message(message) and _is_assistant_message(message):
                                messages.append(
                                    ChatMessage(role="assistant", content=message.content)
                                )
                                yield "", messages

        logger.info("Query processing completed successfully")

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        messages.append(
            ChatMessage(
                role="assistant",
                content=f"Error: {str(e)}",
                metadata={"title": "❌ Error", "status": "done"},
            )
        )
        yield "", messages


async def clear_conversation():
    """Clear the conversation history."""
    global agent
    if agent:
        clear_history(agent, DEFAULT_THREAD_ID)
        logger.info("Conversation history cleared")
        return "Conversation history cleared successfully."
    return "No agent available."


async def get_conversation_summary():
    """Get a summary of the current conversation."""
    global agent
    if agent:
        summary = get_history_summary(agent, DEFAULT_THREAD_ID)
        return summary
    return "No agent available."


async def get_conversation_count():
    """Get the number of messages in the conversation."""
    global agent
    if agent:
        count = len(get_messages(agent, DEFAULT_THREAD_ID))
        return f"Conversation has {count} messages."
    return "No agent available."


async def get_conversation_checkpoints():
    """Get checkpoint information for the current conversation."""
    global agent
    if agent:
        checkpoints = get_thread_checkpoints(agent, DEFAULT_THREAD_ID)
        if checkpoints:
            checkpoint_info = "Conversation checkpoints:\n"
            for i, cp in enumerate(checkpoints[-5:], 1):  # Show last 5 checkpoints
                checkpoint_info += f"{i}. ID: {cp['checkpoint_id'][:8]}... | Messages: {cp['message_count']} | Time: {cp['timestamp']}\n"
            return checkpoint_info
        else:
            return "No checkpoints available."
    return "No agent available."


# Create the Gradio interface
def create_interface() -> gr.Blocks:
    """Create and return the Gradio interface with memory management."""
    logger.info("Creating Gradio interface with memory...")

    with gr.Blocks(
        title="NASA Space Apps Chicago - Data Explorer", theme=gr.themes.Soft()
    ) as interface:
        gr.Markdown("""
        # NASA Space Apps Chicago - Data Explorer
        
        This interface allows you to explore NASA AppEEARS data using natural language queries.
        The agent will:
        1. Search for relevant data in AppEEARS
        2. Download the data
        3. Ingest it into Elasticsearch
        4. Perform RAG queries to answer your question
        
        **The agent now has conversation memory!** It will remember previous queries and context.
        """)

        with gr.Row():
            with gr.Column(scale=4):
                # Chat interface
                chatbot = gr.Chatbot(
                    type="messages",  # Use ChatMessage dataclass for proper agent integration
                    label="Conversation",
                    height=500,
                    show_label=True,
                    container=True,
                    bubble_full_width=False,
                )

                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="Enter your data exploration query...",
                        label="Message",
                        lines=1,
                        scale=4,
                        max_lines=1,
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("Clear Conversation", variant="secondary", size="sm")
                    clear_memory_btn = gr.Button("Clear Memory", variant="secondary", size="sm")

            with gr.Column(scale=1):
                # Memory management panel
                gr.Markdown("### Conversation Memory")

                with gr.Row():
                    summary_btn = gr.Button("Show Summary", size="sm")
                    count_btn = gr.Button("Message Count", size="sm")
                    checkpoints_btn = gr.Button("Checkpoints", size="sm")

                memory_output = gr.Textbox(label="Memory Info", lines=8, interactive=False)

        # Examples
        gr.Markdown("### Example Queries")
        gr.Examples(
            examples=[
                ["Find temperature data for Chicago from 2020 to 2023"],
                ["What is the average precipitation in Illinois during summer months?"],
                ["Show me vegetation index data for the Great Lakes region"],
                ["Can you tell me more about the data you found earlier?"],
                ["What other types of data are available for this region?"],
            ],
            inputs=query_input,
        )

        # Event handlers
        submit_btn.click(
            fn=explore_data, inputs=[query_input, chatbot], outputs=[query_input, chatbot]
        )

        # Enter key submission
        query_input.submit(
            fn=explore_data,
            inputs=[query_input, chatbot],
            outputs=[query_input, chatbot],
            api_name="submit_message",
        )

        clear_btn.click(
            fn=lambda: (
                [],
                "",
            ),  # Empty list for ChatMessage objects, empty string for memory output
            inputs=[],
            outputs=[chatbot, memory_output],
        )

        clear_memory_btn.click(fn=clear_conversation, inputs=[], outputs=[memory_output])

        summary_btn.click(fn=get_conversation_summary, inputs=[], outputs=[memory_output])

        count_btn.click(fn=get_conversation_count, inputs=[], outputs=[memory_output])

        checkpoints_btn.click(fn=get_conversation_checkpoints, inputs=[], outputs=[memory_output])

    logger.info("Gradio interface with memory created successfully")
    return interface


def main():
    """Run the Gradio web interface."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.info("Starting Data Explorer web interface...")

    interface = create_interface()
    logger.info("Launching Gradio interface on 0.0.0.0:7860")
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
