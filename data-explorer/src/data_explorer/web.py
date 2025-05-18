import logging
import os
from typing import Dict, List, Optional, Generator, AsyncGenerator
import asyncio
import gradio as gr
from dotenv import load_dotenv

from .agent import create_agent, run_agent

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Create the agent
agent = None

async def explore_data(query: str, history: Optional[List[Dict[str, str]]] = None) -> AsyncGenerator[str, None]:
    """Run a data exploration query and stream the results."""
    try:
        logger.info(f"Processing query: {query}")
        
        
        # Run the agent with streaming
        logger.info("Invoking agent with query...")
        current_response = ""
        async for chunk in run_agent(query):
            if "error" in chunk:
                logger.error(f"Error from agent: {chunk['error']}")
                yield f"Error: {chunk['error']}"
                return
                
            if "agent" in chunk:
                if "messages" in chunk["agent"]:
                    for message in chunk["agent"]["messages"]:
                        current_response += message.content
                    yield current_response
                    
            # elif "tools" in chunk:
            #     if "messages" in chunk["tools"]:
            #         for message in chunk["tools"]["messages"]:
            #             current_response += message.content
            #         yield current_response
                
            if "steps" in chunk:
                steps_text = "\nSteps taken:\n" + "\n".join(f"- {step}" for step in chunk["steps"])
                yield current_response + steps_text
        
        logger.info("Query processing completed successfully")
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        yield f"Error: {str(e)}"

# Create the Gradio interface
def create_interface() -> gr.Interface:
    """Create and return the Gradio interface."""
    logger.info("Creating Gradio interface...")
    interface = gr.Interface(
        fn=explore_data,  # Gradio will handle the async function directly
        inputs=[
            gr.Textbox(
                label="Query",
                placeholder="Enter your data exploration query...",
                lines=3,
            ),
        ],
        outputs=gr.Textbox(
            label="Results",
            lines=10,
        ),
        title="NASA Space Apps Chicago - Data Explorer",
        description="""
        This interface allows you to explore NASA AppEEARS data using natural language queries.
        The agent will:
        1. Search for relevant data in AppEEARS
        2. Download the data
        3. Ingest it into Elasticsearch
        4. Perform RAG queries to answer your question
        
        Example queries:
        - "Find temperature data for Chicago from 2020 to 2023"
        - "What is the average precipitation in Illinois during summer months?"
        - "Show me vegetation index data for the Great Lakes region"
        """,
        examples=[
            ["Find temperature data for Chicago from 2020 to 2023"],
            ["What is the average precipitation in Illinois during summer months?"],
            ["Show me vegetation index data for the Great Lakes region"],
        ],
        theme=gr.themes.Soft(),
    )
    logger.info("Gradio interface created successfully")
    return interface

def main():
    """Run the Gradio web interface."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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