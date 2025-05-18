"""MCP server implementation using FastMCP."""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from fastmcp import FastMCP
from space_apps_mcp.elastic_tools import ElasticTools
from space_apps_mcp.appeears_tools import AppEEARSTools

load_dotenv()

# Initialize FastMCP server
mcp = FastMCP(
    name="SpaceAppsMCP",
    description="MCP server for NASA Space Apps Chicago data exploration tools",
    version="0.1.0",
    dependencies=[
        "elasticsearch>=8.0.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "space-apps-mcp"
    ]
)

# Initialize tools
elastic_tools = ElasticTools()
appears_tools = AppEEARSTools()

# Register Elastic tools
@mcp.tool()
def list_elastic_indices() -> Dict[str, Any]:
    """List all available Elasticsearch indices."""
    return elastic_tools._list_indices()

@mcp.tool()
def search_elastic_index(index: str, query: Dict[str, Any]) -> Dict[str, Any]:
    """Search an Elasticsearch index with a query."""
    return elastic_tools._search_index(index, query)

@mcp.tool()
def ingest_elastic_document(index: str, document: Dict[str, Any]) -> Dict[str, Any]:
    """Ingest a single document into Elasticsearch."""
    return elastic_tools._ingest_document(index, document)

@mcp.tool()
def bulk_ingest_elastic(index: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Bulk ingest multiple documents into Elasticsearch."""
    return elastic_tools._bulk_ingest(index, documents)

# Register AppEEARS tools
@mcp.tool()
def list_appears_products() -> Dict[str, Any]:
    """List all available AppEEARS products."""
    return appears_tools._list_products()

@mcp.tool()
def get_appears_layers(product_and_version: str) -> Dict[str, Any]:
    """List available layers for a given AppEEARS product."""
    return appears_tools._get_layers(product_and_version)

@mcp.tool()
def submit_appears_point_request(
    layers: List[Dict[str, str]],
    locations: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
    task_name: str = "LlamaAgentTask"
) -> Dict[str, Any]:
        """
        Submit a point request to AppEEARS with multiple layers and locations.

        Args:
            layers (List[Dict[str, str]): List of layer names to request. Each dict should have two keys: layer and product. The product should be the specific product identifier and version in the format "PRODUCT.VERSION" from the AppEEARS product list.
            locations (List[Dict[str, Any]]): List of locations to request.  Each dict should have keys: id, category, latitude, and longitude.
            start_date (str): Start date for the request
            end_date (str): End date for the request
            task_name (str): Name of the task to submit
        """    
        return appears_tools._submit_point_request(layers, locations, start_date, end_date, task_name)

@mcp.tool()
def get_appears_task_status(task_id: str) -> Dict[str, Any]:
    """Get the status of a NASA AppEEARS task."""
    return appears_tools._get_task_status(task_id)

@mcp.tool()
def download_appears_task(task_id: str, output_path: str) -> Dict[str, Any]:
    """Download results from a completed NASA AppEEARS task."""
    return appears_tools._download_task(task_id, output_path)

if __name__ == "__main__":
    mcp.run()