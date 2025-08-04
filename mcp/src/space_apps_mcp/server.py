"""MCP server implementation using FastMCP."""

import logging
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from fastmcp import FastMCP
from space_apps_mcp.elastic_tools import ElasticTools
from space_apps_mcp.appeears_tools import AppEEARSTools
from space_apps_mcp.job_tools import (
    submit_appears_job,
    check_job_status,
    download_job_results,
    list_bundle_files,
    list_appears_jobs,
    get_job_details,
    cancel_appears_job,
    get_job_progress,
)

load_dotenv()

# Initialize FastMCP server
mcp = FastMCP(
    name="SpaceAppsMCP",
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
    task_name: str = "LlamaAgentTask",
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
    """
    Download results from a completed NASA AppEEARS task.

    Args:
        task_id: The ID of the task to download
        output_path: The path to save the downloaded results

    Returns:
        Dictionary with download status and information
    """
    return appears_tools._download_task(task_id, output_path)


# Register Job Management tools
@mcp.tool()
def submit_appears_job_tool(
    layers: List[Dict[str, str]],
    locations: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
    task_name: str = "AgentTask",
) -> Dict[str, Any]:
    """
    Submit an AppEEARS job and return the job ID for tracking.

    This tool submits a job to AppEEARS and tracks it for status monitoring.
    Use this when you need to process data that may take a long time to complete.

    IMPORTANT: This is the first step in the AppEEARS Bundle API workflow:
    1. Submit job with this tool
    2. Monitor with check_job_status_tool() until "completed"
    3. List files with list_bundle_files_tool() to see what's available
    4. Download files with download_job_results_tool()

    Args:
        layers: List of layer configurations, each with 'layer' and 'product' keys
        locations: List of location configurations, each with 'id', 'category', 'latitude', 'longitude'
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        task_name: Name for the task (default: "AgentTask")

    Returns:
        Dictionary with job_id and status information for tracking
    """
    return submit_appears_job(layers, locations, start_date, end_date, task_name)


@mcp.tool()
def check_job_status_tool(job_id: str) -> Dict[str, Any]:
    """
    Check the status of an AppEEARS job.

    Use this tool to monitor the progress of a long-running job.
    The job must have been submitted using submit_appears_job_tool.

    IMPORTANT: This is part of the AppEEARS Bundle API workflow:
    - Monitor job status until it becomes "completed"
    - Once completed, you can use list_bundle_files_tool() to see available files
    - Then use download_job_results_tool() to download the results

    Args:
        job_id: The job identifier returned from submit_appears_job_tool

    Returns:
        Dictionary with current job status and information
    """
    return check_job_status(job_id)


@mcp.tool()
def download_job_results_tool(job_id: str, output_path: str = None) -> Dict[str, Any]:
    """
    Download all results for a completed AppEEARS job using the Bundle API.

    IMPORTANT: AppEEARS uses a Bundle API for file downloads. This tool:
    1. Checks if the job is completed
    2. Lists files in bundle using /bundle/{task_id}
    3. Downloads ALL available files using /bundle/{task_id}/file/{file_id}
    4. Creates a folder for the task and saves all files there

    Workflow:
    - First use check_job_status_tool() to ensure job is "completed"
    - Optionally use list_bundle_files_tool() to see available files
    - Then use this tool to download all results

    Args:
        job_id: The job identifier returned from submit_appears_job_tool
        output_path: Path where to save the results (will create a folder named task_{job_id})
                  If None, uses the DOWNLOAD_PATH environment variable or /tmp

    Returns:
        Dictionary with download status and folder information including all downloaded files
    """
    return download_job_results(job_id, output_path)


@mcp.tool()
def list_bundle_files_tool(job_id: str) -> Dict[str, Any]:
    """
    List files available in a bundle for a completed AppEEARS job.

    IMPORTANT: This is part of the AppEEARS Bundle API workflow. Before downloading files,
    you need to know what files are available in the bundle. This tool:
    1. Checks if the job is completed
    2. Lists all files in the bundle using /bundle/{task_id}

    Typical workflow:
    1. Submit job with submit_appears_job_tool()
    2. Monitor with check_job_status_tool() until "completed"
    3. List files with this tool to see what's available
    4. Download files with download_job_results_tool()

    Args:
        job_id: The job identifier returned from submit_appears_job_tool

    Returns:
        Dictionary with list of files in the bundle including file IDs and names
    """
    return list_bundle_files(job_id)


@mcp.tool()
def list_appears_jobs_tool(limit: int = None, offset: int = None) -> Dict[str, Any]:
    """
    List all AppEEARS jobs using the API.

    Use this tool to see all jobs that have been submitted to AppEEARS and their current status.
    This uses the AppEEARS API as the source of truth.

    Args:
        limit: Maximum number of jobs to return (optional)
        offset: Number of jobs to skip for pagination (optional)

    Returns:
        Dictionary with list of jobs from AppEEARS API
    """
    return list_appears_jobs(limit, offset)


@mcp.tool()
def get_job_details_tool(job_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific AppEEARS job.

    Use this tool to get comprehensive information about a job including parameters, progress, and metadata.

    Args:
        job_id: The job identifier returned from submit_appears_job_tool

    Returns:
        Dictionary with detailed job information
    """
    return get_job_details(job_id)


@mcp.tool()
def cancel_appears_job_tool(job_id: str) -> Dict[str, Any]:
    """
    Cancel an AppEEARS job.

    Use this tool to cancel a job that is still pending or running.
    Note: AppEEARS API may not support job cancellation directly.

    Args:
        job_id: The job identifier returned from submit_appears_job_tool

    Returns:
        Dictionary with cancellation status
    """
    return cancel_appears_job(job_id)


@mcp.tool()
def get_job_progress_tool(job_id: str) -> Dict[str, Any]:
    """
    Get detailed progress information for a running job.

    Use this tool to get progress information including elapsed time and current status.

    Args:
        job_id: The job identifier returned from submit_appears_job_tool

    Returns:
        Dictionary with progress information
    """
    return get_job_progress(job_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
