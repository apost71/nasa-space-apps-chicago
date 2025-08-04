"""
Job management tools for the MCP server that follow the agentic pattern.
These tools use the AppEEARS API as the source of truth for job management.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import requests

from .appeears_tools import AppEEARSTools

logger = logging.getLogger(__name__)

# Initialize AppEEARS tools
appears_tools = AppEEARSTools()


def submit_appears_job(
    layers: List[Dict[str, str]],
    locations: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
    task_name: str = "AgentTask",
) -> Dict[str, Any]:
    """
    Submit an AppEEARS job and return the job ID.

    Args:
        layers: List of layer configurations, each with 'layer' and 'product' keys
        locations: List of location configurations, each with 'id', 'category', 'latitude', 'longitude'
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        task_name: Name for the task

    Returns:
        Dictionary with job_id and status information
    """
    logger.info(
        f"submit_appears_job called with task_name: {task_name}, start_date: {start_date}, end_date: {end_date}"
    )
    try:
        # Submit the job using existing AppEEARS tools
        result = appears_tools._submit_point_request(
            layers=layers,
            locations=locations,
            start_date=start_date,
            end_date=end_date,
            task_name=task_name,
        )

        if result["status"] == "success":
            job_id = result["task_id"]
            logger.info(f"submit_appears_job succeeded - job_id: {job_id}")

            return {
                "status": "success",
                "job_id": job_id,
                "message": f"AppEEARS job submitted successfully. Job ID: {job_id}",
                "task_name": task_name,
                "layers": layers,
                "locations": locations,
                "start_date": start_date,
                "end_date": end_date,
            }
        else:
            logger.error(f"submit_appears_job failed: {result['message']}")
            return {
                "status": "error",
                "message": f"Failed to submit AppEEARS job: {result['message']}",
            }

    except Exception as e:
        logger.error(f"Error submitting AppEEARS job: {str(e)}")
        return {"status": "error", "message": f"Error submitting job: {str(e)}"}


def check_job_status(job_id: str) -> Dict[str, Any]:
    """
    Check the status of an AppEEARS job using the API.

    Args:
        job_id: The job identifier

    Returns:
        Dictionary with current job status and information
    """
    logger.info(f"check_job_status called with job_id: {job_id}")
    try:
        # Get status directly from AppEEARS API
        result = appears_tools._get_task_status(job_id)

        if result["status"] == "success":
            task_status = result["task_status"]
            api_status = task_status.get("status", "unknown")

            # Map API status to our internal status
            status_mapping = {
                "pending": "pending",
                "running": "running",
                "done": "completed",
                "failed": "failed",
                "cancelled": "cancelled",
            }

            job_status = status_mapping.get(api_status.lower(), "pending")

            # Calculate elapsed time if we have creation time
            elapsed_time = None
            if "created" in task_status:
                try:
                    created_time = datetime.fromisoformat(
                        task_status["created"].replace("Z", "+00:00")
                    )
                    elapsed_time = (
                        datetime.now(created_time.tzinfo) - created_time
                    ).total_seconds()
                except:
                    pass

            logger.info(
                f"check_job_status succeeded - job_id: {job_id}, status: {job_status}, api_status: {api_status}"
            )

            return {
                "status": "success",
                "job_id": job_id,
                "job_status": job_status,
                "api_status": api_status,
                "elapsed_time": elapsed_time,
                "task_info": {
                    "task_name": task_status.get("task_name"),
                    "created": task_status.get("created"),
                    "updated": task_status.get("updated"),
                    "progress": task_status.get("progress"),
                    "message": task_status.get("message"),
                },
            }
        else:
            logger.error(f"check_job_status failed - job_id: {job_id}, error: {result['message']}")
            return {
                "status": "error",
                "message": f"Failed to check job status: {result['message']}",
                "job_id": job_id,
            }

    except Exception as e:
        logger.error(f"Error checking job status: {str(e)}")
        return {"status": "error", "message": f"Error checking job status: {str(e)}"}


def download_job_results(job_id: str, output_path: str = None) -> Dict[str, Any]:
    """
    Download all results for a completed AppEEARS job using the Bundle API.

    IMPORTANT: AppEEARS uses a Bundle API for file downloads. This function:
    1. Checks if the job is completed
    2. Lists files in bundle using /bundle/{task_id}
    3. Downloads ALL available files using /bundle/{task_id}/file/{file_id}
    4. Creates a folder for the task and saves all files there

    Workflow:
    - First use check_job_status() to ensure job is "completed"
    - Optionally use list_bundle_files() to see available files
    - Then use this function to download all results

    Args:
        job_id: The job identifier
        output_path: Path where to save the results (will create a folder named task_{job_id})
                  If None, uses the DOWNLOAD_PATH environment variable or /tmp

    Returns:
        Dictionary with download status and folder information including all downloaded files
    """
    # Use configured download path if output_path is not provided
    if output_path is None:
        output_path = os.getenv("DOWNLOAD_PATH", "/tmp")
        logger.info(f"No output_path provided, using DOWNLOAD_PATH: {output_path}")

    logger.info(f"download_job_results called with job_id: {job_id}, output_path: {output_path}")
    try:
        # First check if job is completed
        status_result = check_job_status(job_id)
        if status_result["status"] != "success":
            return {
                "status": "error",
                "message": f"Could not check job status: {status_result['message']}",
            }

        if status_result["job_status"] != "completed":
            return {
                "status": "error",
                "message": f"Job {job_id} is not completed (current status: {status_result['job_status']})",
            }

        # Download the results
        logger.info(f"Calling _download_task for job_id: {job_id}, output_path: {output_path}")
        download_result = appears_tools._download_task(job_id, output_path)
        logger.info(f"_download_task result: {download_result}")

        if download_result["status"] == "success":
            download_folder = download_result["download_folder"]
            file_count = download_result["file_count"]
            total_size = download_result.get("total_size", 0)
            logger.info(
                f"download_job_results succeeded - job_id: {job_id}, download_folder: {download_folder}, file_count: {file_count}, total_size: {total_size}"
            )
            return {
                "status": "success",
                "job_id": job_id,
                "message": f"Results downloaded successfully to {download_folder}",
                "download_folder": download_folder,
                "file_count": file_count,
                "total_size": total_size,
                "files": download_result["files"],
            }
        else:
            logger.error(
                f"download_job_results failed - job_id: {job_id}, error: {download_result['message']}"
            )
            return {
                "status": "error",
                "message": f"Failed to download results: {download_result['message']}",
            }

    except Exception as e:
        logger.error(f"Error downloading job results: {str(e)}")
        return {"status": "error", "message": f"Error downloading results: {str(e)}"}


def list_bundle_files(job_id: str) -> Dict[str, Any]:
    """
    List files available in a bundle for a completed AppEEARS job.

    IMPORTANT: This is part of the AppEEARS Bundle API workflow. Before downloading files,
    you need to know what files are available in the bundle. This function:
    1. Checks if the job is completed
    2. Lists all files in the bundle using /bundle/{task_id}

    Typical workflow:
    1. Submit job with submit_appears_job()
    2. Monitor with check_job_status() until "completed"
    3. List files with this function to see what's available
    4. Download files with download_job_results()

    Args:
        job_id: The job identifier

    Returns:
        Dictionary with list of files in the bundle including file IDs and names
    """
    logger.info(f"list_bundle_files called with job_id: {job_id}")
    try:
        # First check if job is completed
        status_result = check_job_status(job_id)
        if status_result["status"] != "success":
            return {
                "status": "error",
                "message": f"Could not check job status: {status_result['message']}",
            }

        if status_result["job_status"] != "completed":
            return {
                "status": "error",
                "message": f"Job {job_id} is not completed (current status: {status_result['job_status']})",
            }

        # Get the bundle files
        logger.info(f"Calling _list_bundle_files for job_id: {job_id}")
        bundle_result = appears_tools._list_bundle_files(job_id)
        logger.info(f"_list_bundle_files result: {bundle_result}")

        if bundle_result["status"] == "success":
            files = bundle_result["files"]
            logger.info(f"list_bundle_files succeeded - job_id: {job_id}, file_count: {len(files)}")
            return {
                "status": "success",
                "job_id": job_id,
                "files": files,
                "file_count": len(files),
                "message": f"Found {len(files)} files in bundle for job {job_id}",
            }
        else:
            logger.error(
                f"list_bundle_files failed - job_id: {job_id}, error: {bundle_result['message']}"
            )
            return {
                "status": "error",
                "message": f"Failed to list bundle files: {bundle_result['message']}",
            }

    except Exception as e:
        logger.error(f"Error listing bundle files: {str(e)}")
        return {"status": "error", "message": f"Error listing bundle files: {str(e)}"}


def list_appears_jobs(limit: Optional[int] = None, offset: Optional[int] = None) -> Dict[str, Any]:
    """
    List all AppEEARS jobs using the API.

    Args:
        limit: Maximum number of jobs to return (optional)
        offset: Number of jobs to skip (optional)

    Returns:
        Dictionary with list of jobs from AppEEARS API
    """
    logger.info(f"list_appears_jobs called with limit: {limit}, offset: {offset}")
    try:
        # Build query parameters
        params = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        # Make request to AppEEARS API
        response = appears_tools._make_request("GET", "task", params=params)
        jobs = response.json()

        # Process and format the jobs
        formatted_jobs = []
        for job in jobs:
            # Map API status to our internal status
            api_status = job.get("status", "unknown")
            status_mapping = {
                "pending": "pending",
                "running": "running",
                "done": "completed",
                "failed": "failed",
                "cancelled": "cancelled",
            }
            job_status = status_mapping.get(api_status.lower(), "pending")

            formatted_job = {
                "job_id": job.get("task_id"),
                "task_name": job.get("task_name"),
                "status": job_status,
                "api_status": api_status,
                "created": job.get("created"),
                "updated": job.get("updated"),
                "progress": job.get("progress"),
                "message": job.get("message"),
            }
            formatted_jobs.append(formatted_job)

        logger.info(f"list_appears_jobs succeeded - found {len(formatted_jobs)} jobs")
        return {
            "status": "success",
            "jobs": formatted_jobs,
            "total_jobs": len(formatted_jobs),
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        logger.error(f"Error listing AppEEARS jobs: {str(e)}")
        return {"status": "error", "message": f"Error listing jobs: {str(e)}"}


def get_job_details(job_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific AppEEARS job.

    Args:
        job_id: The job identifier

    Returns:
        Dictionary with detailed job information
    """
    logger.info(f"get_job_details called with job_id: {job_id}")
    try:
        # Get task status which includes detailed information
        result = appears_tools._get_task_status(job_id)

        if result["status"] == "success":
            task_status = result["task_status"]

            # Map API status to our internal status
            api_status = task_status.get("status", "unknown")
            status_mapping = {
                "pending": "pending",
                "running": "running",
                "done": "completed",
                "failed": "failed",
                "cancelled": "cancelled",
            }
            job_status = status_mapping.get(api_status.lower(), "pending")

            logger.info(f"get_job_details succeeded - job_id: {job_id}, status: {job_status}")
            return {
                "status": "success",
                "job_id": job_id,
                "job_status": job_status,
                "api_status": api_status,
                "task_name": task_status.get("task_name"),
                "created": task_status.get("created"),
                "updated": task_status.get("updated"),
                "progress": task_status.get("progress"),
                "message": task_status.get("message"),
                "task_type": task_status.get("task_type"),
                "parameters": task_status.get("params"),
                "download_url": task_status.get("download_url"),
                "full_response": task_status,
            }
        else:
            logger.error(f"get_job_details failed - job_id: {job_id}, error: {result['message']}")
            return {
                "status": "error",
                "message": f"Failed to get job details: {result['message']}",
                "job_id": job_id,
            }

    except Exception as e:
        logger.error(f"Error getting job details: {str(e)}")
        return {"status": "error", "message": f"Error getting job details: {str(e)}"}


def cancel_appears_job(job_id: str) -> Dict[str, Any]:
    """
    Cancel an AppEEARS job.

    Note: AppEEARS API may not support job cancellation directly.
    This function will attempt to cancel the job if the API supports it.

    Args:
        job_id: The job identifier

    Returns:
        Dictionary with cancellation status
    """
    logger.info(f"cancel_appears_job called with job_id: {job_id}")
    try:
        # First check if job exists and can be cancelled
        status_result = check_job_status(job_id)
        if status_result["status"] != "success":
            return {
                "status": "error",
                "message": f"Could not check job status: {status_result['message']}",
            }

        job_status = status_result["job_status"]
        if job_status in ["completed", "failed", "cancelled"]:
            return {
                "status": "error",
                "message": f"Cannot cancel job {job_id} - it is already {job_status}",
            }

        # Try to cancel the job via API
        try:
            response = appears_tools._make_request("DELETE", f"task/{job_id}")
            logger.info(f"cancel_appears_job succeeded - job_id: {job_id}")
            return {
                "status": "success",
                "job_id": job_id,
                "message": f"Job {job_id} cancelled successfully",
                "job_status": "cancelled",
            }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 405:
                logger.warning(
                    f"cancel_appears_job not supported - job_id: {job_id}, API returned 405"
                )
                return {
                    "status": "error",
                    "message": "Job cancellation is not supported by the AppEEARS API",
                    "job_id": job_id,
                }
            else:
                logger.error(f"cancel_appears_job failed - job_id: {job_id}, HTTP error: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Failed to cancel job: {str(e)}",
                    "job_id": job_id,
                }

    except Exception as e:
        logger.error(f"Error cancelling job: {str(e)}")
        return {"status": "error", "message": f"Error cancelling job: {str(e)}"}


def get_job_progress(job_id: str) -> Dict[str, Any]:
    """
    Get detailed progress information for a running job.

    Args:
        job_id: The job identifier

    Returns:
        Dictionary with progress information
    """
    logger.info(f"get_job_progress called with job_id: {job_id}")
    try:
        # Get job details which includes progress
        details = get_job_details(job_id)

        if details["status"] != "success":
            return details

        progress_info = {
            "status": "success",
            "job_id": job_id,
            "job_status": details["job_status"],
            "progress": details.get("progress"),
            "message": details.get("message"),
            "created": details.get("created"),
            "updated": details.get("updated"),
        }

        # Calculate elapsed time if we have creation time
        if details.get("created"):
            try:
                created_time = datetime.fromisoformat(details["created"].replace("Z", "+00:00"))
                elapsed_time = (datetime.now(created_time.tzinfo) - created_time).total_seconds()
                progress_info["elapsed_time_seconds"] = elapsed_time
                progress_info["elapsed_time_formatted"] = f"{elapsed_time:.0f} seconds"
            except:
                pass

        logger.info(
            f"get_job_progress succeeded - job_id: {job_id}, status: {progress_info['job_status']}"
        )
        return progress_info

    except Exception as e:
        logger.error(f"get_job_progress failed - job_id: {job_id}, error: {str(e)}")
        return {"status": "error", "message": f"Error getting job progress: {str(e)}"}
