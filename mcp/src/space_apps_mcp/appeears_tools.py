import os
import requests
import json
from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()


class AppEEARSTools:
    def __init__(self):
        self.base_url = os.getenv(
            "APPEEARS_API_URL", "https://appeears.earthdatacloud.nasa.gov/api"
        )
        self.session = requests.Session()
        self.username = os.getenv("APPEEARS_USERNAME", "")
        self.password = os.getenv("APPEEARS_PASSWORD", "")
        self.token = None
        self.token_expiry = None
        self._refresh_token()

    def _refresh_token(self) -> None:
        """Get a new authentication token from the AppEEARS API."""
        try:
            response = requests.post(
                f"{self.base_url}/login",
                auth=(self.username, self.password),
            )
            response.raise_for_status()
            token_data = response.json()

            # Store the token and its expiration
            self.token = token_data["token"]
            # Parse the expiration time from ISO format with timezone
            self.token_expiry = datetime.fromisoformat(
                token_data["expiration"].replace("Z", "+00:00")
            )

            # Set a buffer of 5 minutes before actual expiry
            self.token_expiry = self.token_expiry - timedelta(minutes=5)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise Exception("Invalid AppEEARS credentials")
            raise Exception(f"Failed to get authentication token: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to get authentication token: {str(e)}")

    def _ensure_valid_token(self) -> None:
        """Ensure we have a valid token, refreshing if necessary."""
        if (
            not self.token
            or not self.token_expiry
            or datetime.now(timezone.utc) >= self.token_expiry
        ):
            self._refresh_token()

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make an authenticated request to the AppEEARS API."""
        import logging

        logger = logging.getLogger(__name__)

        self._ensure_valid_token()

        # Add authentication header
        headers = kwargs.pop("headers", {})
        headers.update({"Authorization": f"Bearer {self.token}"})

        url = f"{self.base_url}/{endpoint}"
        logger.info(f"Making {method} request to: {url}")
        logger.debug(f"Request headers: {headers}")
        logger.debug(f"Request kwargs: {kwargs}")

        try:
            response = self.session.request(method, url, headers=headers, **kwargs)
            logger.info(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")

            response.raise_for_status()
            return response
        except Exception as e:
            logger.error(f"Request failed for {method} {url}: {str(e)}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            raise

    def _list_products(self) -> Dict[str, Any]:
        """List all available AppEEARS products"""
        try:
            response = self._make_request("GET", "product")
            return {"status": "success", "products": response.json()}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _get_layers(self, product_and_version: str) -> Dict[str, Any]:
        """List available layers for a given product."""
        try:
            response = self._make_request("GET", f"product/{product_and_version}")
            layers = response.json()
            if not layers:
                return {"status": "error", "message": "No layers found for this product."}

            layer_info = {k: v.get("Description", "") for k, v in layers.items()}
            return {"status": "success", "layers": layer_info}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _submit_point_request(
        self,
        layers: List[Dict[str, str]],
        locations: List[Dict[str, Any]],
        start_date: str,
        end_date: str,
        task_name: str = "LlamaAgentTask",
    ) -> Dict[str, Any]:
        """
        Submit a point request to AppEEARS with multiple layers and locations.

        Args:
            layers (List[Dict[str, str]): List of layer names to request. Each dict should have two keys: layer and product.
            locations (List[Dict[str, Any]]): List of locations to request.  Each dict should have keys: id, category, latitude, and longitude.
            start_date (str): Start date for the request
            end_date (str): End date for the request
            task_name (str): Name of the task to submit
        """
        try:
            # Format dates to MM/DD/YYYY
            start_date = datetime.strptime(start_date, "%Y-%m-%d").strftime("%m-%d-%Y")
            end_date = datetime.strptime(end_date, "%Y-%m-%d").strftime("%m-%d-%Y")

            task = {
                "task_type": "point",
                "task_name": task_name,
                "params": {
                    "dates": [{"startDate": start_date, "endDate": end_date}],
                    "layers": layers,
                    "coordinates": locations,
                },
            }
            print(f"Submitting task: {json.dumps(task, indent=2)}")
            response = self._make_request("POST", "task", json=task)
            task_id = response.json()["task_id"]
            return {
                "status": "success",
                "task_id": task_id,
                "message": f"Task submitted! Task ID: {task_id}",
            }
        except Exception as e:
            print(f"Error submitting task: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of an AppEEARS task"""
        try:
            response = self._make_request("GET", f"task/{task_id}")
            return {"status": "success", "task_status": response.json()}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _list_bundle_files(self, task_id: str) -> Dict[str, Any]:
        """
        List files available in a bundle for a completed task.

        This is part of the AppEEARS Bundle API workflow. After a task is completed,
        the results are stored in a bundle. This function lists all files available
        in that bundle.

        Args:
            task_id: The task identifier

        Returns:
            Dictionary with list of files in the bundle
        """
        import logging

        logger = logging.getLogger(__name__)

        try:
            logger.info(f"Listing bundle files for task_id: {task_id}")
            response = self._make_request("GET", f"bundle/{task_id}")
            files = response.json()
            logger.info(f"Bundle files response type for task_id {task_id}: {type(files)}")
            logger.info(f"Bundle files response for task_id {task_id}: {files}")

            # Log the structure if it's a dict
            if isinstance(files, dict):
                logger.info(f"Bundle response keys: {list(files.keys())}")

            return {"status": "success", "files": files}
        except Exception as e:
            logger.error(
                f"Error listing bundle files for task_id {task_id}: {str(e)}", exc_info=True
            )
            return {"status": "error", "message": str(e)}

    def _download_task(self, task_id: str, output_path: str = None) -> Dict[str, Any]:
        """
        Download all results from a completed AppEEARS task using the bundle API.

        This function implements the complete AppEEARS Bundle API workflow:
        1. Check if task is completed
        2. List files in bundle using /bundle/{task_id}
        3. Download all available files using /bundle/{task_id}/file/{file_id}

        Args:
            task_id: The task identifier
            output_path: Path where to save the downloaded files (will create a folder)
                        If None, uses the DOWNLOAD_PATH environment variable or /tmp

        Returns:
            Dictionary with download status and folder information
        """
        import logging

        logger = logging.getLogger(__name__)

        # Use configured download path if output_path is not provided
        if output_path is None:
            output_path = os.getenv("DOWNLOAD_PATH", "/tmp")
            logger.info(f"No output_path provided, using DOWNLOAD_PATH: {output_path}")

        try:
            logger.info(f"Starting download for task_id: {task_id}, output_path: {output_path}")

            # First check if task is complete
            logger.info(f"Checking task status for task_id: {task_id}")
            status_response = self._make_request("GET", f"task/{task_id}")
            task_status = status_response.json()
            logger.info(f"Task status response: {task_status}")

            if task_status.get("status") != "done":
                logger.error(f"Task {task_id} is not complete. Status: {task_status.get('status')}")
                return {
                    "status": "error",
                    "message": f"Task is not complete. Current status: {task_status.get('status')}",
                }

            logger.info(f"Task {task_id} is complete, proceeding with download")

            # List files in the bundle
            logger.info(f"Listing files in bundle for task_id: {task_id}")
            bundle_response = self._make_request("GET", f"bundle/{task_id}")
            bundle_files = bundle_response.json()
            logger.info(f"Bundle files response type: {type(bundle_files)}")
            logger.info(f"Bundle files response: {bundle_files}")

            # If bundle_files is not a list, try to extract files from the response
            if not isinstance(bundle_files, list):
                logger.warning(f"Bundle files response is not a list: {type(bundle_files)}")
                if isinstance(bundle_files, dict):
                    # Try to find files in the response
                    if "files" in bundle_files:
                        bundle_files = bundle_files["files"]
                        logger.info(f"Extracted files from response: {bundle_files}")
                    elif "data" in bundle_files:
                        bundle_files = bundle_files["data"]
                        logger.info(f"Extracted data from response: {bundle_files}")
                    else:
                        logger.error(f"Could not find files in bundle response: {bundle_files}")
                        return {
                            "status": "error",
                            "message": f"Unexpected bundle response format: {bundle_files}",
                        }
                else:
                    logger.error(
                        f"Bundle files response is not a list or dict: {type(bundle_files)}"
                    )
                    return {
                        "status": "error",
                        "message": f"Unexpected bundle response type: {type(bundle_files)}",
                    }

            if not bundle_files:
                logger.error(f"No files found in bundle for task_id: {task_id}")
                return {"status": "error", "message": "No files found in bundle"}

            logger.info(f"Found {len(bundle_files)} files in bundle for task_id: {task_id}")

            # Create output directory for this task
            if os.path.isdir(output_path):
                # If output_path is a directory, create a subdirectory for this task
                task_folder = os.path.join(output_path, f"task_{task_id}")
            else:
                # If output_path is a file path, use its directory and create task subfolder
                task_folder = os.path.join(os.path.dirname(output_path), f"task_{task_id}")

            logger.info(f"Creating task folder: {task_folder}")

            # Ensure the task directory exists
            os.makedirs(task_folder, exist_ok=True)
            logger.info(f"Task folder created successfully: {task_folder}")

            # Download all files in the bundle
            downloaded_files = []
            for i, file_info in enumerate(bundle_files):
                logger.info(
                    f"Processing file {i + 1}/{len(bundle_files)}: {file_info} (type: {type(file_info)})"
                )

                # Handle different response formats
                if isinstance(file_info, dict):
                    # Dictionary format
                    file_id = file_info.get("file_id")
                    file_name = file_info.get("file_name", f"unknown_file_{file_id}")
                elif isinstance(file_info, str):
                    # String format - might be the file_id directly
                    file_id = file_info
                    file_name = f"file_{file_id}"
                else:
                    logger.warning(
                        f"Unexpected file_info format: {file_info} (type: {type(file_info)})"
                    )
                    continue

                if not file_id:
                    logger.warning(f"No file_id found in file_info: {file_info}")
                    continue

                logger.info(f"Downloading file_id: {file_id}, file_name: {file_name}")

                try:
                    # Download the file using correct API endpoint
                    download_url = f"bundle/{task_id}/{file_id}"
                    logger.info(f"Making download request to: {download_url}")
                    download_response = self._make_request("GET", download_url, stream=True)
                    logger.info(f"Download response status: {download_response.status_code}")

                    # Save to task folder
                    file_path = os.path.join(task_folder, file_name)
                    logger.info(f"Saving file to: {file_path}")

                    file_size = 0
                    with open(file_path, "wb") as f:
                        for chunk in download_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            file_size += len(chunk)

                    logger.info(
                        f"Successfully downloaded file: {file_name}, size: {file_size} bytes"
                    )

                    downloaded_files.append(
                        {
                            "file_name": file_name,
                            "file_path": file_path,
                            "file_id": file_id,
                            "file_size": file_size,
                        }
                    )

                except Exception as file_error:
                    logger.error(
                        f"Error downloading file {file_name} (file_id: {file_id}): {str(file_error)}"
                    )
                    # Continue with other files even if one fails
                    continue

            if not downloaded_files:
                logger.error(f"Failed to download any files for task_id: {task_id}")
                return {"status": "error", "message": "Failed to download any files"}

            total_size = sum(f.get("file_size", 0) for f in downloaded_files)
            logger.info(
                f"Successfully downloaded {len(downloaded_files)} files for task_id: {task_id}, total size: {total_size} bytes"
            )

            return {
                "status": "success",
                "message": f"Downloaded {len(downloaded_files)} files to {task_folder}",
                "task_id": task_id,
                "download_folder": task_folder,
                "files": downloaded_files,
                "file_count": len(downloaded_files),
                "total_size": total_size,
            }
        except Exception as e:
            logger.error(f"Error in _download_task for task_id {task_id}: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"Download error: {str(e)}"}
