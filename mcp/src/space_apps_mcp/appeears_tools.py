import os
import requests
import json
from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
load_dotenv()

class AppEEARSTools:
    def __init__(self):
        self.base_url = os.getenv('APPEEARS_API_URL', 'https://appeears.earthdatacloud.nasa.gov/api')
        self.session = requests.Session()
        self.username = os.getenv('APPEEARS_USERNAME', '')
        self.password = os.getenv('APPEEARS_PASSWORD', '')
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
            self.token = token_data['token']
            # Parse the expiration time from ISO format with timezone
            self.token_expiry = datetime.fromisoformat(token_data['expiration'].replace('Z', '+00:00'))
            
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
        if not self.token or not self.token_expiry or datetime.now(timezone.utc) >= self.token_expiry:
            self._refresh_token()

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make an authenticated request to the AppEEARS API."""
        self._ensure_valid_token()
        
        # Add authentication header
        headers = kwargs.pop('headers', {})
        headers.update({
            'Authorization': f"Bearer {self.token}"
        })
        
        response = self.session.request(
            method,
            f"{self.base_url}/{endpoint}",
            headers=headers,
            **kwargs
        )
        response.raise_for_status()
        return response

    def _list_products(self) -> Dict[str, Any]:
        """List all available AppEEARS products"""
        try:
            response = self._make_request('GET', 'product')
            return {"status": "success", "products": response.json()}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _get_layers(self, product_and_version: str) -> Dict[str, Any]:
        """List available layers for a given product."""
        try:
            response = self._make_request('GET', f'product/{product_and_version}')
            layers = response.json()
            if not layers:
                return {"status": "error", "message": "No layers found for this product."}
            
            layer_info = {k: v.get('Description', '') for k, v in layers.items()}
            return {"status": "success", "layers": layer_info}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _submit_point_request(self, layers: List[Dict[str, str]], locations: List[Dict[str, Any]], 
                            start_date: str, end_date: str, task_name: str = "LlamaAgentTask") -> Dict[str, Any]:
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
                    "coordinates": locations
                }
            }
            print(f"Submitting task: {json.dumps(task, indent=2)}")
            response = self._make_request('POST', 'task', json=task)
            task_id = response.json()['task_id']
            return {"status": "success", "task_id": task_id, "message": f"Task submitted! Task ID: {task_id}"}
        except Exception as e:
            print(f"Error submitting task: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of an AppEEARS task"""
        try:
            response = self._make_request('GET', f'task/{task_id}')
            return {"status": "success", "task_status": response.json()}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _download_task(self, task_id: str, output_path: str) -> Dict[str, Any]:
        """Download results from a completed AppEEARS task"""
        try:
            # First get the download URL
            status_response = self._make_request('GET', f'task/{task_id}')
            task_status = status_response.json()
            
            if task_status.get('status') != 'done':
                return {"status": "error", "message": "Task is not complete"}
            
            # Download the results
            download_url = task_status.get('download_url')
            if not download_url:
                return {"status": "error", "message": "No download URL available"}
            
            response = self._make_request('GET', download_url, stream=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return {"status": "success", "message": f"Downloaded to {output_path}"}
        except Exception as e:
            return {"status": "error", "message": str(e)} 