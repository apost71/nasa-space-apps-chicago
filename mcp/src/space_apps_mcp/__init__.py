"""
MCP Server for Elastic and NASA AppEEARS integration.
"""

__version__ = "0.1.0"

from .server import mcp
from .elastic_tools import ElasticTools
from .appeears_tools import AppEEARSTools

# Expose the server instance for mcp dev
__all__ = ["mcp", "ElasticTools", "AppEEARSTools"] 