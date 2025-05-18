import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# Load environment variables
load_dotenv()


class ElasticTools:
    def __init__(self):
        self.es = Elasticsearch(
            hosts=[f"http://{os.getenv('ELASTIC_HOST', 'localhost')}:{os.getenv('ELASTIC_PORT', '9200')}"],
            basic_auth=(os.getenv('ELASTIC_USERNAME', 'elastic'), os.getenv('ELASTIC_PASSWORD', ''))
        )
    
    def _list_indices(self) -> Dict[str, Any]:
        """List all Elastic indices"""
        try:
            indices = self.es.indices.get_alias().keys()
            return {"status": "success", "indices": list(indices)}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _search_index(self, index: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Search an Elastic index with a query"""
        try:
            result = self.es.search(index=index, body=query)
            return {"status": "success", "results": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _ingest_document(self, index: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest a single document into Elastic"""
        try:
            result = self.es.index(index=index, document=document)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _bulk_ingest(self, index: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bulk ingest documents into Elastic"""
        try:
            actions = [
                {"_index": index, "_source": doc}
                for doc in documents
            ]
            result = self.es.bulk(operations=actions)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "message": str(e)} 