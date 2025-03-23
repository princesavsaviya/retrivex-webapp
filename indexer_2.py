import os
import json
from elasticsearch import Elasticsearch, helpers

# Configuration
ES_HOST = "https://localhost:9200"
INDEX_NAME = "medicine_index_2"
DATA_FOLDER = os.path.join("retrivexwebapp",os.path.join("searcher", "Cleaned_Data_2"))  # Folder containing 150 JSON files

# Connect to Elasticsearch
es = Elasticsearch(ES_HOST,verify_certs = False,basic_auth=('elastic','rF*c2rkY=X_tFWudh-j5'))

def create_index():
    """Creates the index with mappings if it does not already exist."""
    if not es.indices.exists(index=INDEX_NAME):
        index_settings = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "english_analyzer": {
                            "type": "standard",
                            "stopwords": "_english_"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "drug_name":           {"type": "text", "analyzer": "english_analyzer"},
                    "drug_details":        {"type": "text", "analyzer": "english_analyzer"},
                    "product_introduction": {"type": "text", "analyzer": "english_analyzer"},
                    "uses_and_benefits":   {"type": "text", "analyzer": "english_analyzer"},
                    "side_effects":        {"type": "text", "analyzer": "english_analyzer"},
                    "how_to_use":          {"type": "text", "analyzer": "english_analyzer"},
                    "how_drug_works":      {"type": "text", "analyzer": "english_analyzer"},
                    "safety_advice":       {"type": "text", "analyzer": "english_analyzer"},
                    "missed_dose":         {"type": "text", "analyzer": "english_analyzer"},
                    "expert_advice":       {"type": "text", "analyzer": "english_analyzer"},
                    "fact_box":            {"type": "text", "analyzer": "english_analyzer"},
                    "drug_interaction":    {"type": "text", "analyzer": "english_analyzer"},
                    "faq":                 {"type": "text", "analyzer": "english_analyzer"}
                }
            }
        }
        print(f"Creating index '{INDEX_NAME}'...")
        es.indices.create(index=INDEX_NAME, body=index_settings)
    else:
        print(f"Index '{INDEX_NAME}' already exists.")

def generate_actions():
    """Generator function that yields bulk index actions for each document."""
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".json"):
            filepath = os.path.join(DATA_FOLDER, filename)
            print(f"Processing file: {filepath}")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    # Assuming each file contains a JSON array of medicine documents
                    documents = json.load(f)
                    for doc in documents:
                        yield {
                            "_index": INDEX_NAME,
                            "_source": doc
                        }
            except json.JSONDecodeError as e:
                print(f"Error reading {filepath}: {e}")

def bulk_index_documents():
    """Bulk index documents from all JSON files in the data folder."""
    # Use the bulk helper with our generator to avoid loading everything into memory
    success, failures = helpers.bulk(es, generate_actions())
    print(f"Indexed {success} documents. Failures: {failures}")

def main():
    create_index()
    bulk_index_documents()

if __name__ == "__main__":
    main()
