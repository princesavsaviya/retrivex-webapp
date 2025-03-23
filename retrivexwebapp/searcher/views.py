# searcher/views.py

from django.shortcuts import render
from django.core.paginator import Paginator
from django.core.cache import cache

# Sparse index retriever (Elasticsearch)
from .retriever_2 import MedicineSearcher  

# Dense index retriever (FAISS + embeddings)
from .embeded_search import DenseHybridSearch  

import os
import glob
import json
import re
from django.shortcuts import render
from django.http import Http404


def home_view(request):
    """
    Renders the home page with a search box and two checkboxes.
    """
    return render(request, 'home.html')

def search_view(request):
    
    query = request.GET.get('q', '')
    index_type = request.GET.get('index_type', 'sparse')  # 'sparse' or 'dense'
    
    page_param = request.GET.get('page', 1)
    try:
        page = int(page_param)
    except ValueError:
        page = 1

    # Create a cache key that uniquely identifies this search request.
    cache_key = f"search_results:{index_type}:{query}"
    # Try to fetch the result list from cache.
    results = cache.get(cache_key)
    if results is None:
        # If not cached, run the search based on index_type.
        if index_type == 'dense':
            embedding_system = DenseHybridSearch()
            # Use aggregated hybrid search which returns a list of results with aggregated_score, drug_name, and original_data.
            aggregated_results = embedding_system.hybrid_search(query, top_k=50)
            results = []
            for res in aggregated_results:
                doc = {
                    "drug_name": res.get("drug_name", "Unknown"),
                    # Use the 'product_introduction' field from original_data if available.
                    "product_introduction": res.get("original_data", {}).get("product_introduction", "No product introduction available."),
                    "Score": res.get("aggregated_score", 0),
                    "drug_id": compute_drug_id(res.get("original_data", {}))
                }
                results.append(doc)
        else:
            searcher = MedicineSearcher()
            temp_results = searcher.basic_search(query, field="product_introduction", max_results=50)
            results=[]
            for result in temp_results:
                # Extract the relevant fields from the result.
                doc = {
                    "drug_name": result.get("drug_name", "Unknown"),
                    "product_introduction": result.get("product_introduction", "No product introduction available."),
                    "Score": result.get("score", 0),
                    "drug_id": compute_drug_id(result)
                }
                results.append(doc)
        # Cache the full results list for 5 minutes (300 seconds)
        cache.set(cache_key, results, timeout=300)
    
    # Paginate the cached results (10 results per page)
    paginator = Paginator(results, 10)
    page_obj = paginator.get_page(page)
    
    context = {
        'query': query,
        'index_type': index_type,
        'page_obj': page_obj,
    }
    return render(request, 'search_results.html', context)



def compute_drug_id(drug_obj: dict) -> str:
    """
    Compute a unique drug id based on drug_name and url.
    This must match the method used during indexing.
    """
    drug_name = drug_obj.get("drug_name", "Unknown")
    return str(hash(f"{drug_name}"))

def load_all_original_data(data_dir="searcher\Cleaned_Data_2") -> dict:
    """
    Loads original drug records from all JSON files in the specified directory.
    Returns a mapping from drug_id to the complete drug record.
    """
    print(os.path.exists(data_dir))
    mapping = {}
    files = glob.glob(os.path.join(data_dir, "*.json"))
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    drug_id = compute_drug_id(data)
                    mapping[drug_id] = data
                elif isinstance(data, list):
                    for record in data:
                        drug_id = compute_drug_id(record)
                        mapping[drug_id] = record
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return mapping

def detail_view(request, drug_name):
    """
    Displays a detailed page for a single drug record.
    index_type: 'dense' or 'sparse' (passed from the search results link)
    drug_id: Unique identifier for the drug.
    """
    query = request.GET.get('q', '')
    index_type = request.GET.get('index_type', 'sparse')
    page = request.GET.get('page', 1)

    # Load all original drug records from JSON files.
    original_data = load_all_original_data()
    drug_id = str(hash(drug_name))

    # Retrieve the drug record based on the drug_id.
    record = original_data.get(drug_id)
    if not record:
        raise Http404("Drug record not found.")
    

    context = {
        'drug': record,  # your found record
        'query': query,
        'index_type': index_type,
        'page': page
    }
    return render(request, 'detail.html', context)
