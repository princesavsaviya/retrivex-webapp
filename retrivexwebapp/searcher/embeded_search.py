#!/usr/bin/env python
import os
import json
import re
import time
import torch
import numpy as np
import faiss
from functools import lru_cache
from transformers import AutoTokenizer, AutoModel
import argparse
import glob

class DenseHybridSearch:
    def __init__(self, output_dir=r"searcher\embeddings_output", model_name='BAAI/bge-large-en-v1.5', original_data_dir=r"searcher\Cleaned_Data_2"):
        """
        Initialize the dense/hybrid search system.
        Loads the precomputed FAISS index and metadata from embeddings.json.
        Also loads the transformer model and tokenizer for query embedding.
        Additionally, loads the original drug records for aggregation.
        """
        self.output_dir = output_dir
        self.index_file = os.path.join(self.output_dir, "faiss_index_2.index")
        self.embeddings_file = os.path.join(self.output_dir, "embeddings_2.json")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        self.faiss_index = None
        self.drug_data = []
        self.query_cache = {}
        self.original_data = {}  # Mapping from drug_id to full original record
        
        self.load_index()
        self.load_data()
        self.load_original_data(original_data_dir)

    def load_index(self):
        """Load the precomputed FAISS index from file."""
        if os.path.exists(self.index_file):
            self.faiss_index = faiss.read_index(self.index_file)
        else:
            raise FileNotFoundError(f"Index file not found: {self.index_file}")

    def load_data(self):
        """Load the precomputed metadata (embeddings info) from JSON file."""
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'r', encoding='utf-8') as f:
                self.drug_data = json.load(f)
        else:
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_file}")

    def load_original_data(self, data_dir="Cleaned_Data_2"):
        """
        Load original drug records from JSON files in the specified directory.
        Builds a mapping from drug_id to the full drug record.
        """
        mapping = {}
        files = glob.glob(os.path.join(data_dir, "*.json"))
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        drug_id = self.compute_drug_id(data)
                        mapping[drug_id] = data
                    elif isinstance(data, list):
                        for record in data:
                            drug_id = self.compute_drug_id(record)
                            mapping[drug_id] = record
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        self.original_data = mapping

    def compute_drug_id(self, drug_obj: dict) -> str:
        """
        Compute a unique drug id based on drug_name and url.
        This must match the method used during indexing.
        """
        drug_name = drug_obj.get("drug_name", "Unknown")
        return str(hash(f"{drug_name}"))

    def preprocess_query(self, query: str) -> str:
        """Clean and expand common abbreviations in the query text."""
        query = re.sub(r'\s+', ' ', query).strip()
        common_expansions = {
            " ssri ": " selective serotonin reuptake inhibitor ",
            " snri ": " serotonin norepinephrine reuptake inhibitor ",
            " maoi ": " monoamine oxidase inhibitor ",
            " ppi ": " proton pump inhibitor ",
            " nsaid ": " nonsteroidal anti-inflammatory drug ",
            " ace ": " angiotensin converting enzyme ",
            " arb ": " angiotensin receptor blocker ",
            " dm ": " diabetes mellitus ",
            " htn ": " hypertension ",
            " tx ": " treatment ",
            " sx ": " symptoms ",
            " dx ": " diagnosis ",
            " rx ": " prescription ",
            " po ": " oral ",
            " iv ": " intravenous ",
            " im ": " intramuscular ",
            " sc ": " subcutaneous ",
        }
        for abbr, expansion in common_expansions.items():
            query = query.replace(abbr, expansion)
        return query

    @lru_cache(maxsize=100)
    def cached_search(self, query_str: str, top_k: int = 5) -> list:
        """Search the FAISS index using the query embedding (cached)."""
        query_embedding, _ = self.generate_embedding(query_str)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        D, I = self.faiss_index.search(query_embedding, top_k)
        results = [(float(D[0][i]), int(I[0][i])) for i in range(len(I[0]))]
        return results

    def generate_embedding(self, text: str) -> tuple:
        """
        Generate an embedding for the query text.
        (This is needed only for query encoding.)
        """
        start_time = time.time()
        if "bge" in self.tokenizer.name_or_path.lower():
            text = f"Retrieve information about: {text}"
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.tokenizer.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if "bge" in self.tokenizer.name_or_path.lower():
            embedding = outputs.last_hidden_state[:, 0]  # CLS token
        else:
            embedding = outputs.last_hidden_state.mean(dim=1)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        end_time = time.time()
        return embedding[0].detach().cpu().numpy(), (end_time - start_time)

    def boost_results(self, query: str, results: list) -> list:
        """
        Boost and reorder results based on metadata and keyword matching.
        Returns a list of (boosted_score, metadata) tuples.
        """
        query_lower = query.lower()
        query_terms = set(re.findall(r'\b\w+\b', query_lower))
        boosted_results = []
        for score, result in results:
            boost = 1.0
            text = result.get('text', '').lower()
            drug_name = result.get('drug_name', '').lower()
            is_important = result.get('is_important', False)
            if drug_name == query_lower or drug_name in query_lower:
                boost += 0.5
            if query_lower in text:
                boost += 0.4
            if is_important:
                boost += 0.3
            if query_terms:
                text_terms = set(re.findall(r'\b\w+\b', text))
                term_overlap = len(query_terms.intersection(text_terms)) / len(query_terms)
                boost += term_overlap * 0.2
            if len(text) < 200:
                boost -= 0.1
            if result.get('chunk_id', 0) == 0:
                boost += 0.1
            boosted_score = score * boost
            boosted_results.append((boosted_score, result))
        boosted_results.sort(reverse=True, key=lambda x: x[0])
        return boosted_results

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.9) -> list:
        """
        Perform a hybrid search that combines dense vector similarity with keyword matching,
        then aggregates chunk-level scores by drug id.
        
        1. Retrieve candidate chunks from FAISS.
        2. Boost each chunk's score.
        3. Compute a combined score for each chunk.
        4. Aggregate scores by drug id (summing scores).
        5. Select the top 50 drugs from aggregated results.
        6. For each top drug, fetch its complete original record from the original JSON data.
        7. Return a list of results containing only the aggregated score and drug name.
        """
        processed_query = self.preprocess_query(query)
        search_k = top_k * 3  # retrieve more candidates for re-ranking
        vector_results = self.cached_search(processed_query, search_k)
        
        # Convert index results to chunk-level results
        dense_results = []
        for score, idx in vector_results:
            if idx < len(self.drug_data):
                dense_results.append((score, self.drug_data[idx]))
        
        # Boost each chunk's score
        dense_results = self.boost_results(processed_query, dense_results)
        
        # Compute keyword matching component for each chunk
        processed_query_lower = processed_query.lower()
        query_terms = set(re.findall(r'\b\w+\b', processed_query_lower))
        hybrid_chunk_results = []
        for dense_score, result in dense_results:
            text = result.get('text', '').lower()
            text_terms = set(re.findall(r'\b\w+\b', text))
            keyword_matches = len(query_terms.intersection(text_terms))
            keyword_score = keyword_matches / len(query_terms) if query_terms else 0
            combined_score = alpha * dense_score + (1 - alpha) * keyword_score
            hybrid_chunk_results.append((combined_score, result))
        hybrid_chunk_results.sort(reverse=True, key=lambda x: x[0])
        
        # Aggregate scores by drug id
        aggregated = {}
        for score, meta in hybrid_chunk_results:
            drug_id = meta.get('drug_id')
            if drug_id is None:
                continue
            if drug_id not in aggregated:
                aggregated[drug_id] = {
                    "aggregated_score": score,
                    "drug_name": meta.get("drug_name", "Unknown")
                }
            else:
                aggregated[drug_id]["aggregated_score"] += score
        
        # Convert aggregated dict to list and sort, then take top 50 drugs
        aggregated_list = [(data["aggregated_score"], drug_id, data["drug_name"]) for drug_id, data in aggregated.items()]
        aggregated_list.sort(reverse=True, key=lambda x: x[0])
        top_aggregated = aggregated_list[:50]
        
        # For each top drug, fetch original record from self.original_data if available.
        # For simplicity, here we assume the original record has been loaded externally.
        # If not available, we only return aggregated score and drug name.
        final_results = []
        for agg_score, drug_id, drug_name in top_aggregated:
            drug_id_new = self.compute_drug_id({"drug_name": drug_name})
            # If self has an attribute 'original_data', fetch from it. Otherwise, skip.
            original_record = self.original_data.get(drug_id_new) if hasattr(self, 'original_data') else None
            final_results.append({
                "aggregated_score": agg_score,
                "drug_name": drug_name,
                "original_data": original_record,
                "drug_id": drug_id_new
            })
        return final_results

if __name__=="__main__":
    searcher = DenseHybridSearch()
    query = "headache"
    results = searcher.hybrid_search(query)
    count = 0
    for result in results:
        print(f"Score: {result['aggregated_score']}, Drug Name: {result['drug_name']}")
        if result["original_data"]:
            print(f"Original Data: {result['original_data']}")
        print("-" * 80)
        if count > 5:
            break
        count += 1