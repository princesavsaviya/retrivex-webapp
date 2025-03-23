import os
import json
import glob
import torch
import numpy as np
import faiss
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from datetime import datetime
import traceback  # For detailed error reporting
import logging
import re
from functools import lru_cache

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("drug_embedding_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DrugEmbeddingSystem:
    def __init__(self):
        """
        Initialize the drug embedding system with hardcoded parameters.
        """
        # HARDCODED PARAMETERS - Edit these values as needed
        self.json_directory = os.path.join("retrivexwebapp",os.path.join("searcher", "Cleaned_Data_2"))# New JSON data folder
        self.output_dir = os.path.join("retrivexwebapp",os.path.join("searcher", "embedding_output"))  # New output folder for embeddings and FAISS index

        self.max_tokens = 512  # Maximum number of tokens per chunk (increased from 300)
        # Using a stronger model for better retrieval performance
        self.model_name = 'BAAI/bge-large-en-v1.5'  # Stronger model for better embeddings
        
        self.embeddings_file = os.path.join(self.output_dir, "embeddings.json")
        self.index_file = os.path.join(self.output_dir, "faiss_index.index")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize transformer model and tokenizer
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()  # Set model to evaluation mode
        
        # If CUDA is available, use GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        
        # List to store drug data with embeddings
        self.drug_data = []
        
        # FAISS index will be initialized during processing
        self.faiss_index = None
        
        # Performance tracking variables
        self.processing_times = []
        self.doc_counts = []
        self.embedding_times = []
        
        # Get model max token length and set a safety buffer
        self.model_max_length = min(self.tokenizer.model_max_length, 512)  
        self.safety_buffer = 50  # Reduced from 100 since we know the structure better now
        self.effective_max_tokens = self.model_max_length - self.safety_buffer
        
        # Set chunk overlap to 20% of max tokens
        self.chunk_overlap_tokens = int(self.max_tokens * 0.2)
        
        logger.info(f"Model maximum token length: {self.model_max_length}")
        logger.info(f"Using effective maximum token length: {self.effective_max_tokens} (with {self.safety_buffer} token safety buffer)")
        logger.info(f"Chunk overlap tokens: {self.chunk_overlap_tokens}")
        
        # Add caching for frequent queries
        self.query_cache = {}
        
    def get_json_files(self) -> List[str]:
        """Get all JSON files in the specified directory."""
        files = glob.glob(os.path.join(self.json_directory, "*.json"))
        logger.info(f"Found {len(files)} JSON files")
        return files
    
    def load_json_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load JSON data from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {str(e)}")
            return []
    
    def create_document_from_drug_object(self, drug_obj: Dict[str, Any]) -> str:
        """
        Create a document from a drug object, excluding the URL.
        
        Args:
            drug_obj: Drug object from the JSON data
            
        Returns:
            Concatenated text from all fields except URL
        """
        document = ""
        
        # Add drug name with special prefix for better retrieval
        if 'drug_name' in drug_obj:
            document += f"DRUG NAME: {drug_obj['drug_name']}\n\n"
        
        # Process fields in a specific order to prioritize important information
        priority_fields = ['drug_class', 'summary', 'indications', 'dosage', 'side_effects', 
                           'contraindications', 'warnings', 'interactions', 'mechanism_of_action']
        
        # First add priority fields
        for field in priority_fields:
            if field in drug_obj and drug_obj[field]:
                field_name = ' '.join(word.capitalize() for word in field.split('_'))
                value = drug_obj[field]
                # Clean up the value - replace multiple spaces, newlines with single space
                if isinstance(value, str):
                    value = re.sub(r'\s+', ' ', value).strip()
                document += f"{field_name}: {value}\n\n"
        
        # Then add remaining fields except URL and drug_name
        for key, value in drug_obj.items():
            if (key not in priority_fields and 
                key != 'url' and 
                key != 'drug_name' and 
                value):  # Skip empty fields
                
                field_name = ' '.join(word.capitalize() for word in key.split('_'))
                
                # Clean up string values
                if isinstance(value, str):
                    value = re.sub(r'\s+', ' ', value).strip()
                    
                document += f"{field_name}: {value}\n\n"
        
        return document
    
    def get_token_count(self, text: str) -> int:
        """
        Get the exact token count for a text.
        """
        try:
            # For very short texts, just do the encoding
            if len(text) < 1000:
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
                return len(encoded)
            
            # For longer texts, estimate based on samples
            # First, count tokens in a sample of the text
            sample_size = min(1000, len(text) // 3)
            start_sample = text[:sample_size]
            middle_sample = text[len(text)//2 - sample_size//2:len(text)//2 + sample_size//2]
            end_sample = text[-sample_size:]
            
            # Count tokens in each sample
            start_tokens = len(self.tokenizer.encode(start_sample, add_special_tokens=False))
            middle_tokens = len(self.tokenizer.encode(middle_sample, add_special_tokens=False))
            end_tokens = len(self.tokenizer.encode(end_sample, add_special_tokens=False))
            
            # Calculate average tokens per character
            total_sample_chars = len(start_sample) + len(middle_sample) + len(end_sample)
            total_sample_tokens = start_tokens + middle_tokens + end_tokens
            tokens_per_char = total_sample_tokens / total_sample_chars
            
            # Estimate total tokens
            estimated_tokens = int(len(text) * tokens_per_char) + 2  # +2 for special tokens
            return estimated_tokens
            
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            # Return a large number to force truncation in case of errors
            return self.model_max_length + 100
    
    def truncate_text_to_token_limit(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to ensure it doesn't exceed the token limit.
        Uses a binary search approach for precise truncation.
        """
        try:
            # Quick check if the text is already under the limit
            token_count = self.get_token_count(text)
            if token_count <= max_tokens:
                return text
            
            # Binary search to find the right truncation point
            low, high = 0, len(text)
            best_truncated = ""
            best_token_count = 0
            
            while low <= high:
                mid = (low + high) // 2
                truncated = text[:mid]
                curr_token_count = self.get_token_count(truncated)
                
                if curr_token_count <= max_tokens:
                    # This might be a valid truncation point
                    if curr_token_count > best_token_count:
                        best_truncated = truncated
                        best_token_count = curr_token_count
                    low = mid + 1
                else:
                    high = mid - 1
            
            # If we found a good truncation point, return it
            if best_truncated:
                return best_truncated + "... [truncated]"
            
            # Fallback: use the tokenizer's built-in truncation
            encoded = self.tokenizer.encode_plus(
                text,
                max_length=max_tokens,
                truncation=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
            return self.tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Error truncating text: {str(e)}")
            # Emergency truncation - just cut the string
            approx_chars = max_tokens * 4  # Very rough estimate
            return text[:approx_chars] + "... [truncated]"
    
    def extract_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """
        Extract the last N tokens from text to use as overlap for the next chunk.
        """
        try:
            # Get approximate character count for overlap
            # This is more efficient than tokenizing the entire text
            avg_chars_per_token = 4  # Rough estimate
            approx_chars = overlap_tokens * avg_chars_per_token
            
            # Get the last paragraph boundaries if possible
            paragraphs = text.split('\n\n')
            
            if len(paragraphs) > 1:
                # Try to get complete paragraphs for overlap
                overlap_text = ""
                for para in reversed(paragraphs):
                    overlap_text = para + "\n\n" + overlap_text
                    if len(overlap_text) > approx_chars:
                        break
                
                # Check token count of overlap
                overlap_token_count = self.get_token_count(overlap_text)
                if overlap_token_count <= overlap_tokens * 1.5:  # Allow some buffer
                    return overlap_text.strip()
            
            # Fallback: just get the last N characters
            last_chunk = text[-min(len(text), approx_chars*2):]  # Get more than needed
            
            # Try to truncate at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', last_chunk)
            overlap_text = ""
            
            for sentence in reversed(sentences):
                potential_overlap = sentence + " " + overlap_text
                token_count = self.get_token_count(potential_overlap)
                
                if token_count <= overlap_tokens * 1.5:
                    overlap_text = potential_overlap
                else:
                    break
            
            return overlap_text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting overlap: {str(e)}")
            # Fallback - return last few characters
            return text[-min(len(text), 200):].strip()
    
    def chunk_text(self, text: str, drug_name: str) -> List[Tuple[str, bool]]:
        """
        Split text into chunks with meaningful overlap, preserving semantic boundaries.
        Returns a list of (chunk_text, is_important) tuples.
        
        Args:
            text: Text to split into chunks
            drug_name: Name of the drug for this text
            
        Returns:
            List of (chunk_text, is_important) tuples. is_important flag marks chunks
            with key information like drug name, class, indications, etc.
        """
        # First check if the entire text fits within token limit
        total_tokens = self.get_token_count(text)
        if total_tokens <= self.effective_max_tokens:
            # Flag as important if contains drug name and key sections
            is_important = "DRUG NAME:" in text[:500] or "Drug Class:" in text or "Indications:" in text
            return [(text, is_important)]
        
        chunks = []
        
        # Split by paragraphs to maintain semantic boundaries
        paragraphs = text.split('\n\n')
        current_chunk = ""
        current_chunk_tokens = 0
        overlap_text = ""
        
        # Track if the current chunk contains important information
        # (drug name, class, indications, etc.)
        important_patterns = [
            "DRUG NAME:", 
            "Drug Class:", 
            "Indications:", 
            "Mechanism Of Action:",
            "Summary:"
        ]
        
        for i, paragraph in enumerate(paragraphs):
            # Check if this is an important paragraph
            is_important_para = any(pattern in paragraph for pattern in important_patterns)
            
            # Always try to include drug name in each chunk for context
            if i == 0 and "DRUG NAME:" in paragraph:
                drug_name_para = paragraph
            else:
                drug_name_para = ""
            
            # Check if adding this paragraph would exceed token limit
            para_tokens = self.get_token_count(paragraph)
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            potential_tokens = self.get_token_count(potential_chunk)
            
            if potential_tokens <= self.effective_max_tokens:
                # If within limit, add to current chunk
                current_chunk = potential_chunk
                current_chunk_tokens = potential_tokens
            else:
                # Current chunk is full - save it
                if current_chunk:
                    # Flag the chunk as important if it contains important sections
                    is_important = any(pattern in current_chunk for pattern in important_patterns)
                    chunks.append((current_chunk, is_important))
                    
                    # Extract overlap for the next chunk
                    overlap_text = self.extract_overlap_text(current_chunk, self.chunk_overlap_tokens)
                
                # Reset for new chunk
                # Always include drug name in each chunk if available
                if drug_name_para:
                    new_chunk_base = drug_name_para + "\n\n"
                else:
                    new_chunk_base = ""
                
                # Add overlap text if it doesn't make the chunk too long
                if overlap_text:
                    if self.get_token_count(new_chunk_base + overlap_text + "\n\n" + paragraph) <= self.effective_max_tokens:
                        current_chunk = new_chunk_base + overlap_text + "\n\n" + paragraph
                    else:
                        current_chunk = new_chunk_base + paragraph
                else:
                    current_chunk = new_chunk_base + paragraph
                
                current_chunk_tokens = self.get_token_count(current_chunk)
                
                # If even a single paragraph is too large, we need to split it
                if current_chunk_tokens > self.effective_max_tokens:
                    logger.warning(f"Large paragraph detected for drug {drug_name}. Splitting.")
                    # Split the paragraph into sentences
                    if drug_name_para:
                        base_text = drug_name_para + "\n\n"
                    else:
                        base_text = ""
                    
                    self._process_large_paragraph(paragraph, chunks, base_text, drug_name)
                    current_chunk = ""
                    current_chunk_tokens = 0
        
        # Add the final chunk if it has content
        if current_chunk:
            is_important = any(pattern in current_chunk for pattern in important_patterns)
            chunks.append((current_chunk, is_important))
        
        # Ensure all chunks are within token limits
        final_chunks = []
        for i, (chunk, is_important) in enumerate(chunks):
            chunk_tokens = self.get_token_count(chunk)
            if chunk_tokens > self.effective_max_tokens:
                logger.warning(f"Chunk {i} exceeds token limit ({chunk_tokens} > {self.effective_max_tokens}). Truncating.")
                truncated = self.truncate_text_to_token_limit(chunk, self.effective_max_tokens)
                final_chunks.append((truncated, is_important))
            else:
                final_chunks.append((chunk, is_important))
        
        logger.info(f"Split text into {len(final_chunks)} chunks for drug: {drug_name}")
        return final_chunks
    
    def _process_large_paragraph(self, paragraph: str, chunks: List[Tuple[str, bool]], prefix: str, drug_name: str):
        """
        Helper method to process a paragraph that's too large for a single chunk.
        
        Args:
            paragraph: Large paragraph text to process
            chunks: List to append chunks to
            prefix: Text to prepend to each chunk (e.g., drug name)
            drug_name: Name of the drug for logging
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        current_chunk = prefix
        
        for sentence in sentences:
            potential_chunk = current_chunk + sentence + " "
            token_count = self.get_token_count(potential_chunk)
            
            if token_count <= self.effective_max_tokens:
                current_chunk = potential_chunk
            else:
                # Save current chunk and start a new one
                if current_chunk != prefix:  # Avoid empty chunks
                    # Check if this is an important chunk
                    is_important = "DRUG NAME:" in current_chunk[:500]
                    chunks.append((current_chunk, is_important))
                
                # Start new chunk with prefix and this sentence
                current_chunk = prefix + sentence + " "
                
                # If even a single sentence with prefix is too large, truncate it
                if self.get_token_count(current_chunk) > self.effective_max_tokens:
                    logger.warning(f"Very long sentence detected for drug {drug_name}. Truncating.")
                    truncated = self.truncate_text_to_token_limit(current_chunk, self.effective_max_tokens)
                    is_important = "DRUG NAME:" in truncated[:500]
                    chunks.append((truncated, is_important))
                    current_chunk = prefix  # Reset
        
        # Add the final chunk if it's not just the prefix
        if current_chunk != prefix:
            is_important = "DRUG NAME:" in current_chunk[:500]
            chunks.append((current_chunk, is_important))
    
    @torch.no_grad()
    def generate_embedding(self, text: str) -> Tuple[np.ndarray, float]:
        """
        Generate embedding for a text using the transformer model.
        Optimized for BGE models which need special prompt formatting.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Tuple of (embedding vector, time_taken_in_seconds)
        """
        start_time = time.time()
        
        try:
            # For BGE models, add a query prefix for better retrieval performance
            if "bge" in self.model_name.lower():
                text = f"Retrieve information about: {text}"
            
            # Check token count and truncate if needed
            token_count = self.get_token_count(text)
            if token_count > self.effective_max_tokens:
                logger.warning(f"Text exceeds token limit in generate_embedding ({token_count} > {self.effective_max_tokens}). Truncating.")
                text = self.truncate_text_to_token_limit(text, self.effective_max_tokens)
            
            # Tokenize with padding and attention mask
            encoded = self.tokenizer.encode_plus(
                text, 
                add_special_tokens=True,
                max_length=self.model_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                return_attention_mask=True
            )
            
            # Move to device
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Generate embeddings
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # For BGE models, use the [CLS] token embedding
            if "bge" in self.model_name.lower():
                embeddings = outputs.last_hidden_state[:, 0]  # [CLS] token embedding
            else:
                # For other models, use mean pooling
                embeddings = outputs.last_hidden_state
                # Apply attention mask for mean pooling
                mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                masked_embeddings = embeddings * mask
                summed = torch.sum(masked_embeddings, 1)
                summed_mask = torch.clamp(mask.sum(1), min=1e-9)
                embeddings = summed / summed_mask
            
            # Normalize the embeddings (important for cosine similarity)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            end_time = time.time()
            time_taken = end_time - start_time
            self.embedding_times.append(time_taken)
            
            return embeddings[0].cpu().numpy(), time_taken
            
        except Exception as e:
            logger.error(f"Error in generate_embedding: {str(e)}")
            logger.error(traceback.format_exc())
            # Return a zero vector as fallback
            dim = 1024 if "large" in self.model_name.lower() else 768  # Default dimension for many transformer models
            end_time = time.time()
            time_taken = end_time - start_time
            self.embedding_times.append(time_taken)
            return np.zeros(dim), time_taken
    
    def process_files(self):
        """Process all JSON files and generate embeddings."""
        json_files = self.get_json_files()
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        all_embeddings = []
        drug_texts = []
        drug_metadata = []
        
        start_time = time.time()
        processed_docs = 0
        failed_docs = 0
        
        for file_path in tqdm(json_files, desc="Processing files"):
            try:
                drug_objects = self.load_json_data(file_path)
                
                # Process each drug object
                for drug_obj in drug_objects:
                    try:
                        # Extract metadata
                        url = drug_obj.get('url', '')
                        drug_name = drug_obj.get('drug_name', 'Unknown')
                        
                        # Generate a unique ID for this drug
                        drug_id = hash(f"{drug_name}_{url}")
                        
                        # Create document
                        document = self.create_document_from_drug_object(drug_obj)
                        
                        # Split into chunks with semantic boundaries
                        chunks = self.chunk_text(document, drug_name)
                        logger.info(f"Drug '{drug_name}' split into {len(chunks)} chunks")
                        
                        # Process each chunk
                        for i, (chunk, is_important) in enumerate(chunks):
                            try:
                                # Generate embedding
                                embedding, time_taken = self.generate_embedding(chunk)
                                
                                # Only add to results if embedding is not all zeros
                                if np.any(embedding):
                                    all_embeddings.append(embedding)
                                    drug_texts.append(chunk)
                                    
                                    metadata = {
                                        'drug_id': drug_id,
                                        'drug_name': drug_name,
                                        'url': url,
                                        'chunk_id': i,
                                        'total_chunks': len(chunks),
                                        'is_important': is_important
                                    }
                                    drug_metadata.append(metadata)
                                else:
                                    logger.warning(f"Zero embedding generated for chunk {i} of drug {drug_name}")
                                    
                            except Exception as e:
                                logger.error(f"Error processing chunk {i} for drug {drug_name}: {str(e)}")
                                failed_docs += 1
                        
                        # Update progress tracking
                        processed_docs += 1
                        current_time = time.time() - start_time
                        self.doc_counts.append(processed_docs)
                        self.processing_times.append(current_time)
                        
                    except Exception as e:
                        logger.error(f"Error processing drug {drug_obj.get('drug_name', 'Unknown')}: {str(e)}")
                        failed_docs += 1
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
        
        logger.info(f"Processed {processed_docs} drugs successfully, {failed_docs} drugs failed")
        
        # Check if we have embeddings to create the index
        if not all_embeddings:
            logger.error("No valid embeddings were generated. Cannot create FAISS index.")
            return
            
        # Convert to numpy array and create index
        embeddings_array = np.array(all_embeddings).astype('float32')
        self.create_faiss_index(embeddings_array)
        
        # Save data and generate charts
        self.save_data(drug_metadata, drug_texts)
        self.generate_performance_charts()
        
        logger.info(f"Successfully processed {len(drug_metadata)} chunks from {len(set(meta['drug_id'] for meta in drug_metadata))} drugs")
    
    def create_faiss_index(self, embeddings_array: np.ndarray):
        """
        Create FAISS index with IVF for fast similarity search.
        
        Args:
            embeddings_array: Numpy array of embeddings
        """
        # Get dimension of embeddings
        dimension = embeddings_array.shape[1]
        
        # Calculate number of clusters (cells)
        nlist = min(4096, max(int(np.sqrt(embeddings_array.shape[0])), 50))
        logger.info(f"Creating IVF index with {nlist} clusters")
        
        # Create the quantizer (base index for centroids)
        quantizer = faiss.IndexFlatIP(dimension)
        
        # Create the IVF index
        self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Important: IVF indices must be trained before adding vectors
        logger.info("Training the index...")
        self.faiss_index.train(embeddings_array)
        
        # Add vectors to the index
        logger.info("Adding vectors to index...")
        self.faiss_index.add(embeddings_array)
        
        # Set the number of clusters to probe during search (more = more accurate but slower)
        self.faiss_index.nprobe = min(64, nlist)
        
        # Save index to file
        faiss.write_index(self.faiss_index, self.index_file)
        logger.info(f"FAISS IVF index created and saved to {self.index_file}")
    
    def save_data(self, metadata: List[Dict], texts: List[str]):
        """
        Save metadata and texts to file.
        
        Args:
            metadata: List of metadata dictionaries
            texts: List of text chunks
        """
        # Add text to metadata
        for i, meta in enumerate(metadata):
            meta['text'] = texts[i]
        
        # Save to file
        with open(self.embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Embedding data saved to {self.embeddings_file}")
    
    def generate_performance_charts(self):
        """Generate and save charts showing processing performance."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory for charts if it doesn't exist
        charts_dir = os.path.join(self.output_dir, "performance_charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # 1. Document processing time chart
        plt.figure(figsize=(10, 6))
        plt.plot(self.processing_times, self.doc_counts, 'b-', linewidth=2)
        plt.title('Documents Processed Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Number of Documents Processed')
        plt.grid(True)
        doc_chart_path = os.path.join(charts_dir, f"doc_processing_time_{timestamp}.png")
        plt.savefig(doc_chart_path)
        plt.close()
        logger.info(f"Document processing chart saved to {doc_chart_path}")
        
        # 2. Embedding generation time chart
        plt.figure(figsize=(10, 6))
        plt.hist(self.embedding_times, bins=30, color='blue', alpha=0.7)
        plt.title('Embedding Generation Time Distribution')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency')
        plt.grid(True)
        embedding_chart_path = os.path.join(charts_dir, f"embedding_time_{timestamp}.png")
        plt.savefig(embedding_chart_path)
        plt.close()
        logger.info(f"Embedding time chart saved to {embedding_chart_path}")
        
        # 3. Embedding time statistics
        avg_time = np.mean(self.embedding_times)
        median_time = np.median(self.embedding_times)
        min_time = np.min(self.embedding_times)
        max_time = np.max(self.embedding_times)
        
        stats = {
            "timestamp": timestamp,
            "total_documents": self.doc_counts[-1] if self.doc_counts else 0,
            "total_processing_time": self.processing_times[-1] if self.processing_times else 0,
            "avg_embedding_time": avg_time,
            "median_embedding_time": median_time,
            "min_embedding_time": min_time,
            "max_embedding_time": max_time,
            "total_embeddings": len(self.embedding_times)
        }
        
        # Save statistics to file
        stats_file = os.path.join(charts_dir, f"performance_stats_{timestamp}.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)
        
        logger.info(f"Performance statistics saved to {stats_file}")
        logger.info(f"Average embedding time: {avg_time:.4f} seconds")
        logger.info(f"Median embedding time: {median_time:.4f} seconds")
        logger.info(f"Min/Max embedding time: {min_time:.4f}/{max_time:.4f} seconds")
    
    def load_index(self):
        """Load the FAISS index from file."""
        if os.path.exists(self.index_file):
            self.faiss_index = faiss.read_index(self.index_file)
            logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            return True
        else:
            logger.warning(f"Index file {self.index_file} not found")
            return False
    
    def load_data(self):
        """Load the embedding data from file."""
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'r', encoding='utf-8') as f:
                self.drug_data = json.load(f)
            logger.info(f"Loaded data for {len(self.drug_data)} chunks")
            return True
        else:
            logger.warning(f"Embeddings file {self.embeddings_file} not found")
            return False
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess query text for better retrieval.
        
        Args:
            query: Raw query string
            
        Returns:
            Processed query string
        """
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Add drug-specific expansions for common abbreviations
        common_expansions = {
            # Drug classes
            " ssri ": " selective serotonin reuptake inhibitor ",
            " snri ": " serotonin norepinephrine reuptake inhibitor ",
            " maoi ": " monoamine oxidase inhibitor ",
            " ppi ": " proton pump inhibitor ",
            " nsaid ": " nonsteroidal anti-inflammatory drug ",
            " ace ": " angiotensin converting enzyme ",
            " arb ": " angiotensin receptor blocker ",
            " dm ": " diabetes mellitus ",
            " htn ": " hypertension ",
            
            # Common terms
            " tx ": " treatment ",
            " sx ": " symptoms ",
            " dx ": " diagnosis ",
            " rx ": " prescription ",
            
            # Routes of administration
            " po ": " oral ",
            " iv ": " intravenous ",
            " im ": " intramuscular ",
            " sc ": " subcutaneous ",
        }
        
        # Apply expansions
        for abbr, expansion in common_expansions.items():
            query = query.replace(abbr, expansion)
        
        return query
    
    @lru_cache(maxsize=100)
    def cached_search(self, query_str: str, top_k: int = 5) -> List[Tuple[float, int]]:
        """
        Cached version of the vector search function.
        
        Args:
            query_str: Query string
            top_k: Number of results to return
            
        Returns:
            List of (score, index) tuples
        """
        # Generate embedding for query with safeguards for token length
        query_embedding, _ = self.generate_embedding(query_str)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        D, I = self.faiss_index.search(query_embedding, top_k)
        
        # Convert to list of tuples
        results = [(float(D[0][i]), int(I[0][i])) for i in range(len(I[0]))]
        return results
    
    def search(self, query: str, top_k: int = 5, rerank: bool = True) -> List[Dict]:
        """
        Search for similar drug information.
        
        Args:
            query: Query text
            top_k: Number of results to return
            rerank: Whether to apply metadata boosting
            
        Returns:
            List of (score, metadata) tuples
        """
        # Check if index and data are loaded
        if self.faiss_index is None and not self.load_index():
            raise ValueError("FAISS index not loaded. Run process_files() first or ensure the index file exists.")
        
        if not self.drug_data and not self.load_data():
            raise ValueError("Embedding data not loaded. Run process_files() first or ensure the embeddings file exists.")
        
        # Preprocess query
        processed_query = self.preprocess_query(query)
        
        # Check if result is in cache
        cache_key = f"{processed_query}_{top_k}"
        if cache_key in self.query_cache:
            logger.info(f"Using cached results for query: {processed_query}")
            return self.query_cache[cache_key]
        
        # Get more results than requested for reranking
        search_k = top_k * 3 if rerank else top_k
        
        # Search
        vector_results = self.cached_search(processed_query, search_k)
        
        # Collect results
        results = []
        for i, (score, idx) in enumerate(vector_results):
            if idx < len(self.drug_data):
                results.append((float(score), self.drug_data[idx]))
        
        # Apply reranking if requested
        if rerank:
            results = self.boost_results(processed_query, results)
            results = results[:top_k]  # Limit to requested number after boosting
        
        # Cache results
        self.query_cache[cache_key] = results
        
        return results
    
    def boost_results(self, query: str, results: List[Tuple[float, Dict]]) -> List[Tuple[float, Dict]]:
        """
        Boost search results based on metadata and content.
        
        Args:
            query: Search query
            results: List of (score, metadata) tuples
            
        Returns:
            Boosted and reordered results
        """
        query_lower = query.lower()
        query_terms = set(re.findall(r'\b\w+\b', query_lower))
        boosted_results = []
        
        for score, result in results:
            boost = 1.0
            
            # Get text and metadata
            text = result.get('text', '').lower()
            drug_name = result.get('drug_name', '').lower()
            is_important = result.get('is_important', False)
            
            # Major boosts:
            
            # 1. Boost if drug name matches query exactly
            if drug_name == query_lower or drug_name in query_lower:
                boost += 0.5
            
            # 2. Boost if contains exact query phrase
            if query_lower in text:
                boost += 0.4
            
            # 3. Boost important chunks (those with drug name, class, or indications)
            if is_important:
                boost += 0.3
            
            # Minor boosts:
            
            # 4. Boost based on percentage of query terms found
            if query_terms:
                text_terms = set(re.findall(r'\b\w+\b', text))
                term_overlap = len(query_terms.intersection(text_terms)) / len(query_terms)
                boost += term_overlap * 0.2
            
            # 5. Penalize chunks that are too short (likely less informative)
            if len(text) < 200:
                boost -= 0.1
            
            # 6. Boost first chunks of a drug (typically more important info)
            if result.get('chunk_id', 0) == 0:
                boost += 0.1
            
            # Apply the boost
            boosted_score = score * boost
            boosted_results.append((boosted_score, result))
        
        # Sort by boosted score
        boosted_results.sort(reverse=True, key=lambda x: x[0])
        return boosted_results
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.7):
        """
        Hybrid search combining vector similarity and keyword matching.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            alpha: Weight for vector search (1-alpha is weight for keyword search)
            
        Returns:
            List of (score, metadata) tuples
        """
        # First, do a regular vector search to get candidates
        vector_results = self.search(query, top_k=top_k*2, rerank=False)
        
        # Prepare for keyword matching
        processed_query = self.preprocess_query(query)
        query_terms = set(re.findall(r'\b\w+\b', processed_query.lower()))
        
        if not query_terms:
            return vector_results[:top_k]
        
        # Calculate keyword match scores
        hybrid_results = []
        
        for vector_score, result in vector_results:
            # Get text
            text = result.get('text', '').lower()
            
            # Count keyword matches
            text_terms = set(re.findall(r'\b\w+\b', text))
            keyword_matches = len(query_terms.intersection(text_terms))
            
            # Normalize keyword score (0 to 1)
            keyword_score = keyword_matches / len(query_terms) if query_terms else 0
            
            # Combine scores
            combined_score = alpha * vector_score + (1 - alpha) * keyword_score
            
            hybrid_results.append((combined_score, result))
        
        # Sort by combined score
        hybrid_results.sort(reverse=True, key=lambda x: x[0])
        
        return hybrid_results[:top_k]
    
    def search_with_examples(self, query: str, top_k: int = 5, rerank: bool = True):
        """
        Search with detailed example output for debugging and explanation.
        
        Args:
            query: Query text
            top_k: Number of results to return
            rerank: Whether to apply metadata boosting
            
        Returns:
            Dict with search results and explanation
        """
        # Preprocess query
        processed_query = self.preprocess_query(query)
        
        # Generate embedding
        query_embedding, embedding_time = self.generate_embedding(processed_query)
        
        # Get raw vector search results
        raw_results = self.cached_search(processed_query, top_k * 2)
        
        # Get final results with boosting if requested
        if rerank:
            results = self.boost_results(processed_query, 
                                         [(score, self.drug_data[idx]) for score, idx in raw_results 
                                          if idx < len(self.drug_data)])
        else:
            results = [(score, self.drug_data[idx]) for score, idx in raw_results 
                       if idx < len(self.drug_data)]
        
        # Trim to requested number
        results = results[:top_k]
        
        # Prepare example output
        example_output = {
            "query": query,
            "processed_query": processed_query,
            "embedding_time_ms": round(embedding_time * 1000, 2),
            "results": [{
                "score": round(score, 4),
                "drug_name": result["drug_name"],
                "chunk_id": result["chunk_id"],
                "is_important": result.get("is_important", False),
                "text_excerpt": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
            } for score, result in results]
        }
        
        return example_output


def main():
    """Main function to run the embedding system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process drug information JSONs and create embeddings")
    parser.add_argument("--search", help="Search query (if specified, will run search instead of processing)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of search results to return")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid search (vector + keyword)")
    parser.add_argument("--no-rerank", action="store_true", help="Disable metadata boosting/reranking")
    parser.add_argument("--explain", action="store_true", help="Show detailed search explanation")
    
    args = parser.parse_args()
    
    system = DrugEmbeddingSystem()
    
    if args.search:
        # Run search
        try:
            if args.explain:
                # Use detailed search with examples
                results = system.search_with_examples(args.search, args.top_k, not args.no_rerank)
                
                print(f"\nDetailed search results for query: '{results['query']}'")
                print(f"Processed query: '{results['processed_query']}'")
                print(f"Embedding time: {results['embedding_time_ms']} ms")
                print("-" * 80)
                
                for i, result in enumerate(results['results']):
                    print(f"Result {i+1} (Score: {result['score']}):")
                    print(f"Drug: {result['drug_name']} (Chunk {result['chunk_id']+1}, Important: {result['is_important']})")
                    print(f"Text excerpt: {result['text_excerpt']}\n")
                
            elif args.hybrid:
                # Use hybrid search
                results = system.hybrid_search(args.search, args.top_k)
                
                print(f"\nTop {len(results)} hybrid search results for query: '{args.search}'")
                print("-" * 80)
                
                for i, (score, result) in enumerate(results):
                    print(f"Result {i+1} (Score: {score:.2f}):")
                    print(f"Drug: {result['drug_name']}")
                    print(f"URL: {result.get('url', 'N/A')}")
                    print(f"Chunk: {result['chunk_id']+1} of {result['total_chunks']}")
                    print(f"Text excerpt: {result['text'][:200]}...\n")
                
            else:
                # Use standard search
                results = system.search(args.search, args.top_k, not args.no_rerank)
                
                print(f"\nTop {len(results)} results for query: '{args.search}'")
                print("-" * 80)
                
                for i, (score, result) in enumerate(results):
                    print(f"Result {i+1} (Score: {score:.2f}):")
                    print(f"Drug: {result['drug_name']}")
                    print(f"URL: {result.get('url', 'N/A')}")
                    print(f"Chunk: {result['chunk_id']+1} of {result['total_chunks']}")
                    print(f"Text excerpt: {result['text'][:200]}...\n")
                    
        except ValueError as e:
            logger.error(f"Error: {str(e)}")
    else:
        # Process files
        system.process_files()
        
        # Demo search capability
        if system.drug_data:
            sample_drug = system.drug_data[0]['drug_name']
            print(f"\nTry searching with: python {os.path.basename(__file__)} --search \"{sample_drug}\"")
            print(f"For hybrid search: python {os.path.basename(__file__)} --search \"{sample_drug}\" --hybrid")
            print(f"For detailed explanation: python {os.path.basename(__file__)} --search \"{sample_drug}\" --explain")


if __name__ == "__main__":
    main()