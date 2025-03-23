import os
import json
import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, TextField, StringField, FieldType
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory

# Initialize Java VM
lucene.initVM(vmargs=['-Djava.awt.headless=true'])

class MedicineIndexer:
    def __init__(self, index_dir="medicine_index"):
        self.index_dir = index_dir
        self.analyzer = StandardAnalyzer()
        
        # Define field types with different configurations
        # For fields that shouldn't be tokenized but should be stored and indexed
        self.nameType = FieldType()
        self.nameType.setStored(True)
        self.nameType.setTokenized(False)
        self.nameType.setIndexOptions(IndexOptions.DOCS)
        
        # For fields that should be tokenized, stored and indexed
        self.contentType = FieldType()
        self.contentType.setStored(True)
        self.contentType.setTokenized(True)
        self.contentType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
        
    def create_index(self, data_directory):
        """Create index from all JSON files in the directory"""
        if os.path.exists(self.index_dir):
            # Delete the existing index directory
            import shutil
            shutil.rmtree(self.index_dir)
        
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir, exist_ok=True)
        
        # Create disk directory for storing index
        store = SimpleFSDirectory(Paths.get(self.index_dir))
        config = IndexWriterConfig(self.analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        writer = IndexWriter(store, config)
        
        try:
            processed_files = 0
            for filename in os.listdir(data_directory):
                if filename.endswith('.json'):
                    file_path = os.path.join(data_directory, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            medicines = json.load(f)
                            if isinstance(medicines, list):
                                for medicine in medicines:
                                    self.index_medicine(writer, medicine)
                            else:
                                self.index_medicine(writer, medicines)
                            processed_files += 1
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
            
            print(f"Successfully indexed {processed_files} files")
            writer.commit()
        finally:
            writer.close()
    
    def index_medicine(self, writer, medicine_data):
        """Index a single medicine document with proper field types"""
        doc = Document()
        
        # Add drug_name field - not tokenized but indexed and stored
        if 'drug_name' in medicine_data:
            doc.add(Field("drug_name", medicine_data['drug_name'] or "", self.nameType))
        
        # Add content fields - tokenized, indexed and stored
        content_fields = [
            'drug_details', 'product_introduction', 'uses_and_benefits',
            'side_effects', 'how_to_use', 'how_drug_works', 'safety_advice',
            'missed_dose', 'expert_advice', 'fact_box', 'drug_interaction', 'faq'
        ]
        
        for field in content_fields:
            if field in medicine_data and medicine_data[field]:
                doc.add(Field(field, str(medicine_data[field]), self.contentType))
        
        writer.addDocument(doc)

if __name__ == "__main__":
    indexer = MedicineIndexer(index_dir=os.path.join("retrivexwebapp",os.path.join("searcher","medicine_index")))
    indexer.create_index(os.path.join("retrivexwebapp",os.path.join("searcher", "Cleaned_Data_2")))