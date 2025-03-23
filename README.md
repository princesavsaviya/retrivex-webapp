# RETRIVEX WEBAPP

This project implements a retrieval system for drug information. It includes three main components:

- **Embed Searcher** (`embed_searcher_1.py`): Processes JSON files to generate embeddings and creates a FAISS index.
- **Indexer** (`indexer.py`): Uses Lucene to build a search index from JSON data.
- **Indexer_2** (`indexer_2.py`): Uses Elasticsearch to build a search index from JSON data.
- **Retriever** (`retriver.py`): Provides search functionality using the Lucene index.
- **Retriever_2** (`retriver_2.py`): Provides search functionality using the elasticsearch index.
- **Embeded_search** (`embeded_search.py`): Provides search functionality using the BERT index.

- **Note** We have include 150 json file for testing in Cleaned_Data_2 and before running webapp, you do need to do indexing for Elasticsearch by running indexer_2.py and BERT indexing is already done but its only done on 5 json files.


- **Note** You can use anyone of the indexer and retriever from Lucene and Elasticsearch, but make sure to changes the path and module if you make any changes. by default we uses Elasticsearch for parse index and BERT for dense index.


## Project Structure


RETRIVEX WEBAPP
├── retrivexwebapp
│   ├── retriverwebapp
│   ├── searcher
│   │   ├── cleaned_data_2        # Contains JSON files with drug information data
│   │   ├── medicine_index
│   │   ├── embedding_output      # Output directory for indexes, embeddings, etc.
│   │   ├── embeded_search.py     # Modified version of embed_searcher_1.py for webapp
│   │   └── retriver_2.py           # Lucene-based search/retrieval script for webapp
│   ├── de.sqllite3
│   └── manage.py
├── embed_searcher_1.py           # Embedding and FAISS indexing script
└── indexer_2.py                    # Lucene-based indexing script

## Requirements

- Python 3.12
- for more information see requirements.txt

## Setup & Installation

1. **Install dependencies:**

   You can install the required packages via pip. For example:

   pip install torch transformers faiss-cpu numpy matplotlib tqdm pylucene elasticsearch


2. **Elasticsearch Setup:**

   Download Elasticsearch:
      Visit the official Elasticsearch download page:
      https://www.elastic.co/downloads/elasticsearch

   Ensure you have Elasticsearch installed and running. If you do not have Elasticsearch running, you can start it by executing the following file ::

   `elasticsearch-8.17.3-windows-x86_64\elasticsearch-8.17.3\bin\elasticsearch.bat`   

   Also, update the configuration in indexer.py and retriver.py as needed to connect to your Elasticsearch instance.



4. **Java and Lucene Setup:**

   Ensure you have Java installed. The Lucene scripts use the PyLucene package. Follow the PyLucene installation instructions as needed.


## Usage

### Embedding and FAISS Indexing

To process the JSON files, generate embeddings, and create the FAISS index:

python embed_searcher_1.py


After processing, embeddings and the FAISS index will be stored in the `retrivexwebapp/searcher/embedding_output` folder.

### Lucene Indexing

To create a Lucene index from the JSON files in `retrivexwebapp/searcher/Cleaned_Data_2`:

python indexer.py

This will build the index in `retrivexwebapp/searcher/medicine_index`.

### ElasticSearch indexing

To create an Elasticsearch index from the JSON files in retrivexwebapp/searcher/cleaned_data_2

python indexer_2.py

### Retriever Web App (Django)

The retrieval functionality is now implemented as a Django web application.

Note :: Before running webapp, make sure you have done both type of embedding


Run the Django server:

python manage.py runserver

Access the Web App:

Open your web browser and navigate to http://127.0.0.1:8000/ to use the retrieval interface.

The app allows you to enter search queries that utilize the generated embeddings and/or Lucene index to retrieve drug information.

## Additional Information

- **Data Input:** JSON files are expected in the `searcher/cleaned_data_2` directory.
- **Output:** All indexing and embedding files are saved to the `searcher/data` directory.
- **Logging:** Detailed logging is enabled for debugging and performance tracking.


## Contact

For any questions or support, please contact Prince Savsaviya at [princesavsaviya2023.learning@gmail.com].
