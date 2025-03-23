import lucene
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser, MultiFieldQueryParser
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause, PhraseQuery
from org.apache.lucene.search import TermRangeQuery, WildcardQuery, BoostQuery
from org.apache.lucene.index import Term
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import FuzzyQuery, PhraseQuery
from org.apache.lucene.util import BytesRef
from java.nio.file import Paths
import re
from org.apache.lucene.search import BooleanQuery, BooleanClause
from org.apache.lucene.queryparser.classic import QueryParser
import os
# Initialize Java VM
lucene.initVM(vmargs=['-Djava.awt.headless=true'])

class MedicineSearcher:
    def __init__(self, index_dir= os.path.join("searcher", "medicine_index")):
        self.index_dir = index_dir
        self.analyzer = StandardAnalyzer()
        self.directory = SimpleFSDirectory(Paths.get(index_dir))
        self.searcher = IndexSearcher(DirectoryReader.open(self.directory))
        self.max_results = 2
        self.fields = [
            "drug_name", "drug_details", "product_introduction",
            "uses_and_benefits", "side_effects", "how_to_use",
            "how_drug_works", "safety_advice", "missed_dose",
            "expert_advice", "fact_box", "drug_interaction", "faq"
        ]

    def basic_search(self, query_text, field="product_introduction", max_results=3):
        """Simple single-field search"""
        print(f"Performing Basic Search on field: {field}")
        parser = QueryParser(field, self.analyzer)
        query = parser.parse(query_text)
        print(f'Basic Query: {query}')
        return self._execute_search(query, max_results)

    def boolean_search(self, queries, fields, operators, max_results=3):
        """Boolean search with AND, OR, NOT operators"""
        boolean_query = BooleanQuery.Builder()
        
        for query_text, field, operator in zip(queries, fields, operators):
            parser = QueryParser(field, self.analyzer)
            query = parser.parse(query_text)
            print(f"Query: {query} with operator: {operator}")

            if operator == "AND":
                occur = BooleanClause.Occur.MUST
            elif operator == "OR":
                occur = BooleanClause.Occur.SHOULD
            elif operator == "NOT":
                occur = BooleanClause.Occur.MUST_NOT
                
            boolean_query.add(query, occur)

        print(f'Boolean Query: {boolean_query}')
        return self._execute_search(boolean_query.build(), max_results)

    def proximity_search(self, terms, field, slop, max_results=3):
        """Proximity search for terms within specified distance"""
        builder = PhraseQuery.Builder()
        position = 0
        for term in terms:
            builder.add(Term(field, term.lower()), position)
            position += 1
        builder.setSlop(slop)
        query = builder.build()
        
        print(f"Proximity Query: {query}")
        return self._execute_search(query, max_results)

    def wildcard_search(self, field, pattern, max_results=3):
        """Wildcard search using * and ? patterns"""
        query = WildcardQuery(Term(field, pattern))
        
        print(f"Wildcard Query: {query}")
        return self._execute_search(query, max_results)

    def phrase_search(self, phrase, field="product_introduction", max_results=3):
        """Search for exact phrases within specified field"""
        builder = PhraseQuery.Builder()
        terms = phrase.lower().split()
        position = 0
        for term in terms:
            builder.add(Term(field, term), position)
            position += 1
        query = builder.build()
        
        print(f"Phrase Query: {query}")
        return self._execute_search(query, max_results)

    def fuzzy_search(self, term, field="product_introduction", max_edit_distance=2, max_results=3):
        """Fuzzy search for approximate matches"""
        query = FuzzyQuery(Term(field, term.lower()), max_edit_distance)
        
        print(f"Fuzzy Query: {query}")
        return self._execute_search(query, max_results)

    def grouped_search(self, query_groups, max_results=3):
        """Perform grouped boolean queries"""
        main_query = BooleanQuery.Builder()
        for group in query_groups:
            group_query = BooleanQuery.Builder()
            occur = BooleanClause.Occur.MUST if group['operator'] == 'AND' else BooleanClause.Occur.SHOULD
            
            for term, field in zip(group['terms'], group['fields']):
                parser = QueryParser(field, self.analyzer)
                parsed_query = parser.parse(term)
                group_query.add(parsed_query, occur)
                
            group_occur = BooleanClause.Occur.MUST if group['group_operator'] == 'AND' else BooleanClause.Occur.SHOULD
            main_query.add(group_query.build(), group_occur)

        print(f"Grouped Boolean Query: {main_query}")
        return self._execute_search(main_query.build(), max_results)

    def field_group_search(self, field, terms=None, must_match=None, must_not_match=None, max_results=3):
        """Search within a specific field with grouped terms"""
        boolean_query = BooleanQuery.Builder()
        parser = QueryParser(field, self.analyzer)

        if must_match:
            for term in must_match:
                query = parser.parse(term)
                boolean_query.add(query, BooleanClause.Occur.MUST)

        if terms:
            for term in terms:
                query = parser.parse(term)
                boolean_query.add(query, BooleanClause.Occur.SHOULD)

        if must_not_match:
            for term in must_not_match:
                query = parser.parse(term)
                boolean_query.add(query, BooleanClause.Occur.MUST_NOT)

        print(f"Field Group Query: {boolean_query}")
        return self._execute_search(boolean_query.build(), max_results)

    def _execute_search(self, query, max_results):
        """Execute search and format results"""
        hits = self.searcher.search(query, max_results)
        results = []

        if len(hits.scoreDocs) == 0:
            print("No result for query")
            return results  # Return empty list

        for hit in hits.scoreDocs:
            doc = self.searcher.doc(hit.doc)
            result = {field: doc.get(field) for field in self.fields}
            result['Score'] = hit.score
            results.append(result)

        return results

    def interactive_search(self):
        def select_field():
            print("Select a field to search in:")
            for i, field in enumerate(self.fields, start=1):
                print(f"{i}. {field}")
            while True:
                field_choice = input("Enter the number corresponding to your field choice: ").strip()
                if field_choice.isdigit() and 1 <= int(field_choice) <= len(self.fields):
                    return self.fields[int(field_choice) - 1]
                print("Invalid field selection. Please try again.")
        
        self.select_field = select_field
        searcher = self
        query_types = {
            "1": "Basic Search",
            "2": "Boolean Search",
            "3": "Proximity Search",
            "4": "Wildcard Search",
            "5": "Phrase Search",
            "6": "Fuzzy Search",
            "7": "Grouped Search",
            "8": "Field Group Search"
        }
        
        while True:
            print("\nSelect query type:")
            for key, value in query_types.items():
                print(f"{key}. {value}")
            query_choice = input("Enter the number corresponding to your query type: ").strip()
            if query_choice in query_types:
                break
            print("Invalid choice. Please try again.")

        if query_choice == "1":  # Basic Search
            query_text = input("Enter your search query: ").strip()
            field = self.select_field() or "product_introduction"
            results = searcher.basic_search(query_text, field)
        
        elif query_choice == "2":  # Boolean Search
            num_terms = int(input("Enter the number of terms: "))
            queries, fields, operators = [], [], []
            for i in range(num_terms):
                queries.append(input(f"Enter term {i+1}: ").strip())
                fields.append(self.select_field())
                if i < num_terms - 1:
                    operators.append(input(f"Enter operator (AND/OR/NOT) after term {i+1}: ").strip().upper())
            results = searcher.boolean_search(queries, fields, operators)
        
        elif query_choice == "3":  # Proximity Search
            terms = input("Enter words separated by spaces: ").strip().split()
            field = self.select_field()
            slop = int(input("Enter proximity distance: "))
            results = searcher.proximity_search(terms, field, slop)
        
        elif query_choice == "4":  # Wildcard Search
            field = self.select_field()
            pattern = input("Enter wildcard pattern (e.g., 'Vriace*'): ").strip()
            results = searcher.wildcard_search(field, pattern)
        
        elif query_choice == "5":  # Phrase Search
            phrase = input("Enter the exact phrase to search: ").strip()
            field = self.select_field() or "product_introduction"
            results = searcher.phrase_search(phrase, field)
        
        elif query_choice == "6":  # Fuzzy Search
            term = input("Enter the term for fuzzy search: ").strip()
            field = self.select_field() or "product_introduction"
            max_edit_distance = int(input("Enter max edit distance (1 or 2): "))
            results = searcher.fuzzy_search(term, field, max_edit_distance)
        
        elif query_choice == "7":  # Grouped Search
            num_groups = int(input("Enter number of query groups: "))
            query_groups = []
            for i in range(num_groups):
                num_terms = int(input(f"Enter number of terms in group {i+1}: "))
                terms, fields = [], []
                for j in range(num_terms):
                    terms.append(input(f"Enter term {j+1} for group {i+1}: ").strip())
                    
                    print("Select a field to search in:")
                    all_fields = {}
                    for k, field in enumerate(self.fields, start=1):
                        all_fields[f'{k}'] = field
                        print(f"{k}. {field}")
                    
                    fields.append(all_fields[input(f"Enter field for term {j+1} in group {i+1}: ").strip()])
                operator = input("Enter operator between terms (AND/OR): ").strip().upper()
                group_operator = input("Enter operator to connect this group with others (AND/OR): ").strip().upper()
                query_groups.append({'terms': terms, 'fields': fields, 'operator': operator, 'group_operator': group_operator})
                
            results = searcher.grouped_search(query_groups)
        
        elif query_choice == "8":  # Field Group Search
            field = self.select_field()
            must_match = input("Enter must-match terms (comma separated, optional): ").strip().split(",") if input("Add must-match terms? (y/n): ").strip().lower() == "y" else None
            terms = input("Enter optional terms (comma separated, optional): ").strip().split(",") if input("Add optional terms? (y/n): ").strip().lower() == "y" else None
            must_not_match = input("Enter must-not-match terms (comma separated, optional): ").strip().split(",") if input("Add must-not-match terms? (y/n): ").strip().lower() == "y" else None
            results = searcher.field_group_search(field, terms, must_match, must_not_match)
        
        if results:
            print("\nSearch Results:")
            doc = 1 
            for res in results:
                print(f'Documnet {doc} :')
                print(res)
                print('-'*50)
                doc += 1
        else:
            print("No result for query.")
            
if __name__ == "__main__":
    searcher = MedicineSearcher()
    searcher.interactive_search()
    