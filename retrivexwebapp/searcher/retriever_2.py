import json
from elasticsearch import Elasticsearch

class MedicineSearcher:
    def __init__(self, index="medicine_index_2"):
        self.index = index
        self.es = Elasticsearch("https://localhost:9200",verify_certs = False,basic_auth=('elastic','rF*c2rkY=X_tFWudh-j5'))
        self.max_results = 3
        self.fields = [
            "drug_name", "drug_details", "product_introduction",
            "uses_and_benefits", "side_effects", "how_to_use",
            "how_drug_works", "safety_advice", "missed_dose",
            "expert_advice", "fact_box", "drug_interaction", "faq"
        ]
    
    def basic_search(self, query_text, field="product_introduction", max_results=3):
        print(f"Performing Basic Search on field: {field}")
        query = {
            "match": {
                field: query_text
            }
        }
        print(f'Basic Query: {query}')
        return self._execute_search(query, max_results)
    
    def boolean_search(self, queries, fields, operators, max_results=3):
        bool_query = {"bool": {}}
        # Build lists for each clause
        must = []
        should = []
        must_not = []
        
        for query_text, field, operator in zip(queries, fields, operators):
            q = {"match": {field: query_text}}
            print(f"Query: {q} with operator: {operator}")
            if operator.upper() == "AND":
                must.append(q)
            elif operator.upper() == "OR":
                should.append(q)
            elif operator.upper() == "NOT":
                must_not.append(q)
        
        if must:
            bool_query["bool"]["must"] = must
        if should:
            bool_query["bool"]["should"] = should
        if must_not:
            bool_query["bool"]["must_not"] = must_not
        
        print(f'Boolean Query: {bool_query}')
        return self._execute_search(bool_query, max_results)
    
    def proximity_search(self, terms, field, slop, max_results=3):
        phrase = " ".join(terms)
        query = {
            "match_phrase": {
                field: {
                    "query": phrase,
                    "slop": slop
                }
            }
        }
        print(f"Proximity Query: {query}")
        return self._execute_search(query, max_results)
    
    def wildcard_search(self, field, pattern, max_results=3):
        query = {
            "wildcard": {
                field: {
                    "value": pattern
                }
            }
        }
        print(f"Wildcard Query: {query}")
        return self._execute_search(query, max_results)
    
    def phrase_search(self, phrase, field="product_introduction", max_results=3):
        query = {
            "match_phrase": {
                field: phrase
            }
        }
        print(f"Phrase Query: {query}")
        return self._execute_search(query, max_results)
    
    def fuzzy_search(self, term, field="product_introduction", max_edit_distance=2, max_results=3):
        query = {
            "fuzzy": {
                field: {
                    "value": term,
                    "fuzziness": max_edit_distance
                }
            }
        }
        print(f"Fuzzy Query: {query}")
        return self._execute_search(query, max_results)
    
    def grouped_search(self, query_groups, max_results=3):
        # Outer boolean query to combine groups
        outer_bool = {"bool": {}}
        must_groups = []
        should_groups = []
        
        for group in query_groups:
            inner_bool = {"bool": {}}
            terms = group['terms']
            fields = group['fields']
            op = group['operator'].upper()
            # Build inner group query
            if op == "AND":
                inner_bool["bool"]["must"] = [{"match": {field: term}} for term, field in zip(terms, fields)]
            else:
                inner_bool["bool"]["should"] = [{"match": {field: term}} for term, field in zip(terms, fields)]
            
            group_op = group['group_operator'].upper()
            if group_op == "AND":
                must_groups.append(inner_bool)
            else:
                should_groups.append(inner_bool)
        
        if must_groups:
            outer_bool["bool"]["must"] = must_groups
        if should_groups:
            outer_bool["bool"]["should"] = should_groups
        
        print(f"Grouped Boolean Query: {outer_bool}")
        return self._execute_search(outer_bool, max_results)
    
    def field_group_search(self, field, terms=None, must_match=None, must_not_match=None, max_results=3):
        bool_query = {"bool": {}}
        if must_match:
            bool_query["bool"]["must"] = [{"match": {field: term}} for term in must_match if term]
        if terms:
            bool_query["bool"]["should"] = [{"match": {field: term}} for term in terms if term]
        if must_not_match:
            bool_query["bool"]["must_not"] = [{"match": {field: term}} for term in must_not_match if term]
        print(f"Field Group Query: {bool_query}")
        return self._execute_search(bool_query, max_results)
    
    def _execute_search(self, query, max_results):
        body = {
            "query": query,
            "size": max_results
        }
        res = self.es.search(index=self.index, body=body)
        hits = res.get('hits', {}).get('hits', [])
        results = []
        if not hits:
            print("No result for query")
            return results
        for hit in hits:
            doc = hit.get('_source', {})
            doc['Score'] = hit.get('_score')
            results.append(doc)
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
            queries = []
            fields = []
            operators = []
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
            max_edit_distance = int(input("Enter max edit distance (e.g., 1 or 2): "))
            results = searcher.fuzzy_search(term, field, max_edit_distance)
        
        elif query_choice == "7":  # Grouped Search
            num_groups = int(input("Enter number of query groups: "))
            query_groups = []
            for i in range(num_groups):
                num_terms = int(input(f"Enter number of terms in group {i+1}: "))
                terms = []
                fields = []
                for j in range(num_terms):
                    terms.append(input(f"Enter term {j+1} for group {i+1}: ").strip())
                    print("Select a field to search in:")
                    all_fields = {}
                    for k, field in enumerate(self.fields, start=1):
                        all_fields[str(k)] = field
                        print(f"{k}. {field}")
                    fields.append(all_fields[input(f"Enter field for term {j+1} in group {i+1}: ").strip()])
                operator = input("Enter operator between terms (AND/OR): ").strip().upper()
                group_operator = input("Enter operator to connect this group with others (AND/OR): ").strip().upper()
                query_groups.append({'terms': terms, 'fields': fields, 'operator': operator, 'group_operator': group_operator})
            results = searcher.grouped_search(query_groups)
        
        elif query_choice == "8":  # Field Group Search
            field = self.select_field()
            must_match = None
            if input("Add must-match terms? (y/n): ").strip().lower() == "y":
                must_match = [t.strip() for t in input("Enter must-match terms (comma separated): ").split(",")]
            terms = None
            if input("Add optional terms? (y/n): ").strip().lower() == "y":
                terms = [t.strip() for t in input("Enter optional terms (comma separated): ").split(",")]
            must_not_match = None
            if input("Add must-not-match terms? (y/n): ").strip().lower() == "y":
                must_not_match = [t.strip() for t in input("Enter must-not-match terms (comma separated): ").split(",")]
            results = searcher.field_group_search(field, terms, must_match, must_not_match)
        
        if results:
            print("\nSearch Results:")
            for idx, res in enumerate(results, start=1):
                print(f'Document {idx}:')
                print(json.dumps(res, indent=2))
                print('-'*50)
        else:
            print("No result for query.")

if __name__ == "__main__":
    searcher = MedicineSearcher()
    searcher.interactive_search()
