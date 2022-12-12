from modules import SearchEngine

search_engine = SearchEngine()
search_engine.create_vocab()
search_engine.create_docterm_matrix()
vector_query = search_engine.vectorize("Alpinia Galanga")
result = search_engine.ranking(vector_query)[:10]
for value in result:
    print(value[1])
