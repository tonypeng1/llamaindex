from keybert import KeyBERT

# query_str = "Tell me about his school days?"
# query_str = "What are the things that are mentioned about Sam Altman?"

# query_str = "Tell me about the school days of the author of this essay."
# query_str = "What are the things that are mentioned about Sam Altman?"
query_str = "What happened at Interleaf and Viaweb?"

# Initialize KeyBERT model
kw_model = KeyBERT()

# Extract keywords
keywords = kw_model.extract_keywords(query_str, keyphrase_ngram_range=(1, 2), top_n=3)
# keywords = kw_model.extract_keywords(text, top_n=2)

print(keywords)

print(query_str)