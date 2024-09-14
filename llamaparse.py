





# Set OpenAI API key, LLM, and embedding model
# openai.api_key = os.environ['OPENAI_API_KEY']
# llm = OpenAI(model="gpt-3.5-turbo", temperature=0.0)
# Settings.llm = llm

openai.api_key = os.environ['MISTRAL_API_KEY']
llm = MistralAI(model="mistral-large-latest", temperature=0.0)
Settings.llm = llm