from backend.service.rag.retrieval.retriever import Retriever
from backend.service.rag.generation.generator import Generator

# 사람으로 따지면 감독.
class RAGPipeline:

    def __init__(self, embedder, vector_store, docstore, embeddings, child_contents):

        self.retriever = Retriever(
            embedder, 
            vector_store, 
            docstore,   
            embeddings=embeddings,
            child_contents=child_contents,   
            )
        self.generator = Generator()

    def ask(self, query):

        # if self.generator.is_retrieval_query(query):
        rewritten_query = self.generator.rewrite_query(query)
        documents = self.retriever.retrieve(rewritten_query)
        context = "\n".join(documents)

        # else:
        #     context = None

        return self.generator.generate(query, context)