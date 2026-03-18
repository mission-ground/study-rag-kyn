#component = 시스템을 구성하는 하나의 부품
# 텍스트를 벡터로 변경하는 클래스
from sentence_transformers import SentenceTransformer
from typing import List

# from ingestion.loaders import PdfLoader

class Embedder:

    # 생성자
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)



    # 임베딩
    def embed(self, documents: List[str]):
        return self.model.encode(documents, normalize_embeddings=True)
    


