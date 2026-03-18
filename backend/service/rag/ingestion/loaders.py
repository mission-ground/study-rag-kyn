import pdfplumber
import re

class PdfLoader:

    def load(self, path):
        documents = []

        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    page_text = re.sub(r"\s+", " ", page_text) # 줄바꿈 / 탭 / 여러 공백을 전부 하나의 공백으로 정리
                    documents.append(page_text)

        return documents
    
# 실행
# loader =PdfLoader()
# text = loader.load("backend/data/raw/pdf/북브리프_돈의심리학.pdf")
# print(text)