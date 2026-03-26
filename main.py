# main.py
# 역할: 각 클래스를 조립해서 전체 RAG 파이프라인을 실행한다.
# 전체 흐름 : PDFLoader → Chunker → Embedder → Retriever → Generator
from backend.service.rag.ingestion.index_builder import IndexBuilder
from backend.service.rag.rag_pipeline import RAGPipeline
import gradio as gr
import traceback


#1. 데이터 로드 → 청킹 → 임베딩 → 벡터DB 저장 → 검색 → 답변 생성


builder = IndexBuilder()

embedder, vector_store, docstore, embeddings, child_contents = builder.build()

pipeline = RAGPipeline(embedder, vector_store, docstore, embeddings, child_contents)


# Gradio용 함수
def chat_fn(message, history):
    try:
        answer = pipeline.ask(message)
    except Exception as e:
        answer = f"에러 발생: {e}"
        traceback.print_exc()

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    
    return history, history


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# RAG 챗봇")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="질문을 입력하세요!!")

    msg.submit(chat_fn, [msg, chatbot], [chatbot, chatbot])

# 실행
demo.launch()