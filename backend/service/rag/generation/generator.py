from openai import OpenAI
from backend.core.config import GROQ_API_KEY, BASE_URL, LLM_MODEL

#LLM을 호출하는 부분
class Generator:

    def __init__(self):
        self.client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url=BASE_URL
        )
        self.tool_map = {
        "get_time": self.get_time
    }

    def rewrite_query(self, query):

    
        rewrite_prompt = f"""
        다음 질문을 문서 검색에 적합한 문장으로 다시 써라. 단어 키워드를 잘 챙겨라.
        질문: {query}
        """

        rewritten_response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
            {"role": "system", "content": "너는 검색 쿼리를 재작성하는 역할이다."},
            {"role": "user", "content": rewrite_prompt}
            ]
        )
        rewritten_query = rewritten_response.choices[0].message.content.strip()

        print("◎rewritten_query : "+rewritten_query)
        return rewritten_query
        
 
    def build_messages(self, query, context):

        system_messages = {
                "role": "system",
                "content": "존대말 사용. 정보 탐색 질문이 아닐경우 [문서]를 참고하지 말고 [질문]에만 답해라."
            }
        

        # RAG 프롬프트
        rag_prompt = f"""
        아래 문서를 참고해서 질문에 답해라.
        문서에 없는 내용은 추측하지 마라.

        [문서]
        {context}

        [질문]
        {query}
        """

        user_messages = {"role": "user", "content": rag_prompt}

        return [system_messages, user_messages]

    def build_simple_messages(self, query):
          return [
        {
            "role": "system",
            "content": "존대말로 답변해라."
        },
        {
            "role": "user",
            "content": query
        }
    ]

    def call_llm(self, messages, tools=None):
        kwargs={
            "model":LLM_MODEL,
            "messages":messages
        }

        if tools: # 상황에 따라 파라미터를 동적으로 추가
            kwargs["tools"]=tools # 콤마쓰지 않는다. 튜플이 되어버려 에러
            kwargs["tool_choice"]="auto"

        return self.client.chat.completions.create(**kwargs)

    def execute_tool(self, tool_call):
        tool_name = tool_call.function.name

        if tool_name in self.tool_map:
            return self.tool_map[tool_name]()
        
        raise ValueError(f"Unknown tool: {tool_name}")
    
    def handle_tool_calls(self, messages, ai_message):

        if not ai_message.tool_calls:
            return ai_message.content
        
        tool_call = ai_message.tool_calls[0]
        result = self.execute_tool(tool_call)
          

        # 도구 결과를 messages에 다시 추가
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result) if result else "" # None 방지
        })

        # AI를 한 번 더 호출
        second_response = self.call_llm(messages)

        return second_response.choices[0].message.content
               


    def generate(self, query, context):

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "현재 시간을 알려준다",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]
        messages = self.build_messages(query, context)
            
        response = self.call_llm(messages, tools)
        ai_message = response.choices[0].message

        messages.append({
            "role": "assistant", 
            "content": ai_message.content,
            "tool_calls": ai_message.tool_calls
            })

        return self.handle_tool_calls(messages, ai_message)

    def get_time(self):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def is_retrieval_query(self, query):
        keywords = ["문서", "설명", "내용", "무엇", "왜", "어떻게"]
        # 나중에 LLM으로 판단하게 바꾼다
        return any(keyword in query for keyword in keywords)
