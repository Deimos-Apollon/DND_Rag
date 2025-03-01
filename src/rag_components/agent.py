import os

from dotenv import load_dotenv
from llama_index.core import PromptTemplate, Settings
from llama_index.llms.gigachat import GigaChatLLM

from src.rag_components.generation.query_engine import get_query_engine
from src.rag_components.retrieval.retriever_rulebook import get_index


load_dotenv()


# Создаем кастомный промпт
qa_prompt_template = PromptTemplate(
    "Контекстная информация ниже:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Ты ассистент для игры Dungeons And Dragons. Используя контекстную информацию, ты должен давать четкий ответ на вопрос "
    "пользователя. Если в контекстной информации нет релевантной информации, напиши, что ты не знаешь ответ. "
    "Ответь на вопрос максимально подробно и структурированно."
    "Вопрос: {query_str}\n"
    "Ответ: "
)


class Agent:
    def __init__(self, llm=None):
        if not llm:
            llm = GigaChatLLM(
                credentials=os.getenv('gigachat_credentials'),
                verify_ssl_certs=False
            )
            Settings.llm = llm
            Settings.context_window = 8192
            Settings.num_output = 2048
            
        self.llm = llm
        self.index = get_index(    
            path_to_data_dir="data/docs",
            use_faiss=True,
            faiss_persist_dir="data/vector_store" 
        )
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
            text_qa_template=qa_prompt_template
        )

    def answer(self, query):
        response = self.query_engine.query(
            query
        )
        context = "\nNODE\n" + "\nNODE\n".join([node.dict()['node']['text'] for node in response.source_nodes])
        print(f'\n\nNew query: {query}')
        print(response)
        print(context)
        return response.response

