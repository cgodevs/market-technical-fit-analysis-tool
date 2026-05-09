from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from models.profiles import ProfessionalProfile
from config import api_key, LLM_MODEL_NAME, LLM_PROVIDER, LLM_TEMPERATURE
from utils.db_utils import get_static_list_of_industries
from exceptions import StructuredOutputParsingError

def extract_professional_structured_data(text: str) -> dict:
    try:
        llm = init_chat_model(
            model=LLM_MODEL_NAME,
            model_provider=LLM_PROVIDER,
            temperature=LLM_TEMPERATURE,
            api_key=api_key,
        )
        system_prompt = f"""
            Your role is to extract data out of a resume text provided to build it a metadata object.
            Use all sets of experiences identified to build a complete object.
            Work industries list to choose from for the main goal position: {'|'.join(get_static_list_of_industries())}
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        structured_llm = llm.with_structured_output(schema=ProfessionalProfile)
        chain = prompt | structured_llm
        response = chain.invoke({"input": text})
        return response.model_dump()
    except Exception as e:
        raise StructuredOutputParsingError(detail=str(e))
