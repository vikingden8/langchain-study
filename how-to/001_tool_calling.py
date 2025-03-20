import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticToolsParser
from pydantic import BaseModel, Field


# take environment variables from .env.
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))  


ALIYUN_API_KEY = os.environ["ALIYUN_API_KEY"]


# The function name, type hints, and docstring are all part of the tool
# schema that's passed to the model. Defining good, descriptive schemas
# is an extension of prompt engineering and is an important part of
# getting models to perform well.
def Add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b


def Multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b


class add(BaseModel):
    """Add two integers."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


class multiply(BaseModel):
    """Multiply two integers."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


def main():
    llm = ChatOpenAI(model="qwen-max",
                     api_key=ALIYUN_API_KEY,
                     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
                     temperature=0)
    tools = [Add, Multiply]
    llm_with_tools = llm.bind_tools(tools)
    query = "What is 3 * 12?"
    # result = llm_with_tools.invoke(query)
    # print(result)

    query = "What is 3 * 12? Also, what is 11 + 49?"
    # tool_calls = llm_with_tools.invoke(query).tool_calls
    # print(tool_calls)

    
    chain = llm_with_tools | PydanticToolsParser(tools=[Add, Multiply])
    result = chain.invoke(query)
    print(result)


if __name__ == "__main__":
    main()
