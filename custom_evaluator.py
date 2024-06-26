import os


from langchain_openai import ChatOpenAI
from langsmith.schemas import Example, Run
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class CustomEvaluator(BaseModel):
    """
    Use this to evaluate the model response.
    """

    is_accurate: bool = Field(
        description="A boolean value indicating whether the model's response is accurate or not. "
        "Consider that if a response is a percentage, but the reference is the equivalent "
        "as a float, it may be accurate."
    )


def accuracy(run: Run, example: Example) -> dict:
    tools = [CustomEvaluator]

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="anthropic/claude-3-haiku",
        max_tokens=1200,
    )

    eval_template = """
    Evaluate the accuracy of the response based on the reference output.
    Response should be a boolean value indicating whether the model's response
    is accurate or not. Consider that if a response is a percentage,
    but the reference is the equivalent as a float, or vice versa it may be accurate.
    If the response and the reference are similar, the response may be accurate.
    Sometimes a response might be something like 0.78 and the reference 78%, which is accurate.
    Input_query: {input_query}
    Actual_output: {actual_output}
    Expected_output: {expected_output}
    """

    prompt = ChatPromptTemplate.from_template(eval_template)
    llm_with_tools = llm.bind_tools(tools)

    chain = prompt | llm_with_tools
    input_query = run.inputs["question"]
    expected_output = example.outputs["expected"]
    actual_output = run.outputs["response"]

    result = chain.invoke(
        {
            "input_query": input_query,
            "expected_output": expected_output,
            "actual_output": actual_output,
        }
    )

    score = result.tool_calls[0]["args"].get("is_accurate", False)
    score = 1 if score else 0

    return {"key": "accuracy", "score": score}
