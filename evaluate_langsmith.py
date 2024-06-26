from langsmith.evaluation import evaluate
from langchain_community.adapters.openai import convert_openai_messages
from custom_evaluator import accuracy
from langgraph_start import graph
from langsmith import Client
import uuid

client = Client()


def predict(inputs: dict):
    """Use this for answer evaluation"""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    msg = {
        "messages": [
            ("user", inputs["question"]),
        ]
    }
    graph.update_state(
        config, {"messages": [*convert_openai_messages(inputs["chat_history"])]}
    )
    messages = graph.invoke(msg, config=config)
    return {"response": messages["messages"][-1].content}


def main():
    dataset_name = "ConvFinQA_20240625_163835"
    examples = client.list_examples(
        dataset_name=dataset_name,
        splits=["eval"],
    )
    results = evaluate(
        predict,
        data=examples,
        experiment_prefix="Chat Single Turn",
        evaluators=[accuracy],
        metadata={"model": "gpt-4o"},
    )


if __name__ == "__main__":
    main()
