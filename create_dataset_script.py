import json
from langsmith import Client
from datetime import datetime

client = Client()


def load_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def create_chat_history(annotation, qa, pre_text, post_text, table):
    chat_history = []
    dialogue_break = annotation.get("dialogue_break", [])
    exe_ans_list = qa.get("answer", [])

    for i, (question, answer) in enumerate(zip(dialogue_break, exe_ans_list)):
        if i == 0:  # Include pre_text, post_text, and table in the first user question
            chat_history.append(
                {
                    "role": "user",
                    "content": f"{pre_text}\n{post_text}\n{table}\n{question}",
                }
            )
        else:
            chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": str(answer)})

    return chat_history, dialogue_break, exe_ans_list


def create_examples(data):
    examples = []
    for item in data:
        annotation = item.get("annotation", {})
        qa = item.get("qa", {})
        pre_text = item.get("pre_text", "")
        post_text = item.get("post_text", "")
        table = item.get("table", [])

        if annotation.get("dialogue_break") and qa.get("answer"):
            chat_history, dialogue_break, exe_ans_list = create_chat_history(
                annotation, qa, pre_text, post_text, table
            )
            examples.append(
                {
                    "inputs": {
                        "question": dialogue_break[
                            -1
                        ],  # The last question in the dialogue
                        "chat_history": chat_history[
                            :-2
                        ],  # Exclude the last user question and assistant response
                    },
                    "outputs": {
                        "expected": str(
                            exe_ans_list[-1]
                        )  # The last answer in the dialogue
                    },
                    "metadata": {"id": item["id"]},
                }
            )
    return examples


def main():
    dataset_name = f"ConvFinQA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data = load_data("data/train.json")
    dataset = client.create_dataset(dataset_name)
    examples = create_examples(data)

    client.create_examples(
        inputs=[example["inputs"] for example in examples],
        outputs=[example["outputs"] for example in examples],
        metadata=[example["metadata"] for example in examples],
        dataset_id=dataset.id,
    )


if __name__ == "__main__":
    main()
