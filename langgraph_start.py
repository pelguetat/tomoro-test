from datetime import datetime
import os
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages

from langchain_core.messages import ToolMessage

from langgraph.prebuilt import ToolNode

from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool
import json

from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "tomoro-ai"


def langgraph_start(user_input):
    # Convert user input to JSON string if it's a dictionary
    if isinstance(user_input, dict):
        user_input = json.dumps(user_input)
    _printed = set()

    # Stream events from the graph
    events = graph.stream(
        {
            "messages": (
                "user",
                user_input,
            )
        },
        stream_mode="values",
    )
    for event in events:
        _print_event(event, _printed)
        messages = event.get("messages")
        if (
            messages
            and isinstance(messages, list)
            and messages[-1].type == "ai"
            and not messages[-1].tool_calls
        ):
            response = messages[-1].content
            return response


# Define a tool for running Python REPL commands
repl_tool = Tool(
    name="python_repl",
    description=(
        "Use this to do math. Input should be a valid python command. "
        "If you want to see the output of a value, you should print it out with `print(...)`."
    ),
    func=PythonREPL().run,
)


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print(f"Currently in: {current_state[-1]}")
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}

            else:
                break
        return {"messages": result}


# Initialize the language model with specific parameters
llm = ChatOpenAI(model="gpt-4o", temperature=1)

# Define the assistant's prompt template
assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a financial documentation assistant. You are tasked with responding the user questions regarding financial documentation."
                "Just respond with the number of the answer you think is correct. Omit any commentary."
                "If the question is asking for a percentage, convert the number to a percentage if necessary."
                + "\nCurrent time: {time}."
            ),
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# Combine the prompt template with the language model and bind tools
assistant_runnable = assistant_prompt | llm.bind_tools([repl_tool])

# Build the state graph
builder = StateGraph(State)

# Add nodes and edges to the graph
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("tools", ToolNode([repl_tool]))
builder.set_entry_point("assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Initialize in-memory SQLite saver and compile the graph
memory = SqliteSaver.from_conn_string(":memory:")
graph = builder.compile(checkpointer=memory)
