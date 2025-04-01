from src.agents.base_subgraph.base_subgraph import BaseSubgraphFabric
from src.utils.yaml_reader import read_yaml
from src.utils.project_path import Project
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from src.state.state import LinuxHelperState


class AnswerGeneratorSubgraphFabric(BaseSubgraphFabric):
    def __init__(self):
        config = read_yaml(Project.ANSWER_GENERATOR_AGENT / "answer_generator.yaml")
        self.llm = ChatOllama(model=config["model"])
        self.system_prompt = config["system_prompt"]

    def get_subgraph(self):
        graph = StateGraph(LinuxHelperState)
        graph.add_node("generate_answer", self.generate_answer)
        graph.add_edge(START, "generate_answer")
        graph.add_edge("generate_answer", END)
        return graph.compile()

    def generate_answer(self, state: LinuxHelperState):
        messages = state["messages"]
        docs = state["docs"]
        return {
            "answer": self.llm.invoke(
                [SystemMessage(content=self.system_prompt.format(docs=docs))] + messages
            )
        }
