from src.agents.base_subgraph.base_subgraph import BaseSubgraphFabric
from src.state.state import LinuxHelperState
from src.utils.yaml_reader import read_yaml
from src.utils.project_path import Project
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from src.agents.doc_searcher.doc_searcher import DocSearcherSubgraphFabric
from src.agents.answer_generator.answer_generator import AnswerGeneratorSubgraphFabric


class MainGraphFabric(BaseSubgraphFabric):
    def __init__(self):
        self.doc_searcher = DocSearcherSubgraphFabric().get_subgraph()
        self.answer_generator = AnswerGeneratorSubgraphFabric().get_subgraph()

    def get_subgraph(self):
        graph = StateGraph(LinuxHelperState)
        graph.add_node("answer_generator", self.answer_generator)
        graph.add_node("doc_searcher", self.doc_searcher)
        graph.add_edge(START, "doc_searcher")
        graph.add_edge("doc_searcher", "answer_generator")
        graph.add_edge("answer_generator", END)
        return graph.compile()

    def compile(self):
        self.graph = self.get_subgraph()

    def infer(self, message):
        message = HumanMessage(content=message)
        res = self.graph.invoke({"messages": message})
        return res["answer"].content
