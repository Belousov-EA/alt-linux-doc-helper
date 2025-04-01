from src.agents.base_subgraph.base_subgraph import BaseSubgraphFabric
from src.state.state import LinuxHelperState
from langgraph.graph import StateGraph, START, END
from src.utils.project_path import Project
from src.utils.yaml_reader import read_yaml


class DocSearcherSubgraphFabric(BaseSubgraphFabric):
    def __init__(self):
        self.config = read_yaml(Project.DOC_SEARCHER_AGENT / "stub.yaml")

    def get_subgraph(self):
        graph = StateGraph(LinuxHelperState)
        graph.add_node("stub", self.stub)
        graph.add_edge(START, "stub")
        graph.add_edge("stub", END)
        return graph.compile()

    def stub(self, state: LinuxHelperState):
        return {"docs": self.config["docs"]}
