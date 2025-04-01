from langgraph.graph import MessagesState


class LinuxHelperState(MessagesState):
    docs: list[str]
    answer: str
