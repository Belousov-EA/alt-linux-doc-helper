from pathlib import Path
from enum import Enum
from typing import Union


class Project(str, Enum):

    def __new__(cls, value: Union[str, Path]) -> "Project":
        obj = str.__new__(cls, value)
        obj._value_ = str(value)
        return obj

    @property
    def path(self) -> Path:
        return Path(self.value)

    def __truediv__(self, other: Union[str, Path]) -> Path:
        return self.path / str(other)

    # base dirrectories
    ROOT = Path(__file__).parent.parent.parent.resolve()
    SRC = ROOT / "src"
    AGENTS = SRC / "agents"
    TRACE = SRC / "trace"
    DATA = ROOT / "data"

    # agents
    ANSWER_GENERATOR_AGENT = AGENTS / "answer_generator"
    DOC_SEARCHER_AGENT = AGENTS / "doc_searcher"
