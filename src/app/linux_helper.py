import sys
from pathlib import Path
import streamlit as st


if "sys" not in st.session_state:
    if sys.path[-1] != str(Path(__file__).parent.parent.parent.resolve()):
        sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
    st.session_state.sys = True
    from src.agents.main_graph.main_graph import MainGraphFabric


if "graph" not in st.session_state:
    st.session_state.graph = MainGraphFabric()
    st.session_state.graph.compile()


st.title("Alt-linux helper")

user_question = st.text_input("Question", "Console isn't opening")

if st.button("Find answer"):

    res = st.session_state.graph.infer(user_question)
    st.write(res)
