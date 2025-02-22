import streamlit as st
from streamlit_extras.stylable_container import stylable_container as sc



def get_containerstyle(height: int=None, border: bool=False):
    styleouter = st.secrets.app.style1
    styleinner = st.secrets.app.style2
    outer = sc(key="outer", css_styles=styleouter)
    with outer:
        inner = sc(key="inner", css_styles=styleinner)
        with inner:
            if height is not None:
                container = st.container(height=height, border=border)
            else:
                container = st.container(border=border)
    return container