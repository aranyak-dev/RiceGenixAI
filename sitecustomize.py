"""RiceGenixAI Streamlit HTML compatibility patch.

Streamlit renders complex multiline HTML more reliably through st.html than
through st.markdown(..., unsafe_allow_html=True). This compatibility shim
keeps normal Markdown untouched and routes only HTML-bearing calls to st.html.
"""

import streamlit as st

_original_markdown = st.markdown


def _ricegenix_markdown(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html") and isinstance(body, str):
        html = body.lstrip()
        if any(token in html for token in ("<div", "<style", "<audio")):
            html_renderer = getattr(st, "html", None)
            if html_renderer is not None:
                try:
                    return html_renderer(html)
                except Exception:
                    pass
    return _original_markdown(body, *args, **kwargs)


st.markdown = _ricegenix_markdown
