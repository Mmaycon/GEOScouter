import textwrap


def wrap_text_for_plotly(text, width=30):
    """Wrap text with HTML line breaks for Plotly tooltips."""
    if not isinstance(text, str):
        return ""
    return "<br>".join(textwrap.wrap(text, width=width))
