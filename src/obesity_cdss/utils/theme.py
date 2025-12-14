import plotly.graph_objects as go
import plotly.io as pio

CUSTOM_PALETTE = {
    "primary": "#F23064",
    "secondary": "#BF3B5E",
    "tertiary": "#FEFA01",
    "quartenary": "#FD7223",
    "quinary": "#7CFCBF",
    "senary": "#8A0858",
    "septanary": "#2408CD",
    "background": "#262626",
    "secondary_background": "#404040",
    "tertiary_background": "#8D8D8D",
    "text": "#FAFAFA",
    "success": "#7CFCBF",
    "error": "#F23064",
}

OBESITY_COLORSCALE = [
    [0.0, CUSTOM_PALETTE["senary"]],
    [0.5, CUSTOM_PALETTE["primary"]],
    [1.0, CUSTOM_PALETTE["tertiary"]],
]


def apply_plotly_theme() -> None:
    """Configures a Plotly global template to follow a custom palette."""
    custom_template = go.layout.Template()

    custom_template.layout = dict(
        font=dict(family="sans-serif", size=18, color=CUSTOM_PALETTE["text"]),
        paper_bgcolor=CUSTOM_PALETTE["background"],
        plot_bgcolor=CUSTOM_PALETTE["background"],
        title=dict(font=dict(size=22, color=CUSTOM_PALETTE["primary"])),
        xaxis=dict(
            showgrid=True,
            gridcolor=CUSTOM_PALETTE["secondary_background"],
            gridwidth=1,
            zeroline=False,
            showline=False,
            tickfont=dict(color=CUSTOM_PALETTE["tertiary_background"]),
            title=dict(font=dict(color=CUSTOM_PALETTE["text"])),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=CUSTOM_PALETTE["secondary_background"],
            gridwidth=1,
            zeroline=False,
            showline=False,
            tickfont=dict(color=CUSTOM_PALETTE["tertiary_background"]),
            title=dict(font=dict(color=CUSTOM_PALETTE["text"])),
        ),
        legend=dict(
            bgcolor=CUSTOM_PALETTE["secondary_background"],
            bordercolor=CUSTOM_PALETTE["text"],
            borderwidth=0,
            font=dict(color=CUSTOM_PALETTE["text"]),
        ),
        colorscale=dict(
            sequential=OBESITY_COLORSCALE,
            diverging=OBESITY_COLORSCALE,
            sequentialminus=OBESITY_COLORSCALE,
        ),
        colorway=[
            CUSTOM_PALETTE["primary"],
            CUSTOM_PALETTE["secondary"],
            CUSTOM_PALETTE["tertiary"],
            CUSTOM_PALETTE["quinary"],
            CUSTOM_PALETTE["quartenary"],
            CUSTOM_PALETTE["senary"],
            CUSTOM_PALETTE["septanary"],
        ],
    )

    pio.templates["obesity_dark"] = custom_template
    pio.templates.default = "obesity_dark"
