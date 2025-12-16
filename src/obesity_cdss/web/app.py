import streamlit as st

from obesity_cdss.utils.theme import apply_plotly_theme
from obesity_cdss.web.views import insights, metrics, prediction

st.set_page_config(
    page_title="Obesity Risk CDSS",
    page_icon=":material/body_fat:",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_plotly_theme()


def main() -> None:
    if "current_page" not in st.session_state:
        st.session_state.current_page = "prediction"

    def set_page(page_name: str) -> None:
        st.session_state.current_page = page_name

    with st.sidebar:
        st.title(":primary[:material/body_fat:] Preditor de Obesidade")
        st.caption("Sistema de Suporte à Decisão Clínica")

        st.divider()

        st.markdown("**Menu de navegação**")

        st.button(
            "Predição clínica",
            width="stretch",
            type="primary"
            if st.session_state.current_page == "prediction"
            else "secondary",
            icon=":material/assignment_add:",
            on_click=set_page,
            args=("prediction",),
        )

        st.button(
            "Métricas do modelo",
            width="stretch",
            type="primary"
            if st.session_state.current_page == "metrics"
            else "secondary",
            icon=":material/percent:",
            on_click=set_page,
            args=("metrics",),
        )

        st.button(
            "Painel analítico",
            width="stretch",
            type="primary"
            if st.session_state.current_page == "insights"
            else "secondary",
            icon=":material/bar_chart:",
            on_click=set_page,
            args=("insights",),
        )

        st.divider()

        st.markdown(
            ":material/code_blocks: Confira o [código-fonte](https://github.com/umpuela/obesity-cdss)"
        )

    if st.session_state.current_page == "prediction":
        prediction.render_prediction_view()

    elif st.session_state.current_page == "metrics":
        metrics.render_metrics_view()

    elif st.session_state.current_page == "insights":
        insights.render_insights_view()


if __name__ == "__main__":
    main()
