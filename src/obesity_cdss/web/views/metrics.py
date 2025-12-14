import pandas as pd
import plotly.express as px
import streamlit as st

from obesity_cdss.utils.theme import CUSTOM_PALETTE
from obesity_cdss.web.api_client import ApiClient
from obesity_cdss.web.utils import (
    get_name_translations,
    get_value_translations_and_order,
    highlight_cell_max,
)


def render_metrics_view() -> None:
    """Renders the model metrics in Portuguese."""
    st.header(":primary[:material/percent:] Métricas do modelo")
    st.space("small")

    metrics = ApiClient.get_model_metrics()
    data_dict = ApiClient.get_data_dictionary()
    if not metrics or not data_dict:
        st.info("Métricas ou dicionário não disponíveis. O modelo foi treinado?")
        return

    obesity_translation_map, ordered_obesity = get_value_translations_and_order(
        data_dict, "obesity"
    )

    # ==========================================================
    # KPI CARDS
    # ==========================================================
    col1, col2, col3 = st.columns(3)

    col1.metric("Algoritmo", metrics.get("model_name", "N/A"))
    col1.caption("Modelo de classificação com melhor resultado")

    acc = metrics.get("classification_report", {}).get("accuracy", 0)
    col2.metric("Acurácia global", f"{acc:.2%}")
    col2.caption("Acertos totais / Total de casos")

    roc = metrics.get("roc_auc_macro_ovr", 0)
    col3.metric("ROC AUC (Macro OvR)", f"{roc:.4f}")
    col3.caption("Área sob a curva característica de operação do receptor")

    st.divider()

    # ==========================================================
    # CLASSIFICATION REPORT
    # ==========================================================
    st.subheader("Relatório de classificação")
    st.caption("Qual foi o desempenho do modelo por perfil clínico?")

    report = metrics.get("classification_report", {})
    if report:
        df_report = pd.DataFrame(report).transpose()
        df_classes = df_report.drop(
            ["accuracy", "macro avg", "weighted avg"], errors="ignore"
        )
        df_classes.index = df_classes.index.map(obesity_translation_map)
        df_classes = df_classes.reindex(ordered_obesity)

        df_classes.columns = ["Precisão", "Recall", "Pontuação F1", "Suporte"]

        col_report, col_report_text = st.columns(
            [3, 1], gap="large", vertical_alignment="center"
        )
        with col_report:
            st.table(
                df_classes.style.format(
                    {
                        "Precisão": "{:.1%}",
                        "Recall": "{:.1%}",
                        "Pontuação F1": "{:.1%}",
                        "Suporte": "{:.0f}",
                    }
                )
                .apply(
                    highlight_cell_max,
                    subset=["Recall"],
                    bg_color=CUSTOM_PALETTE["tertiary"],
                    text_color=CUSTOM_PALETTE["background"],
                )
                .set_table_styles(
                    [
                        {
                            "selector": "th.col_heading, .index_name",
                            "props": [
                                ("background-color", CUSTOM_PALETTE["primary"]),
                                ("color", "white"),
                                ("text-align", "center !important"),
                            ],
                        },
                        {
                            "selector": "td",
                            "props": [("text-align", "center !important")],
                        },
                        {
                            "selector": "tr:hover",
                            "props": [
                                (
                                    "background-color",
                                    CUSTOM_PALETTE["secondary_background"],
                                )
                            ],
                        },
                    ]
                ),
            )

        with col_report_text:
            top_recall = df_classes["Recall"].max()
            top_recall_name = df_classes[df_classes["Recall"] == top_recall].index[0]

            worst_recall = df_classes["Recall"].min()
            worst_recall_name = df_classes[df_classes["Recall"] == worst_recall].index[
                0
            ]

            explanation_report = f"""
            O modelo identifica **:primary[{top_recall:.1%}]** dos pacientes com
            {top_recall_name} corretamente, mas tem dificuldades em identificar
            quem tem {worst_recall_name} e só os acerta **:primary[{worst_recall:.1%}]**
            das vezes.
            """
            st.markdown(explanation_report)

    st.divider()

    # ==========================================================
    # CONFUSION MATRIX
    # ==========================================================
    st.subheader("Matriz de confusão")
    st.caption("Quais perfis o modelo tem mais dificuldade de prever?")
    col_matrix_text, col_matrix = st.columns(
        [1, 3], gap="large", vertical_alignment="center"
    )

    with col_matrix:
        cm_data = metrics.get("confusion_matrix")
        if cm_data:
            df_cm = pd.DataFrame(cm_data)
            df_cm.index = df_cm.index.map(obesity_translation_map)
            df_cm.columns = df_cm.columns.map(obesity_translation_map)
            df_cm = df_cm.reindex(index=ordered_obesity, columns=ordered_obesity)

            fig_cm = px.imshow(
                df_cm,
                labels=dict(x="Predito", y="Real", color="Quantidade"),
                aspect="auto",
                text_auto=True,
            )
            fig_cm.update_layout(
                height=600,
            )
            st.plotly_chart(fig_cm, width="stretch")

    with col_matrix_text:
        explanation_matrix = """
        A concentração de valores na diagonal principal confirma
        a **:primary[alta taxa de acerto]** do sistema.
        Os erros residuais ocorrem principalmente entre classes adjacentes,
        ou seja, o modelo **:primary[não comete erros grosseiros]**.
        """
        st.markdown(explanation_matrix)

    st.divider()

    # ==========================================================
    # FEATURE IMPORTANCE
    # ==========================================================
    st.subheader("Importância das variáveis")
    st.caption("Quais perguntas do formulário mais influenciaram a decisão do modelo?")

    importances = metrics.get("feature_importances", {})

    if importances:
        df_imp = pd.DataFrame(
            list(importances.items()), columns=["Feature", "Importance"]
        )
        obesity_translation_map = get_name_translations(data_dict)

        df_imp["Feature"] = df_imp["Feature"].map(obesity_translation_map)
        df_imp = df_imp.sort_values(by="Importance", ascending=True)

        col_imp_chart, col_imp_text = st.columns(
            [3, 1], gap="large", vertical_alignment="center"
        )

        with col_imp_chart:
            fig_imp = px.bar(
                df_imp,
                x="Importance",
                y="Feature",
                orientation="h",
            )

            fig_imp.update_layout(
                xaxis_title="Poder preditivo (0-1)",
                yaxis_title="",
                showlegend=False,
                height=500,
            )

            st.plotly_chart(fig_imp, width="stretch")

        with col_imp_text:
            top_feature = df_imp.iloc[-1]["Feature"]
            explanation_imp = f"""
            A variável **:primary[{top_feature}]** demonstrou ser o indicador
            mais forte para a determinação do nível de obesidade
            neste grupo de pacientes.
            """
            st.markdown(explanation_imp)
