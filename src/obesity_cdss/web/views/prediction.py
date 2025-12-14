from collections import defaultdict

import streamlit as st

from obesity_cdss.web.api_client import ApiClient
from obesity_cdss.web.report import generate_pdf_bytes
from obesity_cdss.web.utils import (
    get_value_translations_and_order,
    translate_single_value,
)


def render_prediction_view() -> None:
    """Renders form for patient data and displays prediction results."""
    st.header(":primary[:material/assignment_add:] Predição clínica")
    explanation_metrics = """
    Este sistema preditivo foi criado com o objetivo de auxiliar equipes médicas
    a **:primary[prever o risco de um paciente desenvolver obesidade]**.
    Para fazer essa predição, optamos por remover as variáveis de peso e altura
    para focar em fatores que as influenciam diretamente,
    como **:primary[estilo de vida e hábitos alimentares]**.
    """
    st.markdown(explanation_metrics)
    st.space("small")
    st.markdown("Preencha as informações do paciente nas etapas abaixo.")

    data_dict = ApiClient.get_data_dictionary()
    if not data_dict:
        st.warning("Aguardando conexão com a API...")
        return

    if "form_data" not in st.session_state:
        st.session_state.form_data = {}

    groups = defaultdict(list)
    for key, meta in data_dict.items():
        if meta["type"] == "target":
            continue

        group_name = meta.get("group", "Outros")
        groups[group_name].append((key, meta))

    tabs_names = list(groups.keys())
    tabs = st.tabs(tabs_names)

    for tab, group_name in zip(tabs, tabs_names, strict=True):
        with tab:
            st.subheader(group_name)
            fields = groups[group_name]
            cols = st.columns(2)

            for i, (key, meta) in enumerate(fields):
                label = meta["description"]
                field_type = meta["type"]
                col = cols[i % 2]

                with col:
                    if field_type in ["cat_nominal", "cat_ordinal", "numeric_discrete"]:
                        val_map_en_pt, display_options = (
                            get_value_translations_and_order(data_dict, key)
                        )
                        options_map = {v: k for k, v in val_map_en_pt.items()}

                        selected = st.selectbox(
                            label,
                            options=display_options,
                            key=f"input_{key}",
                            placeholder="Selecione uma opção",
                        )
                        st.session_state.form_data[key] = options_map[selected]

                        if field_type == "numeric_discrete":
                            st.session_state.form_data[key] = float(
                                st.session_state.form_data[key]
                            )

                    elif field_type == "numeric_continuous":
                        if key == "age":
                            st.session_state.form_data[key] = st.number_input(
                                label,
                                min_value=1,
                                max_value=120,
                                step=1,
                                key=f"input_{key}",
                            )
                        else:
                            st.session_state.form_data[key] = st.number_input(
                                label,
                                value=0.0,
                                step=0.1,
                                key=f"input_{key}",
                                disabled=True,
                            )

    st.divider()

    if st.button(
        "Prever", type="primary", width="stretch", icon=":material/lightbulb_2:"
    ):
        with st.spinner("Analisando dados..."):
            result = ApiClient.predict(st.session_state.form_data)

        if result:
            label_pt = translate_single_value(data_dict, "obesity", result["label"])

            st.success("Análise concluída com sucesso!")

            c1, c2, c3 = st.columns(3)
            c1.metric("Classificação", label_pt)
            c2.metric("Probabilidade", f"{result['probability']:.1%}")
            c3.metric("Modelo", result["model_version"])

            pdf_bytes = generate_pdf_bytes(
                form_data=st.session_state.form_data, result=result, data_dict=data_dict
            )

            st.download_button(
                label="Baixar relatório médico (PDF)",
                data=pdf_bytes,
                file_name=f"relatorio_obesidade_{result['label']}.pdf",
                mime="application/pdf",
                type="secondary",
                icon=":material/download:",
            )
