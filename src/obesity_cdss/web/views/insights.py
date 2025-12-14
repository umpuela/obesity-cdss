import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from obesity_cdss.utils.theme import CUSTOM_PALETTE
from obesity_cdss.web.api_client import ApiClient
from obesity_cdss.web.utils import (
    get_name_translations,
    get_value_translations_and_order,
)


def render_insights_view() -> None:
    """Renders the analytical dashboard for historical patient data."""
    st.header(":primary[:material/bar_chart:] Painel analítico")
    st.caption("Visão estratégica baseada nos dados históricos dos pacientes")
    st.space("small")

    response = ApiClient.get_insights_data()
    data_dict = ApiClient.get_data_dictionary()

    if not response or not data_dict:
        st.warning("Dados indisponíveis para análise no momento.")
        return

    kpis = response.get("kpis", {})
    charts_data = response.get("charts", {})
    chart_labels = get_name_translations(data_dict)

    obesity_translation_map, ordered_obesity = get_value_translations_and_order(
        data_dict, "obesity"
    )

    # ==========================================================
    # KPI CARDS
    # ==========================================================
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de pacientes", f"{kpis.get('total_patients', 0):,}")
    col2.metric("Taxa de obesidade", f"{kpis.get('obesity_rate', 0):.1%}")
    col3.metric(
        "Mediana de refeições principais", f"{kpis.get('median_ncp', 0):.0f} por dia"
    )

    st.divider()

    # ==========================================================
    # AGE PYRAMID (GENDER DISTRIBUTION)
    # ==========================================================
    st.subheader("Pirâmide etária dos pacientes")
    st.caption("Qual o perfil demográfico predominante?")
    c1, c2 = st.columns([3, 1], gap="large", vertical_alignment="center")

    with c1:
        df_pyramid = pd.DataFrame(charts_data.get("pyramid_data", []))

        if not df_pyramid.empty:
            gender_map = get_value_translations_and_order(data_dict, "gender")[0]
            df_pyramid["gender"] = df_pyramid["gender"].map(gender_map)

            labels_list = [f"{i} a {i + 4}" for i in range(0, 95, 5)]
            df_pyramid["age_bin"] = pd.cut(
                df_pyramid["age"],
                bins=range(0, 100, 5),
                right=False,
                labels=labels_list,
            )
            df_pyramid["age_bin"] = df_pyramid["age_bin"].astype(str)

            men = (
                df_pyramid[df_pyramid["gender"] == "Masculino"]
                .groupby("age_bin")
                .size()
                .reset_index(name="count")
            )
            women = (
                df_pyramid[df_pyramid["gender"] == "Feminino"]
                .groupby("age_bin")
                .size()
                .reset_index(name="count")
            )

            men["count_neg"] = men["count"] * -1

            fig_pyr = go.Figure()

            # Left size (men)
            fig_pyr.add_trace(
                go.Bar(
                    y=men["age_bin"],
                    x=men["count_neg"],
                    name="Masculino",
                    orientation="h",
                    marker_color=CUSTOM_PALETTE["tertiary"],
                    customdata=men["count_neg"].abs(),
                    hovertemplate="<b>Masculino</b><br>Faixa: %{y}<br>"
                    "Qtd: %{customdata:.0f}<extra></extra>",
                )
            )

            # Right size (women)
            fig_pyr.add_trace(
                go.Bar(
                    y=women["age_bin"],
                    x=women["count"],
                    name="Feminino",
                    orientation="h",
                    marker_color=CUSTOM_PALETTE["primary"],
                    hovertemplate="<b>Feminino</b><br>Faixa: %{y}<br>"
                    "Qtd: %{value:.0f}<extra></extra>",
                )
            )

            fig_pyr.update_layout(
                barmode="relative",
                bargap=0.1,
                xaxis=dict(
                    title="Quantidade de Pacientes",
                    tickvals=[-100, -50, 0, 50, 100],
                    ticktext=["100", "50", "0", "50", "100"],
                ),
                yaxis=dict(
                    title="Faixa Etária",
                    ticksuffix=" anos",
                ),
                legend=dict(orientation="h", title=""),
            )

            st.plotly_chart(fig_pyr, width="stretch")

    with c2:
        st.markdown(
            """
            A base de dados é predominantemente jovem, com pico de concentração
            entre :primary[20 e 24 anos].
            """
        )

    st.divider()

    # ==========================================================
    # TARGET DISTRIBUTION
    # ==========================================================
    c3, c4 = st.columns(2, gap="large")

    with c3:
        st.subheader("Distribuição de risco")
        st.caption("A variável alvo está equilibrada?")
        df_dist = pd.DataFrame(charts_data.get("distribution_obesity", []))
        if not df_dist.empty:
            df_dist["category"] = df_dist["category"].map(obesity_translation_map)

            top_cat = df_dist.loc[df_dist["count"].idxmax()]["category"]
            default_color_name = "Other"
            highlight_color_name = "Top Cat"
            df_dist["color_group"] = df_dist["category"].apply(
                lambda x: highlight_color_name if x == top_cat else default_color_name
            )
            color_map = {
                highlight_color_name: CUSTOM_PALETTE["tertiary"],
                default_color_name: CUSTOM_PALETTE["primary"],
            }

            fig_bar = px.bar(
                df_dist,
                x="category",
                y="count",
                color="color_group",
                category_orders={"category": ordered_obesity},
                color_discrete_map=color_map,
                labels={
                    **chart_labels,
                    "category": "Nível de obesidade",
                    "count": "Quantidade de pacientes",
                },
                hover_data={"color_group": False},
            )
            fig_bar.update_layout(showlegend=False, xaxis_title="")
            st.plotly_chart(fig_bar, width="stretch")

            st.markdown(
                f"""A distribuição da variável alvo está bem equilibrada.
                A maior concentração de pessoas está em **:primary[{top_cat}]**."""
            )

    # ==========================================================
    # AGE, WEIGHT, HEIGHT AND TRANSPORTATION
    # ==========================================================
    with c4:
        st.subheader("Correlação de mobilidade, peso e idade")
        st.caption("O meio de transporte utilizado impacta a saúde do paciente?")

        df_scatter = pd.DataFrame(charts_data.get("scatter_data", []))
        if not df_scatter.empty:
            mtrans_map, ordered_mtrans = get_value_translations_and_order(
                data_dict, "mtrans"
            )
            df_scatter["mtrans"] = df_scatter["mtrans"].map(mtrans_map)

            fig_scatter = px.scatter(
                df_scatter,
                x="age",
                y="weight",
                color="mtrans",
                size="height",
                category_orders={"mtrans": ordered_mtrans},
                opacity=0.7,
                labels=chart_labels,
            )
            st.plotly_chart(fig_scatter, width="stretch")

            st.markdown(
                """
                A :primary[população jovem] tem um ponto de agrupamento em faixas
                de peso mais elevadas.
                Além disso, nota-se que quem anda :primary[a pé] tende a pesar menos que
                aqueles que dirigem :primary[carro ou moto].
                """
            )

    st.divider()

    # ==========================================================
    # TIME USING ELETRONICS (TUE) vs AGE
    # ==========================================================
    st.subheader("Uso de eletrônicos por idade")
    st.caption("Pessoas mais jovens ficam mais tempo em telas?")
    c5, c6 = st.columns([1, 3], gap="large", vertical_alignment="center")

    with c6:
        df_tech = pd.DataFrame(charts_data.get("tech_data", []))

        if not df_tech.empty:
            tue_map, ordered_tue = get_value_translations_and_order(data_dict, "tue")
            df_tech["tue"] = df_tech["tue"].astype(int).astype(str).map(tue_map)

            fig_tech = px.box(
                df_tech,
                x="tue",
                y="age",
                category_orders={"tue": ordered_tue},
                labels=chart_labels,
            )

            st.plotly_chart(fig_tech, width="stretch")

    with c5:
        st.markdown(
            """
            Quanto :primary[maior o tempo diário] em telas, mais jovem tende
            a ser o grupo. Pessoas mais velhas concentram-se principalmente
            na faixa de :primary[menor uso].
            """
        )

    st.divider()
    # ==========================================================
    # FAMILY HISTORY
    # ==========================================================
    c7, c8 = st.columns([2, 1], gap="large", vertical_alignment="center")

    with c7:
        st.subheader("Impacto do histórico familiar")
        st.caption("Como a genética influencia o excesso peso?")

        df_fam = pd.DataFrame(charts_data.get("family_history", []))
        if not df_fam.empty:
            df_fam["obesity"] = df_fam["obesity"].map(obesity_translation_map)

            fh_map = get_value_translations_and_order(data_dict, "family_history")[0]
            df_fam["family_history"] = df_fam["family_history"].map(fh_map)

            fig_hist = px.histogram(
                df_fam,
                x="family_history",
                y="count",
                color="obesity",
                category_orders={"obesity": ordered_obesity},
                barmode="stack",
                barnorm="percent",
                labels=chart_labels,
            )
            fig_hist.update_layout(
                yaxis=dict(
                    ticksuffix="%",
                    title="Porcentagem do grupo",
                ),
            )
            fig_hist.update_traces(
                hovertemplate=(
                    "<b>%{fullData.name} </b><br>"
                    + "Histórico: %{x} <br>"
                    + "Percentual: %{y:.1f}%"
                    + "<extra></extra>"
                )
            )
            st.plotly_chart(fig_hist, width="stretch")

            st.markdown(
                """
                Pacientes :primary[com histórico familiar] de excesso de peso tendem
                a apresentar níveis mais altos de obesidade.
                """
            )

    # ==========================================================
    # SMOKE PIE
    # ==========================================================
    with c8:
        st.subheader("Relevância do tabagismo")
        st.caption("Qual a porcentagem de pacientes fumantes?")

        df_smoke = pd.DataFrame(charts_data.get("smoke_data", []))

        if not df_smoke.empty:
            smoke_map = get_value_translations_and_order(data_dict, "smoke")[0]

            df_smoke["category"] = df_smoke["category"].map(smoke_map)

            fig_smoke = px.pie(df_smoke, values="proportion", names="category")
            fig_smoke.update_traces(
                textinfo="percent+label",
                showlegend=False,
                hovertemplate=("Fumante: %{label}<br>Percentual: %{percent}"),
            )
            st.plotly_chart(fig_smoke, width="stretch")

            st.markdown(
                """
                Apenas :primary[2%] da base de pacientes são fumantes.
                """
            )

    st.divider()

    # ==========================================================
    # LIFESTYLE (FAF + FAVC + WEIGHT)
    # ==========================================================
    st.subheader("Exercício vs. alimentação")
    st.caption(
        "Qual o limite da atividade física frente ao consumo de alimentos calóricos?"
    )

    c9, c10 = st.columns([3, 1], gap="large", vertical_alignment="center")

    with c9:
        df_life = pd.DataFrame(charts_data.get("lifestyle_data", []))

        if not df_life.empty:
            faf_map, ordered_faf = get_value_translations_and_order(data_dict, "faf")
            df_life["faf"] = df_life["faf"].astype(int).astype(str).map(faf_map)

            favc_map = get_value_translations_and_order(data_dict, "favc")[0]
            df_life["favc"] = df_life["favc"].map(favc_map)

            fig_life = px.violin(
                df_life,
                x="faf",
                y="weight",
                color="favc",
                color_discrete_sequence=[
                    CUSTOM_PALETTE["primary"],
                    CUSTOM_PALETTE["tertiary"],
                ],
                box=True,
                points=False,
                category_orders={"faf": ordered_faf},
                labels=chart_labels,
            )
            fig_life.update_layout(
                violinmode="group",
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=1.3,
                    xanchor="center",
                    x=0.5,
                ),
            )
            st.plotly_chart(fig_life, width="stretch")
    with c10:
        st.markdown(
            """
            Idependentemente na frequência de exercício semanal, aqueles que consomem
            alimentos calóricos mantêm :primary[mediana de peso superior]
            aos que não consomem.
            """
        )

    st.divider()

    # ==========================================================
    # VEGETABLE CONSUMPTION (FCVC)
    # ==========================================================
    st.subheader("Frequência de consumo de vegetais")
    st.caption("Qual a relação do peso com a dieta alimentícia?")
    c11, c12 = st.columns([1, 3], gap="large", vertical_alignment="center")

    with c12:
        df_box = pd.DataFrame(charts_data.get("boxplot_data", []))
        if not df_box.empty:
            df_box["obesity"] = df_box["obesity"].map(obesity_translation_map)

            fcvc_map, ordered_fcvc = get_value_translations_and_order(data_dict, "fcvc")
            df_box["fcvc"] = df_box["fcvc"].astype(int).astype(str).map(fcvc_map)

            fig_box = px.box(
                df_box,
                x="fcvc",
                y="weight",
                color="obesity",
                category_orders={"obesity": ordered_obesity, "fcvc": ordered_fcvc},
                labels=chart_labels,
            )
            fig_box.update_layout(template="obesity_dark")
            st.plotly_chart(fig_box, width="stretch")

    with c11:
        st.markdown(
            """
            O consumo frequente de vegetais, por si só,
            :primary[não] apresenta associação forte com menor peso corporal.
            """
        )
