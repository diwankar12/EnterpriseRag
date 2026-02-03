import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from src.services.usage_metrics_service import usage_metrics_service
from src.tribe_config import TRIBE_CONFIG


def render_usage_dashboard():
    # ================= OVERVIEW =================
    st.subheader("Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Requests",
            value=f"{summary['total_requests']:,}",
            delta=None
        )

    with col2:
        total_tokens_m = summary["total_tokens"] / 1_000_000
        st.metric(
            label="Total Tokens",
            value=f"{total_tokens_m:.2f}M" if total_tokens_m >= 1 else f"{summary['total_tokens']:,}",
            delta=None
        )

    with col3:
        st.metric(
            label="Avg Tokens/Request",
            value=f"{summary['avg_tokens_per_request']:,.0f}",
            delta=None
        )

    with col4:
        st.metric(
            label="Estimated Cost",
            value=f"${summary['estimated_cost_usd']:.2f}",
            delta=None,
            help="Based on Gemini Flash pricing: $0.15/1M input tokens, $0.60/1M output tokens"
        )

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(
            label="Active Sessions",
            value=f"{summary['unique_users']:,}",
            delta=None,
            help="Unique session IDs (not authenticated users)"
        )

    with col6:
        st.metric(
            label="Success Rate",
            value=f"{summary['success_rate']:.1f}%",
            delta=None
        )

    with col7:
        st.metric(
            label="Avg Response Time",
            value=f"{summary['avg_duration_ms']:,.0f}ms",
            delta=None
        )

    with col8:
        input_pct = (
            (summary["total_input_tokens"] / summary["total_tokens"]) * 100
            if summary["total_tokens"] > 0 else 0
        )
        st.metric(
            label="Input/Output Ratio",
            value=f"{input_pct:.1f}% / {100 - input_pct:.1f}%",
            delta=None,
            help="Percentage of input vs output tokens"
        )

    # ================= DAILY TOKEN USAGE =================
    with st.container(border=True):
        st.subheader("Token Usage Over Time")

        if daily_usage:
            df_daily = pd.DataFrame(daily_usage)
        else:
            dates = [
                (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(days - 1, -1, -1)
            ]
            df_daily = pd.DataFrame({
                "date": dates,
                "requests": [0] * days,
                "input_tokens": [0] * days,
                "output_tokens": [0] * days,
                "total_tokens": [0] * days,
                "unique_users": [0] * days
            })

        df_daily["input_tokens"] = pd.to_numeric(df_daily["input_tokens"], errors="coerce").fillna(0)
        df_daily["output_tokens"] = pd.to_numeric(df_daily["output_tokens"], errors="coerce").fillna(0)

        INPUT_COLOR_HEX = "#20c997"
        INPUT_FILL_RGBA = "rgba(32, 201, 151, 0.6)"
        OUTPUT_COLOR_HEX = "#6610f2"
        OUTPUT_FILL_RGBA = "rgba(102, 16, 242, 0.6)"

        fig = go.Figure(layout={"template": "plotly_dark"})

        fig.add_trace(
            go.Scatter(
                x=df_daily["date"],
                y=df_daily["input_tokens"],
                name="Input Tokens",
                mode="lines",
                fill="tozeroy",
                fillcolor=INPUT_FILL_RGBA,
                line=dict(color=INPUT_COLOR_HEX, width=2, shape="spline"),
                stackgroup="one",
                hovertemplate="<b>Date:</b> %{x}<br>Input Tokens: %{y:,.0f}<extra></extra>"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_daily["date"],
                y=df_daily["output_tokens"],
                name="Output Tokens",
                mode="lines",
                fill="tonexty",
                fillcolor=OUTPUT_FILL_RGBA,
                line=dict(color=OUTPUT_COLOR_HEX, width=2, shape="spline"),
                stackgroup="one",
                hovertemplate="<b>Date:</b> %{x}<br>Output Tokens: %{y:,.0f}<extra></extra>"
            )
        )

        fig.update_layout(
            height=450,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.05,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                title="Date",
                showgrid=True,
                gridcolor="rgba(255,255,255,0.1)",
                linecolor="rgba(255,255,255,0.3)"
            ),
            yaxis=dict(
                title="Token Count",
                showgrid=True,
                gridcolor="rgba(255,255,255,0.1)",
                linecolor="rgba(255,255,255,0.3)",
                tickformat=",d"
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )

        st.plotly_chart(fig, use_container_width=True)

    # ================= INPUT VS OUTPUT =================
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("Input vs Output Tokens")

            fig_io = go.Figure()

            fig_io.add_trace(go.Bar(
                x=df_daily["date"],
                y=df_daily["input_tokens"],
                name="Input Tokens",
                marker_color="#a78bfa",
                hovertemplate="<b>%{x}</b><br>Input: %{y:,.0f}<extra></extra>"
            ))

            fig_io.add_trace(go.Bar(
                x=df_daily["date"],
                y=df_daily["output_tokens"],
                name="Output Tokens",
                marker_color="#20c997",
                hovertemplate="<b>%{x}</b><br>Output: %{y:,.0f}<extra></extra>"
            ))

            fig_io.update_layout(
                barmode="stack",
                height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Date",
                yaxis_title="Tokens"
            )

            st.plotly_chart(fig_io, width="stretch")

    # ================= TRIBE DISTRIBUTION =================
    with col2:
        with st.container(border=True):
            st.subheader("Usage by Tribe")

            if tribe_distribution:
                df_tribes = pd.DataFrame(tribe_distribution)
                df_tribes["display_name"] = df_tribes["tribe_name"].apply(
                    lambda x: TRIBE_CONFIG.get(x, {}).get("display_name", x)
                )

                df_tribes = df_tribes.sort_values("total_tokens", ascending=True)
                total_all_tokens = df_tribes["total_tokens"].sum()
                df_tribes["percentage"] = (
                    df_tribes["total_tokens"] / total_all_tokens * 100
                ).round(1)

                fig_tribes = px.bar(
                    df_tribes,
                    y="display_name",
                    x="total_tokens",
                    orientation="h",
                    text="percentage",
                    color="total_tokens",
                    color_continuous_scale=["#a78bfa", "#6610f2", "#20c997", "#fd7e14"],
                    labels={"total_tokens": "Total Tokens", "display_name": "Tribe"}
                )

                fig_tribes.update_traces(
                    texttemplate="%{text}%",
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Tokens: %{x:,.0f}<br>Percentage: %{text}%<extra></extra>",
                    marker=dict(line=dict(color="#343a40", width=1))
                )

                fig_tribes.update_layout(
                    height=350,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=20, b=0),
                    xaxis=dict(title="Token Usage", gridcolor="#343a40"),
                    yaxis=dict(title="", tickfont=dict(size=11)),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False
                )

                st.plotly_chart(fig_tribes, width="stretch")
            else:
                st.info("No tribe usage data available yet.")
