import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd


def run():

    path_astar = st.session_state.path
    path_opt = st.session_state.opt

    if path_astar is None:

        st.warning("Run A* Algorithm first!")

        return

    if path_opt is None:

        st.warning("Generate Optimized Path first!")

        return

    # ENERGY CALCULATION

    def compute_energy(path):

        energy = 0

        for i in range(1, len(path)-1):

            v1 = (
                np.array(path[i])
                -
                np.array(path[i-1])
            )

            v2 = (
                np.array(path[i+1])
                -
                np.array(path[i])
            )

            if (
                np.linalg.norm(v1) == 0
                or
                np.linalg.norm(v2) == 0
            ):
                continue

            length = np.linalg.norm(v2)

            cos_theta = (

                np.dot(v1, v2)

                /

                (
                    np.linalg.norm(v1)
                    *
                    np.linalg.norm(v2)
                    + 1e-6
                )
            )

            cos_theta = np.clip(
                cos_theta,
                -1,
                1
            )

            theta = np.arccos(cos_theta)

            energy += (
                length
                +
                6 * abs(theta)
                +
                2 * (theta ** 2)
            )

        return round(energy, 2)

    # TURN COUNT

    def count_turns(path):

        turns = 0

        for i in range(1, len(path)-1):

            v1 = (
                np.array(path[i])
                -
                np.array(path[i-1])
            )

            v2 = (
                np.array(path[i+1])
                -
                np.array(path[i])
            )

            if not np.array_equal(v1, v2):

                turns += 1

        return turns

    # SMOOTHNESS

    def compute_smoothness(path):

        smoothness = 0

        for i in range(1, len(path)-1):

            v1 = (
                np.array(path[i])
                -
                np.array(path[i-1])
            )

            v2 = (
                np.array(path[i+1])
                -
                np.array(path[i])
            )

            if (
                np.linalg.norm(v1) == 0
                or
                np.linalg.norm(v2) == 0
            ):
                continue

            cos_theta = (

                np.dot(v1, v2)

                /

                (
                    np.linalg.norm(v1)
                    *
                    np.linalg.norm(v2)
                    + 1e-6
                )
            )

            cos_theta = np.clip(
                cos_theta,
                -1,
                1
            )

            theta = np.arccos(cos_theta)

            smoothness += abs(theta)

        return round(smoothness, 2)

    # METRICS

    energy = [
        compute_energy(path_astar),
        compute_energy(path_opt)
    ]

    turns = [
        count_turns(path_astar),
        count_turns(path_opt)
    ]

    smoothness = [
        compute_smoothness(path_astar),
        compute_smoothness(path_opt)
    ]

    path_length = [
        len(path_astar),
        len(path_opt)
    ]

    labels = [
        "A* Path",
        "Optimized Path"
    ]

    # BAR GRAPH

    fig = go.Figure()

    fig.add_trace(

        go.Bar(
            name="Energy",
            x=labels,
            y=energy
        )
    )

    fig.add_trace(

        go.Bar(
            name="Turns",
            x=labels,
            y=turns
        )
    )

    fig.add_trace(

        go.Bar(
            name="Smoothness",
            x=labels,
            y=smoothness
        )
    )

    fig.add_trace(

        go.Bar(
            name="Path Length",
            x=labels,
            y=path_length
        )
    )

    fig.update_layout(

        title="Performance Comparison Dashboard",

        barmode='group',

        height=600,

        xaxis_title="Algorithms",

        yaxis_title="Values",

        legend=dict(
            orientation='h',
            y=-0.2,
            x=0.5,
            xanchor='center'
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # METRICS CARDS

    st.subheader("Performance Metrics")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "A* Energy",
            energy[0]
        )

        st.metric(
            "A* Turns",
            turns[0]
        )

    with col2:

        st.metric(
            "Optimized Energy",
            energy[1]
        )

        st.metric(
            "Optimized Turns",
            turns[1]
        )

    # TABLE

    df = pd.DataFrame({

        "Algorithm": labels,

        "Energy": energy,

        "Path Length": path_length,

        "Turns": turns,

        "Smoothness": smoothness
    })

    st.subheader("Comparison Table")

    st.dataframe(
        df,
        use_container_width=True
    )

    # BEST PATH

    if energy[1] < energy[0]:

        st.success(
            "Optimized Path Uses Less Energy"
        )

    else:

        st.info(
            "A* Path Is More Efficient"
        )