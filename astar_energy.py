import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time


if "robot_step" not in st.session_state:

    st.session_state.robot_step = 0

if "simulate_robot" not in st.session_state:

    st.session_state.simulate_robot = False


def run():

    if "grid" not in st.session_state:

        st.session_state.grid = None
        st.session_state.path = None
        st.session_state.opt = None

        st.session_state.start = (2, 2)
        st.session_state.goal = (18, 14)

    grid = st.session_state.grid
    start = st.session_state.start
    goal = st.session_state.goal

    if grid is not None:

        # CONTROL BUTTONS

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "Start Robot Simulation",
                use_container_width=True
            ):

                if st.session_state.opt is not None:

                    st.session_state.simulate_robot = True
                    st.session_state.robot_step = 0

                else:

                    st.warning(
                        "Generate Optimized Path First!"
                    )

        with col2:

            if st.button(
                "Stop Simulation",
                use_container_width=True
            ):

                st.session_state.simulate_robot = False

        # MAIN MAP

        fig, ax = plt.subplots(
            figsize=(8, 8)
        )

        ax.imshow(
            1 - grid,
            cmap='gray'
        )

        # OBSTACLES

        obstacle_y, obstacle_x = np.where(
            grid == 1
        )

        ax.scatter(
            obstacle_x,
            obstacle_y,

            c='black',

            s=40,

            marker='s',

            label='Obstacles'
        )

        # A* PATH

        if st.session_state.path is not None:

            px, py = zip(
                *st.session_state.path
            )

            ax.plot(

                px,
                py,

                color='blue',

                linewidth=2,

                label='A* Path'
            )

        # OPTIMIZED PATH

        if st.session_state.opt is not None:

            bx, by = zip(
                *st.session_state.opt
            )

            ax.plot(

                bx,
                by,

                color='green',

                linestyle='--',

                linewidth=3,

                label='Optimized Path'
            )

        # START

        ax.scatter(

            start[0],
            start[1],

            c='green',

            s=180,

            marker='o',

            edgecolors='black',

            linewidths=2,

            label='Start'
        )

        # GOAL

        ax.scatter(

            goal[0],
            goal[1],

            c='red',

            s=180,

            marker='o',

            edgecolors='black',

            linewidths=2,

            label='Goal'
        )

        # ROBOT SIMULATION

        if (

            st.session_state.simulate_robot
            and
            st.session_state.opt is not None

        ):

            path = st.session_state.opt

            step = st.session_state.robot_step

            if step < len(path):

                # TRAVERSED PATH

                traveled = path[:step+1]

                tx = [p[0] for p in traveled]
                ty = [p[1] for p in traveled]

                ax.plot(

                    tx,
                    ty,

                    color='red',

                    linewidth=4,

                    label='Traversed Path'
                )

                # ROBOT POSITION

                robot_x, robot_y = path[step]

                ax.scatter(

                    robot_x,
                    robot_y,

                    c='yellow',

                    s=350,

                    marker='o',

                    edgecolors='black',

                    linewidths=3,

                    zorder=20,

                    label='Robot'
                )

                st.session_state.robot_step += 1

                handles, labels = ax.get_legend_handles_labels()

                unique = dict(
                    zip(labels, handles)
                )

                ax.legend(

                    unique.values(),
                    unique.keys(),

                    loc='upper right',

                    fontsize=10
                )

                ax.set_title(
                    "Grid Map Environment"
                )

                ax.invert_yaxis()

                ax.set_aspect('equal')

                st.pyplot(
                    fig,
                    clear_figure=True
                )

                time.sleep(0.2)

                st.rerun()

            else:

                st.success(
                    "Robot Reached Goal!"
                )

                st.session_state.simulate_robot = False

        else:

            handles, labels = ax.get_legend_handles_labels()

            unique = dict(
                zip(labels, handles)
            )

            ax.legend(

                unique.values(),
                unique.keys(),

                loc='upper right',

                fontsize=10
            )

            ax.set_title(
                "Grid Map Environment"
            )

            ax.invert_yaxis()

            ax.set_aspect('equal')

            st.pyplot(fig)