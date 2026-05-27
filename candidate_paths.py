import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time


if "candidate_robot_step" not in st.session_state:

    st.session_state.candidate_robot_step = 0

if "candidate_simulation" not in st.session_state:

    st.session_state.candidate_simulation = False


def run():

    grid = st.session_state.grid
    start = st.session_state.start
    goal = st.session_state.goal
    path = st.session_state.path

    if grid is None or path is None:

        st.warning(
            "Run A* Algorithm First!"
        )

        return

    # GENERATE CANDIDATES

    def generate_candidate_paths(path, grid):

        candidate_paths = [path]

        for _ in range(10):

            p_new = path.copy()

            for k in range(1, len(path)-1):

                dir_vec = (
                    np.array(path[k+1])
                    -
                    np.array(path[k-1])
                )

                perp = np.array([
                    -dir_vec[1],
                    dir_vec[0]
                ])

                if np.linalg.norm(perp) > 0:

                    perp = (
                        perp
                        /
                        np.linalg.norm(perp)
                    )

                candidate = (

                    np.array(p_new[k])

                    +

                    perp *
                    np.random.randn() * 2
                )

                candidate = tuple(
                    map(
                        int,
                        np.round(candidate)
                    )
                )

                x, y = candidate

                if (

                    0 <= x < grid.shape[1]
                    and
                    0 <= y < grid.shape[0]
                    and
                    grid[y, x] == 0

                ):

                    p_new[k] = candidate

            candidate_paths.append(p_new)

        return candidate_paths

    candidate_paths = generate_candidate_paths(
        path,
        grid
    )

    # CONTROL BUTTONS

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Start Candidate Animation",
            use_container_width=True
        ):

            st.session_state.candidate_simulation = True
            st.session_state.candidate_robot_step = 0

    with col2:

        if st.button(
            "Stop Candidate Animation",
            use_container_width=True
        ):

            st.session_state.candidate_simulation = False

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    ax.imshow(
        1 - grid,
        cmap='gray'
    )

    # CANDIDATE PATHS

    for p in candidate_paths[1:]:

        px, py = zip(*p)

        ax.plot(
            px,
            py,
            '--',
            linewidth=1
        )

    # MAIN PATH

    px, py = zip(*path)

    ax.plot(

        px,
        py,

        color='blue',

        linewidth=3,

        label='Main A* Path'
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

    if st.session_state.candidate_simulation:

        step = st.session_state.candidate_robot_step

        if step < len(path):

            robot_x, robot_y = path[step]

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

            st.session_state.candidate_robot_step += 1

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
                "Candidate Paths Generation"
            )

            ax.invert_yaxis()

            ax.axis('equal')

            st.pyplot(
                fig,
                clear_figure=True
            )

            time.sleep(0.15)

            st.rerun()

        else:

            st.success(
                "Robot Reached Goal!"
            )

            st.session_state.candidate_simulation = False

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
            "Candidate Paths Generation"
        )

        ax.invert_yaxis()

        ax.axis('equal')

        st.pyplot(fig)