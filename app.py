import streamlit as st
import numpy as np
import heapq
import time

import astar_energy
import astar_3d_terrain
import candidate_paths
import candidate_paths_3d
import comparison_graph

from dynamic_obstacles import (
    initialize_dynamic_obstacles,
    move_dynamic_obstacles
)

st.set_page_config(
    layout="wide"
)

# SESSION STATES

if "grid" not in st.session_state:

    st.session_state.grid = None
    st.session_state.Z = None

    st.session_state.path = None
    st.session_state.opt = None

    st.session_state.start = (2, 2)
    st.session_state.goal = (18, 14)

    st.session_state.show_3d = False

if "dynamic_obstacles" not in st.session_state:

    st.session_state.dynamic_obstacles = []

if "grid_size" not in st.session_state:

    st.session_state.grid_size = 20

if "obstacle_density" not in st.session_state:

    st.session_state.obstacle_density = 0.25

if "show_candidate_2d" not in st.session_state:

    st.session_state.show_candidate_2d = False

if "show_candidate_3d" not in st.session_state:

    st.session_state.show_candidate_3d = False

# SIDEBAR

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Go To",

    [
        "Home",
        "Candidate Paths",
        "Comparison"
    ]
)

# HOME PAGE

if page == "Home":

    st.title(
        "Energy-Aware Path Planning "
        "In 2D And 3D Environment"
    )

    left, right = st.columns([1, 2])

    with left:

        st.subheader("Controls")

        st.session_state.grid_size = st.slider(
            "Map Size",
            10,
            50,
            st.session_state.grid_size
        )

        st.session_state.obstacle_density = st.slider(
            "Obstacle Density",
            0.0,
            0.5,
            st.session_state.obstacle_density
        )

        st.subheader("Start And Goal Positions")

        start_x = st.number_input(
            "Start X",
            0,
            st.session_state.grid_size - 1,
            st.session_state.start[0]
        )

        start_y = st.number_input(
            "Start Y",
            0,
            st.session_state.grid_size - 1,
            st.session_state.start[1]
        )

        goal_x = st.number_input(
            "Goal X",
            0,
            st.session_state.grid_size - 1,
            st.session_state.goal[0]
        )

        goal_y = st.number_input(
            "Goal Y",
            0,
            st.session_state.grid_size - 1,
            st.session_state.goal[1]
        )

        st.session_state.start = (
            start_x,
            start_y
        )

        st.session_state.goal = (
            goal_x,
            goal_y
        )

        enable_dynamic = st.checkbox(
            "Enable Dynamic Obstacles"
        )

        simulation_speed = st.slider(
            "Simulation Speed",
            1,
            10,
            5
        )

        st.markdown("---")

        # GENERATE MAP

        if st.button(
            "Generate Map",
            use_container_width=True
        ):

            size = st.session_state.grid_size
            density = st.session_state.obstacle_density

            np.random.seed(2)

            grid = (
                np.random.rand(size, size)
                < density
            ).astype(int)

            x = np.linspace(0, 50, size)
            y = np.linspace(0, 50, size)

            X, Y = np.meshgrid(x, y)

            Z = (
                5
                + 0.6 * np.sin(X / 6)
                + 0.6 * np.cos(Y / 7)
            )

            start = st.session_state.start
            goal = st.session_state.goal

            grid[start[1], start[0]] = 0
            grid[goal[1], goal[0]] = 0

            if enable_dynamic:

                st.session_state.dynamic_obstacles = (
                    initialize_dynamic_obstacles(size)
                )

                for ox, oy in st.session_state.dynamic_obstacles:

                    if (
                        (ox, oy) != start
                        and
                        (ox, oy) != goal
                    ):

                        grid[oy, ox] = 1

            st.session_state.grid = grid
            st.session_state.Z = Z

            st.session_state.path = None
            st.session_state.opt = None

            st.session_state.show_3d = False

            st.success("Map Generated")

        # RUN A*

        if st.button(
            "Run A* Algorithm",
            use_container_width=True
        ):

            grid = st.session_state.grid
            start = st.session_state.start
            goal = st.session_state.goal

            if grid is None:

                st.warning(
                    "Generate map first!"
                )

            else:

                def heuristic(a, b):

                    return (
                        (a[0] - b[0])**2
                        +
                        (a[1] - b[1])**2
                    )**0.5

                open_set = [(0, start)]

                came_from = {}

                g = {start: 0}

                found = False

                while open_set:

                    _, current = heapq.heappop(
                        open_set
                    )

                    if current == goal:

                        path = [current]

                        while current in came_from:

                            current = came_from[current]
                            path.append(current)

                        st.session_state.path = (
                            path[::-1]
                        )

                        found = True

                        break

                    for dx, dy in [

                        (1, 0),
                        (-1, 0),
                        (0, 1),
                        (0, -1)

                    ]:

                        nx = current[0] + dx
                        ny = current[1] + dy

                        if not (

                            0 <= nx < grid.shape[1]
                            and
                            0 <= ny < grid.shape[0]

                        ):
                            continue

                        if grid[ny, nx] == 1:
                            continue

                        temp = g[current] + 1

                        if (

                            (nx, ny) not in g
                            or
                            temp < g[(nx, ny)]

                        ):

                            came_from[
                                (nx, ny)
                            ] = current

                            g[(nx, ny)] = temp

                            heapq.heappush(

                                open_set,

                                (
                                    temp +
                                    heuristic(
                                        (nx, ny),
                                        goal
                                    ),

                                    (nx, ny)
                                )
                            )

                if found:

                    st.success(
                        "A* Path Generated"
                    )

                else:

                    st.error(
                        "No Path Found!"
                    )

        # OPTIMIZED PATH

        if st.button(
            "Optimized Path",
            use_container_width=True
        ):

            if st.session_state.path is None:

                st.warning(
                    "Run A* first!"
                )

            else:

                path = st.session_state.path
                grid = st.session_state.grid

                smooth = path.copy()

                for i in range(
                    1,
                    len(path)-1
                ):

                    avg = (

                        np.array(path[i-1])
                        +
                        np.array(path[i+1])

                    ) / 2

                    avg = tuple(
                        map(
                            int,
                            np.round(avg)
                        )
                    )

                    if (

                        0 <= avg[0] < grid.shape[1]
                        and
                        0 <= avg[1] < grid.shape[0]

                    ):

                        if grid[
                            avg[1],
                            avg[0]
                        ] == 0:

                            smooth[i] = avg

                st.session_state.opt = smooth

                st.success(
                    "Optimized Path Generated"
                )

        # SHOW 3D

        if st.button(
            "Show 3D Terrain",
            use_container_width=True
        ):

            if st.session_state.path is None:

                st.warning(
                    "Run A* first!"
                )

            else:

                st.session_state.show_3d = True

    # RIGHT SIDE

    with right:

        if enable_dynamic:

            st.session_state.dynamic_obstacles = (
                move_dynamic_obstacles(
                    st.session_state.dynamic_obstacles,
                    st.session_state.grid_size
                )
            )

        tab1, tab2 = st.tabs([
            "2D Visualization",
            "3D Terrain"
        ])

        with tab1:

            astar_energy.run()

        with tab2:

            if not st.session_state.show_3d:

                st.info(
                    "Click Show 3D Terrain"
                )

            else:

                astar_3d_terrain.run()

        time.sleep(
            1 / simulation_speed
        )

# CANDIDATE PAGE

elif page == "Candidate Paths":

    st.title(
        "Candidate Path Generation"
    )

    left, right = st.columns([1, 2])

    with left:

        st.subheader("Controls")

        st.info(
            f"""

            Start Position:
            {st.session_state.start}

            Goal Position:
            {st.session_state.goal}

            """
        )

        st.markdown("---")

        if st.button(
            "Generate Candidate Paths (2D)",
            use_container_width=True
        ):

            if st.session_state.grid is None:

                st.warning(
                    "Generate map in Home page first!"
                )

            elif st.session_state.path is None:

                st.warning(
                    "Run A* Algorithm in Home page first!"
                )

            else:

                st.session_state.show_candidate_2d = True
                st.session_state.show_candidate_3d = False

        if st.button(
            "Generate Candidate Paths (3D)",
            use_container_width=True
        ):

            if st.session_state.grid is None:

                st.warning(
                    "Generate map in Home page first!"
                )

            elif st.session_state.path is None:

                st.warning(
                    "Run A* Algorithm in Home page first!"
                )

            else:

                st.session_state.show_candidate_3d = True
                st.session_state.show_candidate_2d = False

    with right:

        tab1, tab2 = st.tabs([
            "2D Candidate Paths",
            "3D Candidate Paths"
        ])

        with tab1:

            if not st.session_state.show_candidate_2d:

                st.info(
                    "Click Generate Candidate Paths (2D)"
                )

            else:

                candidate_paths.run()

        with tab2:

            if not st.session_state.show_candidate_3d:

                st.info(
                    "Click Generate Candidate Paths (3D)"
                )

            else:

                candidate_paths_3d.run()

# COMPARISON PAGE

elif page == "Comparison":

    st.title(
        "Performance Comparison"
    )

    if st.button(
        "Show Comparison Graphs",
        use_container_width=True
    ):

        comparison_graph.run()