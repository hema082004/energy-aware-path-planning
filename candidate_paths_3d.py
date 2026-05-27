import numpy as np
import plotly.graph_objects as go
import streamlit as st


def run():

    grid = st.session_state.grid
    start = st.session_state.start
    goal = st.session_state.goal
    path_astar = st.session_state.path

    if grid is None:

        st.warning("Generate map first!")

        return

    if path_astar is None:

        st.warning("Run A* Algorithm first!")

        return

    grid_size = grid.shape[0]

    x = np.linspace(0, 50, grid_size)
    y = np.linspace(0, 50, grid_size)

    X, Y = np.meshgrid(x, y)

    Z = (
        5
        + 0.6 * np.sin(X / 6)
        + 0.6 * np.cos(Y / 7)
        + 0.4 * np.sin((X + Y) / 10)
    )

    def is_valid(p, grid):

        x, y = int(round(p[0])), int(round(p[1]))

        if (
            x < 0
            or y < 0
            or x >= grid.shape[1]
            or y >= grid.shape[0]
        ):

            return False

        return grid[y, x] == 0

    # SMOOTH CANDIDATE PATH GENERATION

    def generate_candidate_paths(path, grid):

        candidate_paths = [path]

        for _ in range(10):

            p_new = []

            for i in range(len(path)):

                x, y = path[i]

                if i == 0 or i == len(path)-1:

                    p_new.append((x, y))

                    continue

                prev_pt = np.array(path[i-1])
                next_pt = np.array(path[i+1])

                direction = next_pt - prev_pt

                perp = np.array([
                    -direction[1],
                    direction[0]
                ])

                norm = np.linalg.norm(perp)

                if norm > 0:

                    perp = perp / norm

                smooth_factor = 0.6

                shift = (
                    perp *
                    np.random.uniform(
                        -smooth_factor,
                        smooth_factor
                    )
                )

                new_point = (
                    np.array([x, y])
                    + shift
                )

                nx, ny = map(
                    int,
                    np.round(new_point)
                )

                nx = np.clip(
                    nx,
                    0,
                    grid.shape[1]-1
                )

                ny = np.clip(
                    ny,
                    0,
                    grid.shape[0]-1
                )

                if grid[ny, nx] == 0:

                    p_new.append((nx, ny))

                else:

                    p_new.append((x, y))

            smooth_path = []

            for j in range(len(p_new)-1):

                smooth_path.append(p_new[j])

                mid = (
                    (
                        p_new[j][0]
                        +
                        p_new[j+1][0]
                    ) // 2,

                    (
                        p_new[j][1]
                        +
                        p_new[j+1][1]
                    ) // 2
                )

                if (
                    0 <= mid[0] < grid.shape[1]
                    and
                    0 <= mid[1] < grid.shape[0]
                ):

                    if grid[mid[1], mid[0]] == 0:

                        smooth_path.append(mid)

            smooth_path.append(p_new[-1])

            candidate_paths.append(smooth_path)

        return candidate_paths

    candidate_paths = generate_candidate_paths(
        path_astar,
        grid
    )

    # 3D FIGURE

    fig = go.Figure()

    fig.add_trace(

        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale='Viridis',
            opacity=0.9,
            showscale=True,
            name='Terrain'
        )
    )

    colors = [
        'yellow',
        'cyan',
        'magenta',
        'orange',
        'white',
        'purple',
        'pink'
    ]

    # CANDIDATE PATHS

    for i, p in enumerate(candidate_paths[1:]):

        px = [pt[0] for pt in p]
        py = [pt[1] for pt in p]
        pz = [Z[pt[1], pt[0]] for pt in p]

        fig.add_trace(

            go.Scatter3d(
                x=px,
                y=py,
                z=pz,

                mode='lines',

                line=dict(
                    width=2,
                    color=colors[i % len(colors)]
                ),

                opacity=0.7,

                showlegend=False
            )
        )

    # MAIN PATH

    px = [pt[0] for pt in path_astar]
    py = [pt[1] for pt in path_astar]
    pz = [Z[pt[1], pt[0]] for pt in path_astar]

    fig.add_trace(

        go.Scatter3d(
            x=px,
            y=py,
            z=pz,

            mode='lines',

            line=dict(
                color='red',
                width=8
            ),

            name='Main A* Path'
        )
    )

    # START

    fig.add_trace(

        go.Scatter3d(
            x=[start[0]],
            y=[start[1]],
            z=[Z[start[1], start[0]]],

            mode='markers',

            marker=dict(
                size=8,
                color='green'
            ),

            name='Start'
        )
    )

    # GOAL

    fig.add_trace(

        go.Scatter3d(
            x=[goal[0]],
            y=[goal[1]],
            z=[Z[goal[1], goal[0]]],

            mode='markers',

            marker=dict(
                size=8,
                color='blue'
            ),

            name='Goal'
        )
    )

    # OBSTACLES

    obstacle_x = []
    obstacle_y = []
    obstacle_z = []

    for y in range(grid_size):

        for x in range(grid_size):

            if grid[y, x] == 1:

                obstacle_x.append(X[y, x])
                obstacle_y.append(Y[y, x])
                obstacle_z.append(Z[y, x])

    fig.add_trace(

        go.Scatter3d(
            x=obstacle_x,
            y=obstacle_y,
            z=obstacle_z,

            mode='markers',

            marker=dict(
                size=3,
                color='white'
            ),

            name='Obstacles'
        )
    )

    # 3D LAYOUT

    fig.update_layout(

        title='3D Candidate Paths',

        height=750,

        scene=dict(

            xaxis_title='X',

            yaxis_title='Y',

            zaxis_title='Height',

            bgcolor='black'
        ),

        paper_bgcolor='black',

        font=dict(
            color='white'
        ),

        legend=dict(
            orientation='h',
            y=-0.15,
            x=0.5,
            xanchor='center',
            bgcolor='rgba(0,0,0,0.7)'
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # TOP VIEW IMPLEMENTATION

    fig2 = go.Figure()

    fig2.add_trace(

        go.Heatmap(
            z=Z,
            colorscale='Viridis',
            colorbar=dict(
                title="Height"
            )
        )
    )

    # CANDIDATE PATHS TOP VIEW

    for i, p in enumerate(candidate_paths[1:]):

        px = [pt[0] for pt in p]
        py = [pt[1] for pt in p]

        fig2.add_trace(

            go.Scatter(
                x=px,
                y=py,

                mode='lines',

                line=dict(
                    width=2,
                    color=colors[i % len(colors)]
                ),

                opacity=0.7,

                showlegend=False
            )
        )

    # MAIN PATH TOP VIEW

    fig2.add_trace(

        go.Scatter(
            x=[pt[0] for pt in path_astar],
            y=[pt[1] for pt in path_astar],

            mode='lines',

            line=dict(
                color='red',
                width=5
            ),

            name='Main A* Path'
        )
    )

    # START POINT

    fig2.add_trace(

        go.Scatter(
            x=[start[0]],
            y=[start[1]],

            mode='markers',

            marker=dict(
                size=10,
                color='green'
            ),

            name='Start'
        )
    )

    # GOAL POINT

    fig2.add_trace(

        go.Scatter(
            x=[goal[0]],
            y=[goal[1]],

            mode='markers',

            marker=dict(
                size=10,
                color='blue'
            ),

            name='Goal'
        )
    )

    # OBSTACLES TOP VIEW

    obstacle_x = []
    obstacle_y = []

    for y in range(grid_size):

        for x in range(grid_size):

            if grid[y, x] == 1:

                obstacle_x.append(x)
                obstacle_y.append(y)

    fig2.add_trace(

        go.Scatter(
            x=obstacle_x,
            y=obstacle_y,

            mode='markers',

            marker=dict(
                size=4,
                color='white'
            ),

            name='Obstacles'
        )
    )

    # TOP VIEW LAYOUT

    fig2.update_layout(

        title='Top View Of 3D Candidate Paths',

        height=700,

        xaxis_title='X',

        yaxis_title='Y',

        paper_bgcolor='black',

        plot_bgcolor='black',

        font=dict(
            color='white'
        ),

        legend=dict(
            orientation='h',
            y=-0.15,
            x=0.5,
            xanchor='center',
            bgcolor='rgba(0,0,0,0.7)'
        ),

        yaxis=dict(
            scaleanchor="x"
        )
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )