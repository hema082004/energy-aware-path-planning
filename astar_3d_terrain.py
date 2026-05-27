import numpy as np
import plotly.graph_objects as go
import streamlit as st


def run():

    grid = st.session_state.grid
    path_astar = st.session_state.path
    path_opt = st.session_state.opt

    start = st.session_state.start
    goal = st.session_state.goal

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

    def map_path(path):

        px = []
        py = []
        pz = []

        for (x, y) in path:

            x = int(np.clip(x, 0, grid_size - 1))
            y = int(np.clip(y, 0, grid_size - 1))

            px.append(X[y, x])
            py.append(Y[y, x])
            pz.append(Z[y, x])

        return px, py, pz

    px, py, pz = map_path(path_astar)

    traces = []

    # TERRAIN

    surface = go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Viridis',
        opacity=0.9,
        name='Terrain'
    )

    traces.append(surface)

    # A* PATH

    astar_line = go.Scatter3d(
        x=px,
        y=py,
        z=pz,

        mode='lines',

        line=dict(
            color='red',
            width=6
        ),

        name='A* Path'
    )

    traces.append(astar_line)

    # OPTIMIZED PATH

    if path_opt is not None:

        bx, by, bz = map_path(path_opt)

        opt_line = go.Scatter3d(
            x=bx,
            y=by,
            z=bz,

            mode='lines',

            line=dict(
                color='green',
                width=6
            ),

            name='Optimized Path'
        )

        traces.append(opt_line)

    # START

    start_pt = go.Scatter3d(
        x=[X[start[1], start[0]]],
        y=[Y[start[1], start[0]]],
        z=[Z[start[1], start[0]]],

        mode='markers',

        marker=dict(
            size=8,
            color='green'
        ),

        name='Start'
    )

    traces.append(start_pt)

    # GOAL

    goal_pt = go.Scatter3d(
        x=[X[goal[1], goal[0]]],
        y=[Y[goal[1], goal[0]]],
        z=[Z[goal[1], goal[0]]],

        mode='markers',

        marker=dict(
            size=8,
            color='blue'
        ),

        name='Goal'
    )

    traces.append(goal_pt)

    # OBSTACLES 3D

    obstacle_x = []
    obstacle_y = []
    obstacle_z = []

    for y in range(grid_size):

        for x in range(grid_size):

            if grid[y, x] == 1:

                obstacle_x.append(X[y, x])
                obstacle_y.append(Y[y, x])
                obstacle_z.append(Z[y, x])

    obstacle_trace = go.Scatter3d(

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

    traces.append(obstacle_trace)

    # 3D FIGURE

    fig = go.Figure(data=traces)

    fig.update_layout(

        title='3D Environment Using 2D Path',

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
        ),

        height=700
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # TOP VIEW IMPLEMENTATION

    fig2 = go.Figure()

    # TERRAIN

    fig2.add_trace(

        go.Heatmap(
            z=Z,
            colorscale='Viridis',
            colorbar=dict(
                title='Height'
            )
        )
    )

    # A* PATH

    fig2.add_trace(

        go.Scatter(
            x=[p[0] for p in path_astar],
            y=[p[1] for p in path_astar],

            mode='lines',

            line=dict(
                color='red',
                width=5
            ),

            name='A* Path'
        )
    )

    # OPTIMIZED PATH

    if path_opt is not None:

        fig2.add_trace(

            go.Scatter(
                x=[p[0] for p in path_opt],
                y=[p[1] for p in path_opt],

                mode='lines',

                line=dict(
                    color='green',
                    width=5
                ),

                name='Optimized Path'
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
                size=7,
                color='white',
                symbol='square'
            ),

            name='Obstacles'
        )
    )

    # TOP VIEW LAYOUT

    fig2.update_layout(

        title='Top View Of 3D Environment',

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
            scaleanchor='x'
        )
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )