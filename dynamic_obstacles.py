import random


def initialize_dynamic_obstacles(grid_size):

    obstacles = []

    for _ in range(5):

        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 1)

        obstacles.append((x, y))

    return obstacles


def move_dynamic_obstacles(obstacles, grid_size):

    moved = []

    for x, y in obstacles:

        dx, dy = random.choice([
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1)
        ])

        nx = max(0, min(grid_size - 1, x + dx))
        ny = max(0, min(grid_size - 1, y + dy))

        moved.append((nx, ny))

    return moved