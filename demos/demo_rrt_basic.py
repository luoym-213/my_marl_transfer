import demos
from PathPlanning.RRT.rrt import RRT  # type: ignore


def main():
    # 起点 & 终点
    start = [0, 0]
    goal = [5, 10]

    # 障碍物 [x, y, radius]
    obstacle_list = [
        (3, 3, 1),
        (3, 6, 2),
        (3, 8, 2),
        (5, 5, 2),
        (7, 5, 2),
        (9, 5, 2),
    ]

    # 采样区域
    rand_area = [-2, 15]

    rrt = RRT(
        start=start,
        goal=goal,
        obstacle_list=obstacle_list,
        rand_area=rand_area,
        expand_dis=1.0,
        path_resolution=0.5,
        max_iter=200,
    )

    path = rrt.planning(animation=True)

    if path is None:
        print("❌ Path not found")
    else:
        print("✅ Path found")
        print("Path length:", len(path))
        print("Path:", path)


if __name__ == "__main__":
    main()