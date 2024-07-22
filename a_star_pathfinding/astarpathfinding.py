from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement
import cv2
import numpy as np

from . import tool

class AStarPathfinding:
    def __init__(self, max_rows: int, max_cols: int) -> None:
        self.max_rows = max_rows
        self.max_cols = max_cols

    def get_start_area(self, matrix: list, key_cls: str) -> (tuple[int, int] | None):
        if key_cls is None:
            return None
        instance_loc = tool.find_loc(matrix=matrix, keyword=key_cls, rows=self.max_rows, cols=self.max_cols)
        return instance_loc
    
    def get_target_area(self, matrix: list, key_cls: str) -> (tuple[int, int] | None):
        if key_cls is None:
            return None
        instance_loc = tool.find_loc(matrix=matrix, keyword=key_cls, rows=self.max_rows, cols=self.max_cols)
        return instance_loc
    
    def create_path(self, matrix: list, start_area: tuple, target_area: tuple, diagonal_movement="always", display_demo=True) -> (list | None):
        if target_area is None or start_area is None:
            return
        
        grid = Grid(matrix=matrix)
        grid.cleanup()
        
        start_x, start_y = start_area
        end_x, end_y = target_area
        start_area = grid.node(start_x, start_y)
        end_area = grid.node(end_x, end_y)

        if "never" == diagonal_movement:
            finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        elif "always" == diagonal_movement:
            finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        elif "if_at_most_one_obstacle" == diagonal_movement:
            finder = AStarFinder(diagonal_movement=DiagonalMovement.if_at_most_one_obstacle)
        elif "only_when_no_obstacle" == diagonal_movement:
            finder = AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)
        else:
            print("Invalid diagonal_movement!")
            exit()
        
        path, runs = finder.find_path(start_area, end_area, grid)

        if display_demo:
            print("\n======= Demo Map =======\n")
            print(grid.grid_str(path=path, start=start_area, end=end_area))
            print("\n======= Demo Map End =======\n")

        converted_result = tool.convert_to_list_of_tuple(path)
        return converted_result
    
    def _get_selected_grid_coordinates(self, areas: dict, path_list: list) -> list:
        path_list.sort()
        get_value = []
        for row, col in path_list:
            key = f"xy_{row}{col}"
            if key in areas:
                get_value.append(areas[key])
        return get_value

    def show_path(self, frame, start_area: tuple, target_area: tuple, areas: dict, path: list, start_color=(255, 0, 0), end_color=(255, 220, 0), path_color=(0, 255, 0)):
        if target_area is None or start_area is None:
            return frame
        path_list_rowcol = [(y, x) for x, y in path] # convert (x,y) to (row,col)
        selected_areas = self._get_selected_grid_coordinates(areas=areas, path_list=path_list_rowcol)

        for value in selected_areas:
            cv2.polylines(frame, [np.array(value, np.int32)], True, path_color, thickness=3)
        cv2.polylines(frame, [np.array(areas[f"xy_{start_area[1]}{start_area[0]}"], np.int32)], True, start_color, thickness=5)
        cv2.polylines(frame, [np.array(areas[f"xy_{target_area[1]}{target_area[0]}"], np.int32)], True, end_color, thickness=5)
        return frame