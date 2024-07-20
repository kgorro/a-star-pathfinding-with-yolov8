import cv2
import os
import math

def find_loc(matrix: list, keyword: str, rows: int, cols: int) -> (tuple[int, int] | None):
    for row in range(rows):
        for col in range(cols):
            if matrix[row][col] == keyword:
                return (col, row)  # (x, y) format
    return None

# def find_loc(matrix: list[list[str]]) -> dict[str, list[tuple[int, int]]] | None:
#     mydic = {}
#     rows = len(matrix)
#     cols = len(matrix[0]) if rows > 0 else 0

#     for row in range(rows):
#         for col in range(cols):
#             if matrix[row][col] != '#':
#                 element = matrix[row][col]
#                 if element not in mydic:
#                     mydic[element] = []
#                 mydic[element].append((col, row))  # Store coordinates as (x, y)

#     if not mydic:
#         return None
#     return mydic

def convert_to_list_of_tuple(value) -> list:
    path_list= []
    for i in range(len(value)):
        path_list.append(tuple(value[i]))
    return path_list

def array_2D_to_1D(matrix):
    array_1D = []
    for row in matrix:
        for element in row:
            array_1D.append(element)
    return array_1D
    
def convert_array_1D_to_2D(array_1D: list, max_row: int, max_col: int) -> list[list[int]]:
    rows = max_row
    columns = max_col
    matrix = [[0] * columns for _ in range(rows)]
    
    _row = 0
    _column = 0

    for index in range(len(array_1D)):
        matrix[_row][_column] = array_1D[index]
        _column += 1
        if _column == columns:
            _column = 0
            _row += 1
    return matrix

def calculate_distance(pointA: tuple, pointB: tuple) -> float:
    x1, y1 = pointA
    x2, y2 = pointB

    squared_diff_x = (x2 - x1) ** 2
    squared_diff_y = ((-y2) - (-y1)) ** 2

    distance = math.sqrt(squared_diff_x + squared_diff_y)
    return distance

def file_exist_decorator(func):
    def wrapper(self, file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        return func(self, file_path)
    return wrapper

def create_matrix_map(max_row: int, max_col: int, keyword):
    matrix = [[keyword for _ in range(max_col)] for _ in range(max_row)]
    return matrix

def video_frame(event, x, y, flags, param) -> None:
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)