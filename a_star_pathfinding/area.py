class Area:
    def __init__(self, frame_grid_max_rows: int, frame_grid_max_cols: int) -> None:
        self.frame_grid_max_rows = frame_grid_max_rows
        self.frame_grid_max_cols = frame_grid_max_cols
    
    def get_areas_grid_coordinates(self, frame_width: int, frame_height: int) -> list[dict]:
        x = frame_width // self.frame_grid_max_cols
        y = frame_height // self.frame_grid_max_rows
        grid_coordinates = {}

        for row in range(self.frame_grid_max_rows):
            for col in range(self.frame_grid_max_cols):
                key = f"xy_{row}{col}"
                top_left = (x * col, y * row)
                bottom_left = (x * col, y * (row + 1))
                bottom_right = (x * (col + 1), y * (row + 1))
                top_right = (x * (col + 1), y * row)
                grid_coordinates[key] = [top_left, bottom_left, bottom_right, top_right]

        center_coordinates = self._get_the_center_of_squares(grid_coordinates=grid_coordinates)
        return [center_coordinates, grid_coordinates]
    
    def _get_the_center_of_squares(self, grid_coordinates: dict) -> dict:
        center_coordinates = {}
        for key, corners in grid_coordinates.items():
            top_left, bottom_left, bottom_right, top_right = corners
            center_x = (top_left[0] + bottom_left[0] + bottom_right[0] + top_right[0]) // 4
            center_y = (top_left[1] + bottom_left[1] + bottom_right[1] + top_right[1]) // 4
            center_coordinates[key] = (center_x, center_y)
        
        return center_coordinates
    
# for key, value in getGridCoordinates().items():
#     print(f"{key}: {value}")
    