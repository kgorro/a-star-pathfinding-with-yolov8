class Area:
    def __init__(self, frame_grid_max_rows: int, frame_grid_max_cols: int) -> None:
        self.frame_grid_max_rows = frame_grid_max_rows
        self.frame_grid_max_cols = frame_grid_max_cols
    
    def get_areas_grid_coordinates(self, frame_width: int, frame_height: int) -> dict:
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

        return grid_coordinates
    
    
# for key, value in getGridCoordinates().items():
#     print(f"{key}: {value}")
    