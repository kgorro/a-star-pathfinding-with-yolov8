class TabuSearchPathfinding(AStarPathfinding):
    def __init__(self, max_rows: int, max_cols: int, tabu_tenure: int = 5) -> None:
        super().__init__(max_rows, max_cols)
        self.tabu_tenure = tabu_tenure
        self.tabu_list = []
    
    def _generate_neighbors(self, current_path: list) -> list:
        # Generate neighbors by slightly altering the last step in the path
        neighbors = []
        last_point = current_path[-1]
        for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_x = last_point[0] + direction[0]
            new_y = last_point[1] + direction[1]
            if (0 <= new_x < self.max_rows) and (0 <= new_y < self.max_cols):
                new_path = current_path[:-1] + [(new_x, new_y)]
                if new_path not in neighbors and tuple(new_path) not in self.tabu_list:
                    neighbors.append(new_path)
        return neighbors
    
    def _evaluate_path(self, path: list) -> int:
        # Extend cost evaluation to consider obstacles or weights from the matrix
        return len(path)  # Modify based on the application

    def _update_tabu_list(self, path: list) -> None:
        # Store paths as tuples for immutability
        self.tabu_list.append(tuple(path))
        if len(self.tabu_list) > self.tabu_tenure:
            self.tabu_list.pop(0)

    def tabu_search(self, matrix: list, start_area: tuple, target_area: tuple, max_iterations: int = 100) -> list:
        initial_path = super().create_path(matrix, start_area, target_area)
        if not initial_path:
            return []
        
        best_path = initial_path
        best_cost = self._evaluate_path(best_path)

        for _ in range(max_iterations):
            neighbors = self._generate_neighbors(best_path)
            if not neighbors:
                break
            
            current_best = None
            current_best_cost = float('inf')

            for neighbor in neighbors:
                cost = self._evaluate_path(neighbor)
                if cost < current_best_cost:
                    current_best_cost = cost
                    current_best = neighbor
            
            if current_best:
                best_path = current_best
                best_cost = current_best_cost
                self._update_tabu_list(current_best)

        return best_path if best_cost < self._evaluate_path(initial_path) else initial_path

    def create_path(self, matrix: list, start_area: tuple, target_area: tuple, diagonal_movement="always", display_demo=True) -> (list | None):
        if target_area is None or start_area is None:
            return
        
        # Use Tabu Search instead of A* for pathfinding
        best_path = self.tabu_search(matrix, start_area, target_area)

        if display_demo:
            grid = Grid(matrix=matrix)
            grid.cleanup()
            print("\n======= Demo Map =======\n")
            print(grid.grid_str(path=best_path, start=start_area, end=target_area))
            print("\n======= Demo Map End =======\n")

        converted_result = tool.convert_to_list_of_tuple(best_path)
        return converted_result
