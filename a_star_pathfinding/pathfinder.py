from ultralytics import YOLO
import cv2
import numpy as np
import cvzone
import pandas as pd

from . import tool
from .astarpathfinding import AStarPathfinding
from .area import Area

class Pathfinder(Area):
    def __init__(
                self,
                cap: cv2.VideoCapture,
                frame_name: str,
                weights_path: str,
                coco_file_path: str,
                max_frame_rows_and_cols: tuple,
                entire_area_dimension=None
            ) -> None:
        self.max_frame_row, self.max_frame_col = max_frame_rows_and_cols
        self.astarpathfinding = AStarPathfinding(max_rows=self.max_frame_row, max_cols=max_frame_rows_and_cols[1])
        self.area = Area(frame_grid_max_rows=self.max_frame_row, frame_grid_max_cols=self.max_frame_col)

        # cv2
        self._check_source_capture(cap)
        self.cap = cap
        self.frame_name = frame_name

        # YOLO
        self.model = self._load_yolo_model(file_path=weights_path)
        self.class_names = self._read_class_names(file_path=coco_file_path)
        self.class_index = self._extract_to_dic(file_path=coco_file_path)

        self.binary_matrix_map = tool.create_matrix_map(max_row=self.max_frame_row, max_col=self.max_frame_col, keyword=1)

        self.entire_area_dimension = entire_area_dimension
        self.total_areas = self.max_frame_row * self.max_frame_col
        self.path_lists = self._init_path_lists()
        # exit()
        self.mapping_instance = ['#'] * self.total_areas


    """
        YOLO MODEL SET UP
    """
    def _check_source_capture(self, cap: cv2.VideoCapture) -> None:
        print("Connecting to the camera... ", end="")
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        print("DONE")

    @tool.file_exist_decorator
    def _load_yolo_model(self, file_path: str) -> YOLO:        
        model = YOLO(file_path)
        return model

    @tool.file_exist_decorator
    def _read_class_names(self, file_path: str) -> list:
        with open(file_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names

    @tool.file_exist_decorator
    def _extract_to_dic(self, file_path: str) -> dict:
        class_IDs = {}
        with open(file_path, 'r') as file:
            for index, line in enumerate(file):
                class_name = line.strip()
                class_IDs[class_name] = str(index)
        return class_IDs

    def _init_path_lists(self) -> list:
        path_lists = []
        for i in range(self.max_frame_row):
            for j in range(self.max_frame_col):
                path_lists.append([])
        return path_lists

    def track_object(self,
                     frame,
                     areas: dict,
                     show_instance_details=True,
                     conf=0.30,
                     bd_box=True,
                     txt=True,
                     center_point=True,
                     bdb_color=(255, 255, 0),
                     txt_color=(10, 10, 10),
                     center_pnt_color=(100, 125, 210)
                     ) -> list:
        path_tuple_of_lists = (self.path_lists)
        binary_map_1D = tool.array_2D_to_1D(matrix=self.binary_matrix_map)

        predict_result = self.model.predict(source=[frame], conf=conf, save=False)
        converted_result = pd.DataFrame(predict_result[0].boxes.data).astype("float")
        for index_, row in converted_result.iterrows():
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            confidence = round(row[4], 5)
            detected_class_index = int(row[5])
            class_ID_name = self.class_names[detected_class_index]
           
            bdbox_center_x = int(x1 + x2) // 2
            bdbox_center_y = int(y1 + y2) // 2
            bdbox_center_pnt = (bdbox_center_x, bdbox_center_y)
            wd, ht = x2 - x1, y2 - y1

            area_index = 0
            for key, value in areas.items():
                if cv2.pointPolygonTest(np.array(value, np.int32), bdbox_center_pnt, False) >= 0:
                    list_to_append = path_tuple_of_lists[area_index]
                    list_to_append.append(bdbox_center_pnt)
                    
                    if show_instance_details:
                        if bd_box:
                            cvzone.cornerRect(
                                    frame,
                                    bbox=(x1, y1, wd, ht),
                                    l=0,
                                    t=2,
                                    rt=2,
                                    colorR=bdb_color,
                                    colorC=bdb_color
                                )
                        if txt:
                            cvzone.putTextRect(
                                    frame, text=f"{class_ID_name} {confidence}%",
                                    pos=(x1, y1),
                                    scale=1,
                                    thickness=1,
                                    colorR=txt_color
                                )
                        if center_point:
                            cv2.circle(
                                    frame,
                                    center=bdbox_center_pnt,
                                    radius=10,
                                    color=center_pnt_color,
                                    thickness=-1
                                )
                    
                    self.mapping_instance[area_index] = f'{detected_class_index}'
                    binary_map_1D[area_index] = 0 if len(list_to_append) > 0 else 1
                area_index += 1
        
        matrix_ins_loc = tool.convert_array_1D_to_2D(array_1D=self.mapping_instance, max_col=self.max_frame_col, max_row=self.max_frame_row)
        self.binary_matrix_map = tool.convert_array_1D_to_2D(array_1D=binary_map_1D, max_col=self.max_frame_col, max_row=self.max_frame_row)
        return [frame, matrix_ins_loc, self.binary_matrix_map]



    """
        CV2 SET UP
    """
    def convert_frame_to_BIRDS_PERSPECTIVE(self, frame, entire_area: list, default_frame_size: list, new_frame_size: list):
        def_width, def_height = default_frame_size
        new_width, new_height = new_frame_size
        
        pnts1 = np.float32(
                [
                    entire_area[0], # top-left point
                    entire_area[1], # bottom-left point
                    entire_area[3], # top-right point
                    entire_area[2], # bottom-right point
                ]
            )
        pnts2 = np.float32(
                [
                    [0,0],
                    [0, def_height],
                    [def_width, 0],
                    [def_width, def_height]
                ]
            )
        matrix = cv2.getPerspectiveTransform(pnts1, pnts2)

        frame = cv2.warpPerspective(frame, matrix, [new_width, new_height], cv2.INTER_LINEAR)
        return frame

    def resize_frame(self, frame, new_width: int, new_height: int):
        new_dimension = [new_width, new_height]
        resized_frame = cv2.resize(frame, new_dimension, interpolation=cv2.INTER_AREA)
        return resized_frame
    
    def resize_frame(self, frame, new_width: float, new_height: float):
        resized_frame = cv2.resize(frame, dsize=(0, 0), fx=new_width, fy=new_height, interpolation=cv2.INTER_AREA)
        return resized_frame

    def show_whole_area_zone(self, frame, color=(200, 200, 0)):
        if self.entire_area_dimension is None:
            print("The \"self.entire_area\" was not initialize")
            return frame
        
        cv2.polylines(
                frame,
                pts=[np.array(self.entire_area_dimension, np.int32)],
                isClosed=True,
                color=color,
                thickness=2
            )
        return frame

    def show_grid_zones(self, frame, areas: dict, zones_color=(0, 0, 0), thickness=2):
        for key, value in areas.items():
            cv2.polylines(
                    img=frame,
                    pts=[np.array(value, np.int32)],
                    isClosed=True,
                    color=zones_color,
                    thickness=thickness
                )
        return frame

    def show_obstacle_zones(self, frame, areas: dict, cls_ids: list[str], none_obs=None, zones_color=(0, 0, 255), thickness=2):
        cls_ids_len = len(cls_ids)
        if none_obs is None:
            for key_ind in range(cls_ids_len):
                index = 0
                for key, value in areas.items():
                    if self.mapping_instance[index] == cls_ids[key_ind]:
                        cv2.polylines(
                                img=frame,
                                pts=[np.array(value, np.int32)],
                                isClosed=True,
                                color=zones_color,
                                thickness=thickness
                            )
                    index += 1
            return frame
        
        index = 0
        for key, value in areas.items():
            if self.mapping_instance[index] == none_obs:
                cv2.polylines(
                        img=frame,
                        pts=[np.array(value, np.int32)],
                        isClosed=True,
                        color=zones_color,
                        thickness=thickness
                    )
            index += 1
        return frame

    def show_1s_and_0s_map(self, frame, center_areas: dict, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, textThickness=2, color=(255, 255, 255)):
        binary_map_1D = tool.array_2D_to_1D(matrix=self.binary_matrix_map)
        index = 0
        for key, centers_point in center_areas.items():
            cv2.putText(frame, f"{binary_map_1D[index]}", (centers_point[0], centers_point[1]), fontFace=fontFace, fontScale=fontScale, color=color, thickness=textThickness)
            index += 1

        return frame

    def CV_END(self) -> None:
        self.cap.release()
        cv2.destroyAllWindows()
        print("End of the program")

    def resetting(self, all=True, b_map=False, p_list=False) -> None:
        if all:
            self.path_lists = self._init_path_lists()
            self.binary_matrix_map = tool.create_matrix_map(max_row=self.max_frame_row, max_col=self.max_frame_col, keyword=1)
            self.mapping_instance = ['#'] * self.total_areas
            return
        
        if b_map:
            self.binary_matrix_map = tool.create_matrix_map(max_row=self.max_frame_row, max_col=self.max_frame_col, keyword=1)
            self.obstacle_indexes = self.def_map_val

        if p_list:
            self.path_lists = self._init_path_lists()