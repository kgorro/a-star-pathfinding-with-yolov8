import cv2
from a_star_pathfinding.pathfinder import Pathfinder
from a_star_pathfinding.tool import video_frame

def main() -> None:
    # Uncomment these to get your entire_zone_area
    cv2.namedWindow("Normal Frame")
    cv2.setMouseCallback("Normal Frame", video_frame)
    
    top_left, bottom_left, bottom_right, top_right = [(203, 45), (148, 465), (845, 465), (778, 45)]
    entire_zone_area = [top_left, bottom_left, bottom_right, top_right]

    pf = Pathfinder(
            cap=cv2.VideoCapture("inference/video/sample.mp4"),
            frame_name="Pathfinding with YOLOv8",
            weights_path="weights/yolov8_weights.pt",
            coco_file_path="utils/coco.names",
            max_frame_rows_and_cols=(5, 7),
            entire_area_dimension=entire_zone_area
        )

    count = 0
    while True:
        success, frame = pf.cap.read()
        if not success:
            pf.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # reset the video from the start
            print("reset video")
            continue    
        
        if count % 3 != 0:
            count = 0
            continue
        count += 1


        # Optional
        original_height, original_width = frame.shape[:2]
        frame = pf.resize_frame(frame=frame, new_height=0.5, new_width=0.5)
        new_height, new_width = frame.shape[:2]
        transform_frame = pf.convert_frame_to_BIRDS_PERSPECTIVE(frame=frame, entire_area=entire_zone_area, default_frame_size=[new_width, new_height], new_frame_size=[new_width, new_height]) 
        

        target_area = (6, 3) # for demo purpose
        areas = pf.area.get_areas_grid_coordinates(frame_width=new_width, frame_height=new_height)
        transform_frame, mapping_instance, matrix_binary_map  = pf.track_object(frame=transform_frame, areas=areas)
        start_area = pf.astarpathfinding.get_start_area(matrix=mapping_instance, key_cls=pf.class_index["person"])
        created_path = pf.astarpathfinding.create_path(matrix=matrix_binary_map, start_area=start_area, target_area=target_area)
        

        # Optional
        transform_frame = pf.show_grid_zones(frame=transform_frame, areas=areas)
        transform_frame = pf.show_obstacle_zones(frame=transform_frame, areas=areas, cls_ids=[pf.class_index["table"]])
        transform_frame = pf.astarpathfinding.show_path(frame=transform_frame, start_area=start_area, target_area=target_area, areas=areas, path=created_path)
        cv2.imshow(pf.frame_name, transform_frame)
        frame = pf.show_whole_area_zone(frame=frame)
        cv2.imshow("Normal Frame", frame)
        
        pf.resetting()
        if cv2.waitKey(1) & 0xFF == 27: # ESC
            break

    pf.CV_END()
        

if __name__ == "__main__":
    main()