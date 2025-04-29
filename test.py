from langrs import LangRS
from langrs.common import apply_nms_areas

model = LangRS(image='data/image_100cm.tif', prompt="roof", output_path="output")

model.generate_boxes( window_size=500, overlap=200, box_threshold=0.1, text_threshold=0.1)

filtered_boxes = model.outlier_rejection()

boxes = filtered_boxes['robust_covariance']
boxes_nms = apply_nms_areas(boxes, iou_threshold=0.5, inverse_area=False)
print("Filtered boxes:", len(boxes))
print("Filtered boxes after NMS:", len(boxes_nms))  

# This will probably crash if uncommented
model.generate_masks(boxes_nms)

