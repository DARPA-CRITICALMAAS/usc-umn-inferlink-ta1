import layoutparser as lp
import cv2
import pytesseract
from helper.legend_detection_helper import non_max_suppression
from helper.legend_detection_helper import generate_roi
from helper.legend_detection_helper import remove_unicode

def detect_legend(map_name, map_dir = "/data/weiweidu/criticalmaas_data/validation_fault_line", \
                 json_dir = '/data/weiweidu/criticalmaas_data/validation'):
    
    image = cv2.imread(f"{map_dir}/{map_name}.png")

    y1, x1, y2, x2 = generate_roi(map_name, json_dir=json_dir)

    image_copy = image.copy()[x1:x2, y1:y2]
    image = image[..., ::-1][x1:x2, y1:y2]


    model = lp.Detectron2LayoutModel('lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config',
                                    label_map={1:"TextRegion", 2:"ImageRegion", 3:"TableRegion", 4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"},
                                    device='cuda:3')


    layout = model.detect(image)

    boxes = [list(box.coordinates) for box in layout]
    box_nms = non_max_suppression(boxes)
    detected_legend = []

    for i, layout_region in enumerate(box_nms): 
        y1, x1, y2, x2 = [int(p)for p in layout_region]
        if y1 == y2 or x1 == x2:
            continue
        image_segment = image[x1:x2, y1:y2]
        cv2.imwrite(f"data/subregions/{map_name}_layout_res_{i}.jpeg", image_segment)
        text = pytesseract.image_to_string(image_segment)
        if text == '':
            continue
        refined_text = remove_unicode(text.encode('utf-8'))
        if 'Fault' in refined_text or 'fault' in refined_text or 'ault' in refined_text:
#             print('=== unicode text === ', refined_text)
            detected_legend.append(refined_text.strip())

    return detected_legend

if __name__ == '__main__':
    detected_text = detect_legend('CA_NV_DeathValley')
    for text in detected_text:
        print(text.encode('utf-8'))
        print('=====')
    
