import json

import numpy as np
import tensorflow as tf
import cv2


def load_labelmap(path):
    with open(path, "r") as d:
        label_dict = json.load(d)
    return label_dict


def convert_output(inference_result:dict):
    num_detections = int(inference_result.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy() for key, value in inference_result.items()}
    output_dict['num_detections'] = num_detections

    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    output = []
    for index, detection_score in enumerate(output_dict['detection_scores']):
        if detection_score >= .5:
            output.append(
                {
                    "DETECTION_SCORE": float(detection_score),
                    "DETECTION_CLASS": category_index[str(output_dict["detection_classes"][index])],
                    "DETECTION_COORDINATES": list([float(cord) for cord in output_dict["detection_boxes"][index]])
                }
            )
    return output


category_index = load_labelmap("street_labels.json")

model = tf.saved_model.load("C:\\DEV\\repos\\label-images\\keras_dataset\\ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03\\saved_model")
model = model.signatures['serving_default']

image = cv2.imread("C:\\DEV\\repos\\label-images\\test_data\\IMG_3092.JPG")
input_tensor = tf.convert_to_tensor(image)
input_tensor = input_tensor[tf.newaxis, ...]

output_dict = model(input_tensor)
box_list = convert_output(output_dict)

height, wide, _ = image.shape
for box in box_list:
    coordinates = box["DETECTION_COORDINATES"]
    y1 = int(coordinates[0] * height)
    x1 = int(coordinates[1] * wide)
    y2 = int(coordinates[2] * height)
    x2 = int(coordinates[3] * wide)

    cv2.rectangle(image, (x2, y2), (x1, y1), (255, 0, 0), 2)
    # cv2.putText()

cv2.imshow("test", image)
cv2.waitKey()