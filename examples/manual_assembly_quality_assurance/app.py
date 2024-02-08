import optparse

import cv2
from inference import InferencePipeline

import supervision as sv

parser = optparse.OptionParser()

parser.add_option(
    "-o", "--order", action="store", dest="order", help="Order of parts to be assembled"
)
parser.add_option(
    "-k", "--roboflow_api_key", action="store", dest="api_key", help="Roboflow API Key"
)
parser.add_option("-m", "--model_id", action="store", dest="model_id", help="Model ID")

order = parser.order

tracked_ids = set({})
current_step = 0
active_parts_not_in_place = set({})
last_detection_was_wrong = False

tracker = sv.ByteTrack()
smoother = sv.DetectionsSmoother()


def on_prediction(inference_results, frame):
    global current_step
    global last_detection_was_wrong

    predictions = sv.Detections.from_inference(inference_results)
    predictions = tracker.update_with_detections(predictions)
    predictions = smoother.update_with_detections(predictions)

    message = (
        f"Wrong part! Expected {order[current_step]}"
        if len(active_parts_not_in_place) > 0
        else None
    )

    for prediction in predictions:
        if prediction[4] in tracked_ids or current_step == len(order) - 1:
            continue

        class_name = classes[int(prediction[3])]

        if class_name == order[current_step]:
            tracked_ids.add(prediction[4])

            if current_step == len(order) - 1:
                break

            current_step = current_step + 1
            active_parts_not_in_place.clear()
            break
        elif class_name == order[current_step - 1]:
            tracked_ids.add(prediction[4])
            last_detection_was_wrong = False
            active_parts_not_in_place.clear()
        else:
            active_parts_not_in_place.add(prediction[4])
            last_detection_was_wrong = True

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    text_anchor = sv.Point(x=500, y=100)
    next_step = f"Next part: {order[current_step]}" if message is None else message

    annotated_image = bounding_box_annotator.annotate(
        scene=frame.image, detections=predictions
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=predictions,
        labels=[classes[int(i[3])] for i in predictions],
    )
    annotated_image = sv.draw_text(
        scene=annotated_image,
        text=next_step,
        text_anchor=text_anchor,
        text_scale=2,
        background_color=sv.Color(r=255, g=255, b=255),
        text_color=sv.Color(r=0, g=0, b=0),
    )

    cv2.imshow("Inference", annotated_image)
    cv2.waitKey(1)


pipeline = InferencePipeline.init(
    model_id=parser.model_id,
    video_reference=0,
    on_prediction=on_prediction,
    api_key=parser.api_key,
)

pipeline.start()

pipeline.join()
