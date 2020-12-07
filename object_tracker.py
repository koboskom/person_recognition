import os
from threading import Thread
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import subprocess as sp


def getcolor(img):
    """
    avarage color detection
    """
    avg_color = np.mean(np.mean(img, axis=0), axis=0)
    red = avg_color[0]
    green = avg_color[1]
    blue = avg_color[2]
    c = np.array([red,green,blue])
    variance = np.sqrt(np.sum(np.square(c - np.max(c)))/3)
    if variance <= 10:
        return [255, 255, 0]
    elif red > green and red > blue:
        return [255, 0, 0]
    elif green > red and green > blue:
        return [0, 255, 0]
    else:
        return [0, 0, 255]


progress = 0
in_file = None
out_file = None
thread = None

def start_tracker(input_file: str, output_file: str, output_file_log: str):
    global in_file
    global out_file
    global log_file
    global thread
    in_file = input_file
    out_file = output_file
    log_file = output_file_log
    thread = Thread(target = tracker)
    thread.start()

def get_progress():
    global progress
    return progress

def opentxt(filetxt):
    """
    log display
    """
    programName = "notepad.exe"
    fileName = filetxt
    sp.Popen([programName, fileName])

def player(video):
    """
    video display using openCV
    """
    print(video)
    cap = cv2.VideoCapture(video)
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def tracker():
    global progress
    global in_file
    global out_file
    max_cosine_distance = 0.4
    nn_budget = None

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    utils.load_config()
    input_size = 416
    saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-tiny-416', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    #load video and open log file
    vid = cv2.VideoCapture(in_file)
    f = open(log_file, "w+")

    # get data from video
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

    # specify detection
    class_name = 'person'

    # get data to count fps and progress
    frame_count = 0
    frame_num = 0
    tot_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    start_time = time.time()

    # start of video porcessing frame by frame
    while True:
        return_value, frame = vid.read()
        if not return_value:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_count +=1
        frame_num += 1
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.5
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # read in all class names from config
        class_names = utils.read_class_names('./data/classes/coco.names')
        person_class_idx = list(class_names.values()).index(class_name)

        # counting objects
        deleted_indx = [i for i in range(num_objects) if int(classes[i]) != person_class_idx]
        count = num_objects - len(deleted_indx)
        cv2.putText(frame, "Count: {}".format(count), (width-200, 30), 0, 1, (0, 255, 0), 2)

        # log file save
        f.write("Frame: "+ str(frame_num) + " Count: "+ str(count) + "\n")

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, class_name, features)]

         # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            # cropping box to identify color
            dx = int(bbox[3]-bbox[1])*0.1
            dy = int(bbox[2]-bbox[0])*0.1
            cropped = frame[(int(bbox[1] + dx)):(int(bbox[3] - dx)), (int(bbox[0] + dy)):(int(bbox[2] - dy))]
            #define color
            if cropped.size != 0:
                color = getcolor(cropped)
            else:
                color = [255, 255, 0]
            #frame people with colored frames
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        out.write(result)
        progress = frame_count/tot_frames
    f.close()
    end_time = time.time()
    progress = 1
    print("Avg FPS: {}".format( frame_count/(end_time - start_time)))
    cv2.destroyAllWindows()
    session.close()
