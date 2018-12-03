import os
import sys
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

# Enable multithreading
cv2.setUseOptimized(True)
cv2.setNumThreads(4)


##################################################
#              VARIABLE DEFINITIONS              #
##################################################

cwd = os.path.dirname(sys.argv[0])
out_path = os.path.join(cwd, 'yolo/data/pred')
img_path = os.path.join(cwd, 'yolo/data/train')
test_file = os.path.join(cwd, 'yolo/data/test.txt')
classes_file = os.path.join(cwd, 'yolo/data/localization.names')
model_config = os.path.join(cwd, 'yolo/yolo-localization-train.cfg')
model_weights = os.path.join(cwd, 'yolo/weights/yolo-localization_20000.weights')

test_size = (416, 416)
conf_threshold = 0.6
nms_threshold = 0.4


##################################################
#              FUNCTION DEFINITIONS              #
##################################################

# Get the names of all output layers in the network with unconnected outputs
def getOutputsNames(net):
    layers_names = net.getLayerNames()
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw rectangles around predictions
def drawPrediction(frame, class_id, conf, left, top, right, bot):
    cv2.rectangle(frame, (left, top), (right, bot), (0, 0, 255), 2, cv2.LINE_AA)

    # Get the label for the class name and its confidence
    label = f'{conf:.3f}'
    if classes:
        assert(class_id < len(classes))
        label = f'{classes[class_id]}:{label}'

    # Display the label at the top of the bounding box
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
    top = max(top, label_size[1])
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    fh, fw = frame.shape[:2]

    # Scan all bounding boxes output from the network and keep only the ones with high confidence scores
    # Assign the box's class label as the class with the highest score
    boxes = []
    class_ids = []
    confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * fw)
                center_y = int(detection[1] * fh)
                width = int(detection[2] * fw)
                height = int(detection[3] * fh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non-maxima suppression to eliminate redundant overlapping boxes with lower confidences
    preds = 0
    confs = []
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for ind in indices:
        i = ind[0]
        box = boxes[i]
        left, top, width, height = box
        preds += 1
        confs.append(confidences[i])
        drawPrediction(test_img, class_ids[i], confidences[i], left, top, left + width, top + height)

    return preds, np.mean(confs)

##################################################
#                BEGIN EVALUATION                #
##################################################

net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Pick random images from test set
num_images = 4
test_images = {}
with open(test_file, 'r') as tf:
    img_paths = tf.read().splitlines()
    for _ in range(num_images):
        img = os.path.join('yolo', random.choice(img_paths))
        test_images[img.split('/')[-1]] = cv2.imread(img)

# Create list of all classes
classes = []
with open(classes_file, 'r') as cf:
    for name in cf.readlines():
        classes.append(name.strip())

# Create output directory for images
if not os.path.isdir(out_path):
    os.makedirs(out_path)

for name, test_img in test_images.items():
    # Create a 4D blob from an image
    blob = cv2.dnn.blobFromImage(test_img, 1/255, test_size, [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    preds, avg_conf = postprocess(test_img, outs)

    # Create image file with the detection boxes
    output_img = os.path.join(out_path, name)
    cv2.imwrite(output_img, test_img.astype(np.uint8))

    print(f'Image: {name}')
    print(f'Number of predictions: {preds}')
    print(f'Average prediction confidence: {avg_conf}\n')
