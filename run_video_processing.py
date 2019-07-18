# import cv2
# from run_webcam_demo import model, display_instances, class_names
# import sys

# args = sys.argv
# if(len(args) < 2):
# 	print("run command: python video_demo.py 0 or video file name")
# 	sys.exit(0)
# name = args[1]
# if(len(args[1]) == 1):
# 	name = int(args[1])
	
# stream = cv2.VideoCapture(name)
	
# while True:
# 	ret , frame = stream.read()
# 	if not ret:
# 		print("unable to fetch frame")
# 		break
# 	results = model.detect([frame], verbose=1)

# 	# Visualize results
# 	r = results[0]
# 	masked_image = display_instances(frame, r['rois'], r['masks'], r['class_ids'], 
#                             class_names, r['scores'])
# 	cv2.imshow("masked_image",masked_image)
# 	if(cv2.waitKey(1) & 0xFF == ord('q')):
# 		break
# stream.release()
# cv2.destroyWindow("masked_image")

import cv2, sys
import numpy as np
from run_webcam_demo import model, display_instances, class_names

args = sys.argv
if(len(args) < 2):
	print("run command: python video_demo.py 0 or video file name")
	sys.exit(0)
name = args[1]
if(len(args[1]) == 1):
	name = int(args[1])

capture = cv2.VideoCapture(name)

# capture = cv2.VideoCapture('videofile.mp4')
size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('videofile_masked.avi', codec, 60.0, size)

while(capture.isOpened()):
    ret, frame = capture.read()
    if ret:
        # add mask to frame
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )
        output.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
output.release()
cv2.destroyAllWindows()