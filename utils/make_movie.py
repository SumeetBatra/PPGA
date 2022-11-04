
import os
import cv2

image_folder = '/home/sbatra/QDPPO/logs/method1-maega-walker2d-debug/cma_maega/trial_0/heatmaps'
video_name = 'video.mp4'

images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, 100, (width, height))

for image in images:
    print(image)
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()