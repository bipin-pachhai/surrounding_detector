from Detector import *
classFile="coco.names"
model_URL="http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz"
imagePath="test/1.jpg"
imagePath2="test/4.jpg"
imagePath4="test/4.jpg"
videoPath="test/5.mp4"
threshold=0.5

detector=Detector()
detector.readClasses(classFile)

detector.downloadModel(model_URL)
detector.loadModel()
detector.predictImage(imagePath4,threshold)
#detector.predictImage(imagePath2,threshold)
detector.predictVideo(videoPath,threshold)