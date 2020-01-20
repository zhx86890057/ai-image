import cv2
# net = cv2.dnn.readNetFromTorch('models/eccv16/composition_vii.t7')
net = cv2.dnn.readNetFromTorch('models/instance_norm/feathers.t7')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV);
image = cv2.imread('img/1.jpg')

w = 600
h = 800
# w = None
# h = None
inWidth = w if w is not None else image.shape[1]
inHeight = h if h is not None else image.shape[0]
blob = cv2.dnn.blobFromImage(image, 1.0, (inWidth, inHeight), (103.939, 116.779, 123.680), swapRB=False, crop=False)
net.setInput(blob)
out = net.forward()
out = out.reshape(3, out.shape[2], out.shape[3])
out[0] += 103.939
out[1] += 116.779
out[2] += 123.68
out /= 255
out = out.transpose(1, 2, 0)
cv2.imshow('Styled image', out)
cv2.waitKey(0)