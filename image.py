
# python detect_mask_image.py --image examples/example_01.png

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
import datetime
# xây dựng trình phân tích cú pháp đối số và phân tích cú pháp đối số
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="Đường dẫn đến ảnh")
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="Đường dẫn mô hình thư mục mô hình máy dò khuôn mặt")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="Đường đẫn đến thư mực đã được huấn luyện máy dò khẩu trang")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="Xác xuất tối thiểu để lọc các phát hiện yếu")
args = vars(ap.parse_args())

# tải mô hình máy dò khuôn mặt được tuần tự hóa của chúng tôi từ đĩa
print("[INFO] Tải mô hình phát hiện khuôn mặt...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# tải mô hình phát hiện mặt nạ từ đĩa
print("[INFO] Tải mô hình phát hiện khẩu trang...")
model = load_model(args["model"])

# tải hình ảnh đầu vào từ đĩa, sao chép nó và lấy các kích thước không gian hình ảnh
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

# tạo một đốm màu từ hình ảnh
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

# thông qua qua đốm màu qua mạng và nhận được tính năng phát hiện khuôn mặt
print("[INFO] Tính toán phát hiện khuôn mặt...")
net.setInput(blob)
detections = net.forward()
datetime_object = datetime.datetime.now()
d = datetime_object.strftime("%m-%d-%Y, %H-%M-%S")
# vòng qua các phát hiện
for i in range(0, detections.shape[2]):
	# trích xuất độ tin cậy (tức là xác suất) liên quan đến phát hiện
	confidence = detections[0, 0, i, 2]

	# lọc ra các phát hiện yếu bằng cách đảm bảo độ tin cậy lớn hơn độ tin cậy tối thiểu
	if confidence > args["confidence"]:
		# tính toán tọa độ (x, y) của hộp giới hạn cho đối tượng
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# đảm bảo các hộp giới hạn nằm trong kích thước của khung
		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

		# trích xuất ROI của khuôn mặt, chuyển đổi nó từ thứ tự kênh BGR sang RGB, thay đổi kích thước thành 224x224 và xử lý trước
		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)

		# đưa khuôn mặt qua mô hình để xác định xem khuôn mặt có đeo mặt nạ hay không
		(mask, withoutMask) = model.predict(face)[0]

		# xác định nhãn lớp và màu sắc mà chúng ta sẽ sử dụng để vẽ hộp giới hạn và văn bản
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)		

		# hiển thị nhãn và hình chữ nhật hộp giới hạn trên khung đầu ra
		cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
		cv2.imwrite("ImageOut/"+ label +"-" + str(d) + ".jpg",image)

# hiển thị ảnh đầu ra
cv2.imshow("Image Out", image)

cv2.waitKey(0)