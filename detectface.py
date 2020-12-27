from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import datetime

def detect_and_predict_mask(frame, faceNet, maskNet):

	# lấy kích thước của khung và sau đó tạo một đốm màu từ nó
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# truyền blob qua mạng và nhận được các phát hiện khuôn mặt
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# khởi tạo danh sách khuôn mặt, vị trí tương ứng và danh sách dự đoán từ mạng mặt nạ
	faces = []
	locs = []
	preds = []

	# lặp qua các phát hiện
	for i in range(0, detections.shape[2]):
		# trích xuất độ tin cậy (nghĩa là xác suất) được liên kết với phát hiện
		confidence = detections[0, 0, i, 2]

		# lọc ra các phát hiện yếu bằng cách đảm bảo sự tự tin là lớn hơn độ tin cậy tối thiểu
		if confidence > args["confidence"]:
			# tính các tọa độ (x, y) của khung giới hạn cho đối tượng
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# đảm bảo các hộp giới hạn nằm trong kích thước của khung
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# trích xuất ROI khuôn mặt, chuyển đổi nó từ kênh BGR sang kênh RGB đặt hàng, thay đổi kích thước của nó thành 224x224 và xử lý trước
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# thêm mặt và hộp giới hạn vào danh sách tương ứng của họ
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# chỉ đưa ra dự đoán nếu phát hiện ít nhất một khuôn mặt
	if len(faces) > 0:
		# để suy luận nhanh hơn, chúng tôi sẽ đưa ra dự đoán hàng loạt về * tất cả *
		# đối mặt cùng một lúc thay vì dự đoán từng người một
		# trong vòng lặp `for` ở trên
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# trả lại 2 tuple của các vị trí khuôn mặt và tương ứng của chúng  địa điểm
	return (locs, preds)

# xây dựng trình phân tích cú pháp đối số và phân tích các đối số
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="Đường dẫn đến mô hình máy dò khuôn mặt")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="Đường đẫn đến thư mực đã được huấn luyện máy dò khẩu trang")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="Xác xuất tối thiểu để lọc các phát hiện yếu")
args = vars(ap.parse_args())

# tải mô hình phát hiện khuôn mặt nối tiếp của chúng tôi từ đĩa
print("[INFO] Tải mô hình máy dò khuôn mặt...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# tải mô hình phát hiện mặt nạ từ đĩa
print("[INFO] Tải mô hình máy dò khẩu trang...")
maskNet = load_model(args["model"])

# khởi tạo luồng video và cho phép cảm biến camera nóng lên
print("[INFO] Mở webcam...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# lặp qua các khung từ luồng video
sampleNum=0

while True:

	# lấy khung từ luồng video luồng và thay đổi kích thước để có chiều rộng tối đa 400 pixel
	frame = vs.read()
	frame = imutils.resize(frame, width=800)

	datetime_object = datetime.datetime.now()
	d = datetime_object.strftime("%m-%d-%Y, %H-%M-%S")
	# phát hiện khuôn mặt trong khung và xác định xem chúng có đang mặc mặt nạ hay không
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# lặp trên các vị trí khuôn mặt được phát hiện và tương ứng địa điểm của chúng
	for (box, pred) in zip(locs, preds):
		# giải nén hộp giới hạn và dự đoán
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# xác định nhãn lớp và màu chúng tôi sẽ sử dụng để vẽ hộp giới hạn và văn bản

		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		if (label == "No Mask"):
			sampleNum = sampleNum+1
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			cv2.imwrite("Noname/Nomask-"'.' + str(d)+ "-" + str(sampleNum) + ".jpg",gray[startY:endY, startX:endX])

		# bao gồm xác suất trong nhãn
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# hiển thị nhãn và hình chữ nhật hộp giới hạn trên đầu ra khung
		cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# hiển thị khung đầu ra
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# nếu phím esc được nhấn, thoát khỏi vòng lặp
	if key == 27:
		break

# làm một chút dọn dẹp
cv2.destroyAllWindows()
vs.stop()