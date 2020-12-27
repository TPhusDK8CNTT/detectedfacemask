from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# xây dựng trình phân tích cú pháp đối số và phân tích các đối số
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="dataset", help="đường dẫn đến kho dữ liệu đầu vào")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="Đường đẫn biểu đồ đầu ra")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="đường dẫn đầu ra của mô hình máy dò mặt nạ")
args = vars(ap.parse_args())

# khởi tạo tốc độ học tập ban đầu, số kỷ nguyên cần đào tạo cho,
# và cỡ lô
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# lấy danh sách hình ảnh trong thư mục tập dữ liệu , sau đó khởi tạo
# danh sách dữ liệu (tức là hình ảnh) và hình ảnh lớp
print("[INFO] tải ảnh...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# lặp trên các đường dẫn hình ảnh
for imagePath in imagePaths:
	# trích xuất nhãn lớp từ tên tệp
	label = imagePath.split(os.path.sep)[-2]

	# tải hình ảnh đầu vào (224x224) và tiền xử lý nó
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# cập nhật danh sách dữ liệu và nhãn tương ứng
	data.append(image)
	labels.append(label)

# chuyển đổi dữ liệu và nhãn thành mảng NumPy
data = np.array(data, dtype="float32")
labels = np.array(labels)

# thực hiện mã hóa một lần nóng trên nhãn
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# phân vùng dữ liệu thành các phần tách đào tạo và kiểm tra bằng 75%
# dữ liệu cho đào tạo và 25% còn lại để thử nghiệm
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# xây dựng trình tạo hình ảnh đào tạo để tăng dữ liệu
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# tải mạng MobileNetV2, đảm bảo các bộ lớp FC đầu
# rời khỏi
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# xây dựng phần đầu của mô hình sẽ được đặt trên đỉnh của
# mô hình cơ sở
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# đặt mô hình FC đầu lên trên mô hình cơ sở (điều này sẽ trở thành
# mô hình thực tế chúng tôi sẽ đào tạo)
model = Model(inputs=baseModel.input, outputs=headModel)

# lặp trên tất cả các lớp trong mô hình cơ sở và đóng băng chúng để chúng sẽ
# * không * được cập nhật trong quá trình đào tạo đầu tiên
for layer in baseModel.layers:
	layer.trainable = False

# biên dịch mô hình
print("[INFO] Biên dịch mô hình...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# đào tạo người đứng đầu mạng
print("[INFO] Trưởng đào tạo...")
H = model.fit(aug.flow(trainX, trainY, batch_size=BS), steps_per_epoch=len(trainX) // BS, validation_data=(testX, testY), validation_steps=len(testX) // BS, epochs=EPOCHS)

# đưa ra dự đoán về bộ thử nghiệm
print("[INFO] đánh giá mạng...")
predIdxs = model.predict(testX, batch_size=BS)

# cho mỗi hình ảnh trong bộ thử nghiệm, chúng tôi cần tìm chỉ mục của
# nhãn với xác suất dự đoán lớn nhất tương ứng
predIdxs = np.argmax(predIdxs, axis=1)

# hiển thị báo cáo phân loại được định dạng độc đáo
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# tuần tự hóa mô hình vào đĩa
print("[INFO] lưu mô hình máy dò khẩu trang...")
model.save(args["model"], save_format="h5")

# vẽ sự mất mát và độ chính xác đào tạo
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="Lỗi")
plt.plot(np.arange(0, N), H.history["val_loss"], label="Gtrị_Lỗi")
plt.plot(np.arange(0, N), H.history["accuracy"], label="ChínhXác")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="Gtrị_ChínhXác")
plt.title("Biểu đồ thể hiện độ chính xác và độ lỗi")
plt.xlabel("Bước lặp")
plt.ylabel("Lỗi/Chính Xác")
plt.legend(loc="lower left")
plt.savefig(args["plot"])