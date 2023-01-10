import sys
import tensorflow as tf
import csv
import matplotlib.pyplot as plt

# подключение CUDA
import os
def selectGpuById(id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(id)

# проверка используемых девайсов
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# config1 = tf.ConfigProto()
# config1.gpu_options.per_process_gpu_memory_fraction = 0.4 # 40% памяти графического процессора
# session = tf.Session(config1=config1)

# Использовать набор данных рукописного ввода MNIST
mnist = tf.keras.datasets.mnist
# train_dataset = tf.keras.preprocessing.image_dataset_from_directory('set/Training', subset="training", seed=42, validation_split=0.15, batch_size=batch_size, image_size=(28, 28))


# Подготовить данные для обучения
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# Create a convolutional neural network
model = tf.keras.models.Sequential([

    # Сверточный слой. Изучаем 32 фильтра, используя ядро 3x3
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),

    # Слой с максимальным пулом, используя размер пула 2x2
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Сглаживаем единицы
    tf.keras.layers.Flatten(),

    # Добавляем скрытый слой с отсевом
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    # Добавляем выходной слой с единицами вывода для всех 10 цифр

    tf.keras.layers.Dense(10, activation="softmax")
])

# Обучаем нейронную сеть
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
fit = model.fit(x_train, y_train, epochs=15)

# Оценить производительность нейронной сети
results = model.evaluate(x_test, y_test, verbose=2)

# # Запись метрик в файл
# ls_acc = fit.history['accuracy']
# ls_los = fit.history['loss']
# with open('config3', 'w') as fileptr:
#     writer = csv.writer(fileptr, delimiter=",", lineterminator="\r")
#     writer.writerow(['epoch', 'loss', 'accuracy'])
#     for i in range(15):
#         writer.writerow([i+1, ls_los[i], ls_acc[i]])

# Save model to file
if len(sys.argv) == 2:
    filename = sys.argv[1]
    model.save(filename)
    print(f"Model saved to {filename}")

print(fit.history["accuracy"])
print(fit.history["loss"])
#
# # Строим график потерь и точночти в зависимости от эпохи
# plt.plot(fit.history["accuracy"], label="Accuracy", color="yellow")
# plt.plot(fit.history["loss"], label="Loss", color="orange")
# plt.xlabel("Эпоха")
# plt.ylabel("Потери, Точность")
# plt.legend()
# plt.savefig("conf3.png")