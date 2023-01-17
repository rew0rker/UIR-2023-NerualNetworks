import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
# tfds.disable_progress_bar()

# подключение CUDA
import os

def selectGpuById(id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(id)

# проверка используемых девайсов
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

train_ds, validation_ds, test_ds = tfds.load(
    "cats_vs_dogs",
    # Зарезервируйте 10% для проверки и 10% для тестирования
    split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
    as_supervised=True,  # Включить labels (метки/заголовки)
)

print("Number of training samples: %d" % tf.data.experimental.cardinality(train_ds))
print("Number of validation samples: %d" % tf.data.experimental.cardinality(validation_ds))
print("Number of test samples: %d" % tf.data.experimental.cardinality(test_ds))


size = (150, 150) ## зададим размер всех изображений на 150x150 и применим этот размер ко всем датасетам
train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))

# объединим данные и используем кэширование и предварительную выборку для оптимизации скорости загрузки
batch_size = 32
train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)

# переходим к случайной аугументации данных
from tensorflow import keras
from tensorflow.keras import layers

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

import numpy as np
import matplotlib.pyplot as plt

for images, labels in train_ds.take(1):
    plt.figure(figsize=(10, 10))
    first_image = images[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(
            tf.expand_dims(first_image, 0), training=True
        )
        plt.imshow(augmented_image[0].numpy().astype("int32"))
        plt.title(int(labels[0]))
        plt.axis("off")

    plt.show()

# инициализируем базовую предобученную на imagenet модель нейросети Xception
base_model = keras.applications.Xception(
    weights="imagenet",  # Загружаем веса, предварительно обученные в ImageNet.
    input_shape=(150, 150, 3),
    include_top=False,
)  # Не включаем классификатор ImageNet в верхнем слое.

# Замораживаем нашу базовую модель
base_model.trainable = False

# Создаем новую модель(новый верхний выходной слой)
inputs = keras.Input(shape=(150, 150, 3))
x = data_augmentation(inputs)  # добавляем рандомную аугментацию данных

# Предварительно обученные веса Xception требуют, чтобы ввод был масштабирован
# от (0, 255) до диапазона (-1., +1.), слой масштабирования
# выходные данные: `(input * scale) + offset`
scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(x)

# Базовая модель содержит слои пакетной нормы. Мы хотим, чтобы они оставались в режиме вывода
# когда мы разморозим базовую модель для тонкой настройки, поэтому мы убедимся, что base_model здесь работает в режиме вывода.
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Регуляризация с отсевом
outputs = keras.layers.Dense(1)(x)  # создаем выходной слой(dense -просто распростарненный слой)
model = keras.Model(inputs, outputs)

model.summary()  # резюмирование модели(вывод конфигурации)

# далее приступаем к обучению верхнего слоя
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)
epochs = 10  # test various variations
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

# Разморозить base_model. Обратите внимание, что он продолжает работать в режиме вывода
# так как мы передали `training=False` при вызове. Это значит, что
# слои batchnorm не будут обновлять свою пакетную статистику.
# Это предотвратит отмену всех тренировочных слоев слоями пакетной нормы.
# мы сделали до сих пор.
base_model.trainable = True
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 10
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)


hist = model.fit.history
print(f"history: --", hist)
acc = model.fit.history["accuracy"]
print(f"acc: --", acc)
loss = model.fit.history["loss"]
print(f"loss: --", loss)