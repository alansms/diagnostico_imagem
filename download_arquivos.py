import os
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64
IMAGE_SIZE = (180, 180)
# Diretório base onde as imagens foram extraídas e organizadas em subpastas:
DATASET_DIR = "/Users/alansms/PycharmProjects/PythonProject2/CP-2/extracted_images"

# Supondo que a estrutura seja:
# extracted_images/
#    train/
#       NORMAL/
#       PNEUMONIA/
#    val/
#       NORMAL/
#       PNEUMONIA/
#    test/
#       NORMAL/
#       PNEUMONIA/
train_dir = os.path.join(DATASET_DIR, "train")
val_dir   = os.path.join(DATASET_DIR, "val")
test_dir  = os.path.join(DATASET_DIR, "test")

# Carrega o dataset de treino
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels="inferred",       # Rótulos são inferidos a partir dos nomes das subpastas
    label_mode="binary",     # Rótulos como 0 ou 1
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123
)

# Carrega o dataset de validação
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123
)

# Carrega o dataset de teste
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Melhorar o desempenho com cache e prefetch:
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("Dados de treino carregados:", train_ds)