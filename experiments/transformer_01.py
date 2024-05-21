import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import cv2

from datasets import Dataset, Features, Array3D, Value
from transformers import TrOCRProcessor, AutoTokenizer, TFVisionEncoderDecoderModel, AutoModelForCausalLM, VisionEncoderDecoderConfig
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Reduction
from IPython.display import clear_output

processor_checkpoint = "microsoft/trocr-base-handwritten"
tokenizer_checkpoint = "FacebookAI/xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, force_download=True)
processor = TrOCRProcessor.from_pretrained(processor_checkpoint, force_download=True)
clear_output()

def transform_image(image):
    image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_AREA)
    return image

def process_data(batch):
    images = [np.asarray(image) for image in batch['image']]
    # pixel_values = processor(images, return_tensors="tf", do_rescale=False, do_normalize=False).pixel_values
    pixel_values = processor(images, do_rescale=False, return_tensors="tf").pixel_values
    encoded_labels = tokenizer(batch['label'], padding="max_length", truncation=True, max_length=50, return_tensors="tf")
    return {
        "pixel_values": pixel_values, 
        "decoder_input_ids": encoded_labels['input_ids'], 
        "decoder_attention_mask": encoded_labels['attention_mask'],
        "labels": encoded_labels['input_ids']}

# read image
print("-----START TO READ IMAGES-----")
image_path = "experiments/cropped_images_boc_00_cleaned"
files = os.listdir(image_path)
images = np.asarray([cv2.cvtColor(plt.imread(os.path.join(image_path, file)), cv2.COLOR_RGB2BGR) for file in files], dtype="object")
image_names = np.asarray([os.path.splitext(file)[0].split(" ")[0] for file in files], dtype="object")

# apply image augmentation
print("-----START TO AUGMENT IMAGES-----")
new_images = []
new_image_names = []
idx = 0
datagen = ImageDataGenerator(rotation_range=5, shear_range=0.5, zoom_range=0.1, fill_mode='nearest')
for image in images:
    new_images.append(image)
    new_image_names.append(image_names[idx])
    raw_img = np.expand_dims(image, axis=0)
    aug_iter = datagen.flow(raw_img, batch_size=1)
    for i in range(4):
        augmented_images = next(aug_iter)
        augmented_bgr_image = augmented_images[0].astype(np.uint8)
        new_images.append(augmented_bgr_image)
        new_image_names.append(image_names[idx])   
    idx = idx+1
new_images = np.asarray(new_images, dtype="object")
new_images = np.asarray([transform_image(image) for image in new_images], dtype="object")
new_image_names = np.asarray(new_image_names, dtype="object")

# convert data to huggingface dataset
print("-----START TO CONVERT TO HUGGINGFACE DATASET OBJECT-----")
features = Features({
    'image': Array3D(dtype="float32", shape=(224, 224, 3)),
    'label': Value(dtype="string")})
data = [{'image': image, 'label': label} for image, label in zip(new_images, new_image_names)]
dataset = Dataset.from_list(data, features=features)

# convert to tf dataset
print("-----START TO CONVERT TO TENSORFLOW DATASET OBJECT-----")
dataset = dataset.map(process_data, batched=True, remove_columns=['image', 'label'])
train_dataset = dataset.to_tf_dataset(
    columns=['pixel_values', 'decoder_input_ids', 'decoder_attention_mask'],
    label_cols='labels',
    batch_size=32,
    shuffle=True
)

# construct_model
print("-----START TO CONSTRUCT THE MODEL-----")
config_encoder = VisionEncoderDecoderConfig.from_pretrained(processor_checkpoint).encoder
config_decoder = AutoModelForCausalLM.from_pretrained(tokenizer_checkpoint, is_decoder=True).config
config_encoder.output_hidden_states = True
config_decoder.is_decoder=True
config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
model = TFVisionEncoderDecoderModel(config=config)
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# print("-----START TO TRAIN THE MODEL-----")
checkpoint_filepath = '/tmp/ckpt/checkpoint.weights.h5'
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE), options = run_opts)
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,save_weights_only=True, mode='max')
model.fit(train_dataset, epochs=3, verbose=1, callbacks=[model_checkpoint_callback])