from transformers import VisionEncoderDecoderModel, ViTImageProcessor, XLMRobertaTokenizer
import matplotlib.pyplot as plt
import cv2

image = plt.imread("cleaned_outputs_(new)\img_0.png")

plt.imshow(image)
plt.show()

processor = ViTImageProcessor.from_pretrained("microsoft/trocr-base-handwritten")
tokenizer = XLMRobertaTokenizer.from_pretrained("channudam/khmer-xlm-roberta-base")
model = VisionEncoderDecoderModel.from_pretrained("KhmerOCRModelLMAPNumber/checkpoint-36864")

pixel_values = processor(image, do_rescale=False, return_tensors="pt").pixel_values
generated_id = model.generate(pixel_values=pixel_values, max_length=20)[0]
text = tokenizer.decode(generated_id, skip_special_tokens=True)
print(text)
