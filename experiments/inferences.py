from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import matplotlib.pyplot as plt
import cv2

image = cv2.cvtColor(plt.imread("Screenshot 2024-05-24 212400.png"), cv2.COLOR_RGB2BGR)
# image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_AREA)

plt.imshow(image)
plt.show()

processor = TrOCRProcessor.from_pretrained("KhmerOCRModel_TokenizerCasualLM_v2")
model = VisionEncoderDecoderModel.from_pretrained("KhmerOCRModel_TokenizerCasualLM_v2", from_tf=True)

pixel_values = processor(image, do_rescale=False, return_tensors="pt").pixel_values
generated_id = model.generate(pixel_values=pixel_values, max_length=20)
text = processor.batch_decode(generated_id, skip_special_tokens=True)
print(text)
