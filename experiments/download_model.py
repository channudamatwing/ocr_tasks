from transformers import XLMRobertaForMaskedLM, XLMRobertaTokenizer, ViTImageProcessor, VisionEncoderDecoderModel

model_roberta = XLMRobertaForMaskedLM.from_pretrained("channudam/khmer-xlm-roberta-base")
tokenizer = XLMRobertaTokenizer.from_pretrained("channudam/khmer-xlm-roberta-base")
model_roberta.save_pretrained("local-pretrained/khmer-xlm-roberta-base")
tokenizer.save_pretrained("local-pretrained/khmer-xlm-roberta-base")

model_ved = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
processor = ViTImageProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model_ved.save_pretrained("local-pretrained/trocr-base-handwritten")
processor.save_pretrained("local-pretrained/trocr-base-handwritten")