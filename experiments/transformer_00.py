from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForMaskedLM, create_optimizer, AdamWeightDecay
from transformers import DataCollatorForLanguageModeling
import tensorflow as tf
from transformers.keras_callbacks import PushToHubCallback
import os

eli5 = load_dataset("eli5_category", split="train[:5000]", trust_remote_code=True)
eli5 = eli5.train_test_split(test_size=0.2)
eli5 = eli5.flatten()
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")
block_size = 128

def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["answers.text"]])

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

if __name__ == '__main__': 

    print("-----START TO EXTRACT TEXT FROM ANSWERS-----")

    tokenized_eli5 = eli5.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=eli5["train"].column_names,
    )

    print("-----START TO GROUP THE TEXT-----")

    lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="tf")

    print("-----START TO LOAD A PRE-TRAINED MODEL-----")

    model = TFAutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")

    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

    print("-----START TO PREPARE TF DATASET-----")

    tf_train_set = model.prepare_tf_dataset(
        lm_dataset["train"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_test_set = model.prepare_tf_dataset(
        lm_dataset["test"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    print(model.summary())

    path_checkpoint = "model_checkpoints"
    directory_checkpoint = os.path.dirname(path_checkpoint)

    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=path_checkpoint,
        save_weights_only=True,
        verbose=1)

    print("-----START TO FINETUNING-----")

    model.compile(optimizer=optimizer)

    model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=[callback])

    


