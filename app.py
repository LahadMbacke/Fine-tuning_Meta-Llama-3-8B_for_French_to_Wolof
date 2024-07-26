from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
from datasets import load_dataset
from evaluate import load

# Charger le modèle et le tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ajouter des tokens spéciaux pour la traduction
# Ajouter des tokens spéciaux et un token de padding
special_tokens_dict = {'additional_special_tokens': ['<fr>', '<wo>']}
tokenizer.add_special_tokens(special_tokens_dict)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
model.resize_token_embeddings(len(tokenizer))



# Charger et préparer le dataset
dataset = load_dataset('csv', data_files='/kaggle/input/data-fr-wo/train.csv', split='train')
dataset = dataset.train_test_split(test_size=0.2)

# Fonction de prétraitement
def preprocess_function(examples):
    inputs = [f"<fr>{text}<wo>" for text in examples["FRENCH"]]
    targets = examples["WOLOF"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# # Charger la métrique BLEU
# bleu = load("sacrebleu")

# def compute_metrics(eval_preds):
#     preds, labels = eval_preds
#     # Décoder les prédictions et les labels
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
#     # Calculer le score BLEU
#     result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
#     return {"bleu": result["score"]}

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    save_total_limit=2,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    # compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("./Lahad_meta-llama/Meta-Llama-3-8B_fr_wo")
tokenizer.save_pretrained("./Lahad_meta-llama/Meta-Llama-3-8B_fr_wo")



######################################################tTEST""
# Fonction de traduction
def translate(text, src_lang="fr", tgt_lang="wo"):
    input_text = f"<{src_lang}>{text}<{tgt_lang}>"
    encoded = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**encoded, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
# Test de traduction
print(translate("Je suis follement amoureux d'elle", src_lang="fr", tgt_lang="wo"))