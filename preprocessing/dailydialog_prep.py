import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
import os
from huggingface_hub import login

# login
login(token=os.environ["HF_TOKEN"])

# ── config ────────────────────────────────────────────────────────────
DIMENSIONS = ["I", "N", "F", "P"]
PAIRS = {"I": ("I","E"), "N": ("N","S"), "F": ("F","T"), "P": ("P","J")}
THRESHOLDS = {"I": 0.61, "N": 0.66, "F": 0.57, "P": 0.63}  # deine optimalen Thresholds eintragen
BATCH_SIZE = 32
HF_REPO = "DrinkIcedT/roberta-large_MBTI"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── load models ─────────────────────────────────────────────────────────────
print("Loading models...")
tokenizers = {}
models = {}
for dim in DIMENSIONS:
    repo = f"{HF_REPO}_{dim}"
    tokenizers[dim] = AutoTokenizer.from_pretrained("roberta-large")
    models[dim] = AutoModelForSequenceClassification.from_pretrained(repo).to(DEVICE)
    models[dim].eval()
print("All models loaded.")

# ── DEBUG TEST START ──────────────────────────────────────────────────────────
print("\n--- Running Debug Test ---")
test_texts = [
    "I love being alone and reading books in my room.", 
    "I am the life of the party and love talking to everyone!"
]

for dim in DIMENSIONS:
    print(f"Testing Dimension {dim}:")
    for txt in test_texts:
        inputs = tokenizers[dim](txt, return_tensors="pt", truncation=True).to(DEVICE)
        with torch.no_grad():
            logits = models[dim](**inputs).logits
            prob = torch.sigmoid(logits).item()
            print(f"  Text: '{txt[:30]}...' -> Prob: {prob:.4f}")
print("--- End of Debug Test ---\n")

df = pd.read_csv("/cephyr/users/timkra/Alvis/MA/data/dailydialog_prep.csv", sep = ",")  # oder .parquet, .json etc.

texts_a = df["Person A"].tolist()
texts_b = df["Person B"].tolist()

records = (
    [{"text": t, "person": "A", "idx": i} for i, t in enumerate(texts_a)] +
    [{"text": t, "person": "B", "idx": i} for i, t in enumerate(texts_b)]
)
texts = [r["text"] for r in records]

print(f"Sample Text: '{texts[0]}'")

# ── Batch Inference ───────────────────────────────────────────────────────────
def predict_batch(texts, dim):
    all_probs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Predicting {dim}"):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizers[dim](
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length" # Exakt wie im Training
        ).to(DEVICE)
        
        with torch.no_grad():
            logits = models[dim](**inputs).logits
            probs = torch.sigmoid(logits).view(-1)
            
            batch_probs = probs.cpu().tolist()
            
        all_probs.extend(batch_probs)
    return all_probs
# ── Label dimensions ───────────────────────────────────────────────────
print("Running inference...")
probs_per_dim = {}
for dim in DIMENSIONS:
    probs_per_dim[dim] = predict_batch(texts, dim)

# ── construct MBTI labels  ─────────────────────────────────────────────────
def build_label(probs_per_dim, idx):
    label = ""
    for dim, (pos, neg) in PAIRS.items():
        label += pos if probs_per_dim[dim][idx] > THRESHOLDS[dim] else neg
    return label

labels = [build_label(probs_per_dim, i) for i in range(len(texts))]


# ── build dataset and push ─────────────────────────────────────────
result_df = pd.DataFrame({
    "idx": [r["idx"] for r in records],
    "person": [r["person"] for r in records],
    "post": texts,
    "label": labels,
    **{f"prob_{dim}": probs_per_dim[dim] for dim in DIMENSIONS}
})

print(result_df["label"].value_counts())
print(f"Check Probs I: {probs_per_dim['I'][:5]}")

hf_dataset = Dataset.from_pandas(result_df)
hf_dataset.push_to_hub("DrinkIcedT/dailydialog_mbti_labeled")
print("Done!")
