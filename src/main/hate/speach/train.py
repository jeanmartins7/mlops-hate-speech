import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm.auto import tqdm
import os

input_path = os.getenv("INPUT_PROCESSED_DATA_PATH", "src/main/resource/data/processed_told-br-dataset.csv")
model_output_dir = os.getenv("MODEL_OUTPUT_DIR", "model_artifacts/bert_hate_speech_model/")

print(f"Carregando dados pré-processados de: {input_path}")
try:
    df = pd.read_csv(input_path)
    print(f"Dados carregados. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Erro: Arquivo de dados pré-processados não encontrado em {input_path}.")
    exit()

X = df['text']
y = df['toxic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                    stratify=y)

print(f"Dados divididos: Treino={len(X_train)} amostras, Teste={len(X_test)} amostras")

model_name = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Usando dispositivo: {device}")

class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


train_dataset = HateSpeechDataset(X_train, y_train, tokenizer)
test_dataset = HateSpeechDataset(X_test, y_test, tokenizer)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()
epochs = 3

print(f"Iniciando treinamento por {epochs} épocas...")
model.train()  # [cite: 420, 760]
for epoch in range(epochs):  # [cite: 421, 756]
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):  # [cite: 423, 762]
        input_ids = batch['input_ids'].to(device)  # [cite: 424, 763]
        attention_mask = batch['attention_mask'].to(device)  # [cite: 424, 763]
        labels = batch['labels'].to(device)  # [cite: 424, 764]

        optimizer.zero_grad()  # [cite: 425, 765]
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)  # [cite: 426, 766]
        loss = outputs.loss  # [cite: 426, 766]
        total_loss += loss.item()  # [cite: 427]

        loss.backward()  # [cite: 428, 767]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Proteção contra gradientes explosivos
        optimizer.step()  # [cite: 430, 768]

    avg_train_loss = total_loss / len(train_loader)
    print(f"Época {epoch + 1} Concluída. Loss de Treinamento: {avg_train_loss:.4f}")  # [cite: 433, 434]

# Avaliar o modelo
print("Iniciando avaliação do modelo no conjunto de teste...")
model.eval()  # [cite: 439, 770]
all_preds = []  # [cite: 440, 771]
all_labels = []  # [cite: 444, 772]

with torch.no_grad():  # [cite: 445, 777]
    for batch in tqdm(test_loader, desc="Avaliação"):  # [cite: 446, 773]
        input_ids = batch['input_ids'].to(device)  # [cite: 448, 774]
        attention_mask = batch['attention_mask'].to(device)  # [cite: 448, 774]
        labels = batch['labels'].to(device)  # [cite: 775]

        outputs = model(input_ids, attention_mask=attention_mask)  # [cite: 449, 778]
        logits = outputs.logits  # [cite: 450, 779]

        preds = torch.argmax(logits, dim=1).cpu().numpy()  # [cite: 451, 780]
        all_preds.extend(preds)  # [cite: 452, 780]
        all_labels.extend(labels.cpu().numpy())  # [cite: 452, 780]

accuracy = accuracy_score(all_labels, all_preds)  # [cite: 457, 781]
report = classification_report(all_labels, all_preds, target_names=['Free', 'Hate'])  # [cite: 457, 782]
conf_matrix = confusion_matrix(all_labels, all_preds)  # [cite: 357, 788]

print(f"\nAcurácia do Modelo: {accuracy:.4f}")  # [cite: 293, 298, 783]
print("\nRelatório de Classificação:\n", report)  # [cite: 457, 786]
print("\nMatriz de Confusão:\n", conf_matrix)  # [cite: 459, 788, 797, 798, 799, 800]

os.makedirs(model_output_dir, exist_ok=True)
model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
print(f"\nModelo e Tokenizer salvos em: {model_output_dir}")