import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('tokenizers/punkt')
except Exception as e:
    print(f"Erro ao encontrar 'punkt': {e}. Tentando baixar...")
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except Exception as e:
    print(f"Erro ao encontrar 'stopwords': {e}. Tentando baixar...")
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except Exception as e:
    print(f"Erro ao encontrar 'wordnet': {e}. Tentando baixar...")
    nltk.download('wordnet')

model_load_dir = os.getenv("MODEL_LOAD_DIR", "model_artifacts/bert_hate_speech_model/")

def preprocess_text_for_prediction(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'@user', '@', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z0-9\s.,?!]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('portuguese'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    cleaned_text = ' '.join(tokens)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

print(f"Carregando modelo e tokenizer de: {model_load_dir}")
try:
    tokenizer = BertTokenizer.from_pretrained(model_load_dir)
    model = BertForSequenceClassification.from_pretrained(model_load_dir)
    model.eval()
    print("Modelo e Tokenizer carregados com sucesso.")
except Exception as e:
    print(f"Erro ao carregar modelo ou tokenizer: {e}")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Usando dispositivo para inferência: {device}")


def classify_text(text):
    """
    Classifica um texto de entrada como discurso de ódio (1) ou não (0).
    """
    processed_text = preprocess_text_for_prediction(text)

    inputs = tokenizer(processed_text, return_tensors='pt', truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_label = torch.argmax(outputs.logits, dim=1).item()

    return predicted_label


if __name__ == "__main__":
    test_texts = [
        "Eu amo a minha família.",
        "PQP, eu odeio esse macaco, acabou com a partida.",
        "Que dia lindo, vamos aproveitar o sol.",
        "Você é um idiota, não sabe o que está falando."
    ]

    for text in test_texts:
        print(f"\nTexto: '{text}'")
        label = classify_text(text)
        if label == 1:
            print("Resultado: Contém discurso de ódio.")
        else:
            print("Resultado: Não contém discurso de ódio.")