import pandas as pd
import re
import nltk
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

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


def preprocess_text(text):
    """
    Realiza o pré-processamento do texto conforme as etapas definidas no ensaio.
    """
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


if __name__ == "__main__":
    input_path = os.getenv("INPUT_DATA_PATH", "src/main/resource/data/ptbr_train.csv")
    output_path = os.getenv("OUTPUT_PROCESSED_DATA_PATH", "src/main/resource/data/processed_told-br-dataset.csv")



    print(f"Carregando dados de: {input_path}")
    try:
        df = pd.read_csv(input_path)
        print(f"Shape original: {df.shape}")

        print("Iniciando pré-processamento dos textos...")
        df['text'] = df['text'].apply(preprocess_text)
        print("Pré-processamento concluído.")

        df = df[df['text'].str.strip() != '']
        print(f"Shape após limpeza de textos vazios: {df.shape}")

        df.to_csv(output_path, index=False)
        print(f"Dados pré-processados salvos em: {output_path}")

    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {input_path}. Certifique-se de que o dataset está no caminho correto.")
    except Exception as e:
        print(f"Ocorreu um erro durante o pré-processamento: {e}")