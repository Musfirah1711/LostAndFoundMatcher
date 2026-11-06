import subprocess
import sys

def main():
    print("[1/2] Downloading spaCy model: en_core_web_sm ...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    print("[2/2] Downloading Sentence Transformer model: all-MiniLM-L6-v2 ...")
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("all-MiniLM-L6-v2")
    print("Setup complete. Run: streamlit run app.py")

if __name__ == "__main__":
    main()