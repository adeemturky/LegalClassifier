from transformers import pipeline
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime

def main():
    data_folder = "../data/txt_files"
    output_file = "../outputs/classified_documents.csv"

    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

    labels = [
        "عقود", "أحكام قضائية", "قوانين", "فتاوى قانونية",
        "إنذارات/إشعارات", "تفويض/وكالة", "رخص تجارية",
        "مذكرات قانونية", "اتفاقيات تسوية"
    ]

    files = [f for f in os.listdir(data_folder) if f.endswith(".txt")]
    print(f"📂 Found {len(files)} files for classification.\n")

    results = []

    for file_name in tqdm(files, desc="🔎 Classifying files"):
        file_path = os.path.join(data_folder, file_name)

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()[:1000]  # First 1000 chars only

        output = classifier(
            text,
            candidate_labels=labels,
            hypothesis_template="هذا النص يتحدث عن {}."
        )

        results.append({
            "file_name": file_name,
            "text_sample": text[:200],
            "predicted_label": output['labels'][0],
            "score": output['scores'][0],
            "classification_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n✅ Classification complete! Results saved to: {output_file}\n")

if __name__ == "__main__":
    main()
