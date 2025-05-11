from transformers import pipeline
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime

def chunk_text(text, chunk_size=512):
    """تقسيم النص إلى أجزاء صغيرة (512 كلمة مثلا)"""
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def main():
    data_folder = "../data/txt_files"
    output_file = "../outputs/classified_documents_advanced.csv"

    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

    labels = [
        "عقود", "أحكام قضائية", "قوانين", "فتاوى قانونية",
        "إنذارات/إشعارات", "تفويض/وكالة", "رخص تجارية",
        "مذكرات قانونية", "اتفاقيات تسوية"
    ]

    files = [f for f in os.listdir(data_folder) if f.endswith(".txt")]
    print(f"📂 Found {len(files)} files for advanced classification.\n")

    results = []

    for file_name in tqdm(files, desc="🔎 Advanced Classifying files"):
        file_path = os.path.join(data_folder, file_name)

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        chunks = chunk_text(text, chunk_size=512)
        predictions = []

        for chunk in chunks:
            output = classifier(
                chunk,
                candidate_labels=labels,
                hypothesis_template="هذا النص يتحدث عن {}."
            )
            predictions.append(output['labels'][0])

        # Majority voting
        final_label = max(set(predictions), key=predictions.count)
        label_score = predictions.count(final_label) / len(predictions)

        results.append({
            "file_name": file_name,
            "predicted_label": final_label,
            "label_confidence": round(label_score, 3),  # نسبة التصويت
            "num_chunks": len(chunks),
            "classification_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n✅ Advanced classification complete! Results saved to: {output_file}\n")

if __name__ == "__main__":
    main()
