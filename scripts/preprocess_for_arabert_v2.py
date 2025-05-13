import pandas as pd
import os
from sklearn.model_selection import train_test_split
from arabert.preprocess import ArabertPreprocessor

# التصنيفات المحدّثة (5 فقط)
label2id = {
    "نظام أو لائحة": 0,
    "وثيقة تنظيمية": 1,
    "قرار إداري": 2,
    "مستند تعاقدي": 3,
    "وثيقة قضائية": 4
}

def clean_text(text, preprocessor):
    return preprocessor.preprocess(text)

def main():
    data_file = "../results/classified_legal_docs.csv"
    txt_folder = "../data/txt_files"
    output_folder = "../outputs/processed_dataset"
    os.makedirs(output_folder, exist_ok=True)

    model_name = "aubmindlab/bert-base-arabertv02"
    arabert_prep = ArabertPreprocessor(model_name=model_name)

    df = pd.read_csv(data_file)

    # قراءة النصوص من الملفات
    df["text"] = df["اسم_الملف"].apply(lambda x: open(os.path.join(txt_folder, x), encoding='utf-8').read())
    df["cleaned_text"] = df["text"].apply(lambda x: clean_text(x, arabert_prep))

    # تحويل التصنيفات إلى أرقام
    df["label_id"] = df["التصنيف"].map(label2id)

    # حذف الصفوف اللي ما انوجد لها تصنيف
    df = df.dropna(subset=["label_id"])

    # طباعة توزيع البيانات لكل تصنيف
    print("توزيع الفئات:\n", df["label_id"].value_counts())

    # حفظ النسخة الكاملة
    df[["cleaned_text", "label_id"]].to_csv(f"{output_folder}/full_dataset.csv", index=False, encoding='utf-8-sig')

    # ✅ تقسيم البيانات مع stratify
   # تقسيم أول فيه stratify
    train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label_id"])

    # تقسيم ثاني بدون stratify (لأن temp فيه فئة واحدة فقط أحيانًا)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)


    train[["cleaned_text", "label_id"]].to_csv(f"{output_folder}/train.csv", index=False, encoding='utf-8-sig')
    val[["cleaned_text", "label_id"]].to_csv(f"{output_folder}/val.csv", index=False, encoding='utf-8-sig')
    test[["cleaned_text", "label_id"]].to_csv(f"{output_folder}/test.csv", index=False, encoding='utf-8-sig')

    print("✅ تم تنظيف وتقسيم البيانات باستخدام stratify بنجاح!")

if __name__ == "__main__":
    main()
