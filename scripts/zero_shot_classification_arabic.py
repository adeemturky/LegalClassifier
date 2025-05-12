from transformers import pipeline
import pandas as pd
import os
import re
from tqdm import tqdm
from datetime import datetime
import torch

class AdvancedArabicLegalClassifier:
    def __init__(self):
        # تعريف النماذج المحسنة
        self.models = {
            "mDeBERTa-Legal": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            "Legal-Arabic-BERT": "aubmindlab/bert-base-arabertv02",
            "XLM-R-Legal": "joeddav/xlm-roberta-large-xnli"
        }
        
        # التصنيفات المحسنة مع أمثلة
        self.labels = [
            "نظام أو لائحة",      # للوثائق التشريعية الرسمية
            "وثيقة تنظيمية",      # للاشتراطات والضوابط
            "قرار إداري",         # للقرارات الرسمية
            "مستند تعاقدي",       # للعقود والاتفاقيات
            "وثيقة قضائية"        # للأحكام والفتاوى
        ]
        
        # القواعد المعززة مع أوزان
        self.enhanced_rules = [
            (r'(نظام|لائحة|اللائحة)\s', ("نظام أو لائحة", 0.95)),
            (r'(اشتراطات|ضوابط|معايير)\s', ("وثيقة تنظيمية", 0.9)),
            (r'(قرار|تعميم|تعليمات)\s(وزاري|إداري)', ("قرار إداري", 0.92)),
            (r'(عقد|اتفاقية|مذكرة\sتفاهم)\s', ("مستند تعاقدي", 0.88)),
            (r'(حكم|قرار\sقضائي|فتوى)\s', ("وثيقة قضائية", 0.91)),
            (r'صحيفة\sأم\sالقرى', ("نظام أو لائحة", 0.97))
        ]
        
        # إعدادات التصنيف
        self.template = "هذه الوثيقة القانونية تصنف ضمن فئة {}."
        self.min_confidence = 0.65
        self.results = []
        self.metadata_patterns = {
            'رقم_الوثيقة': r'رقم\sالوثيقة\s*:\s*(\d+)',
            'تاريخ_الإصدار': r'تاريخ\sالاعتماد\s*:\s*([\d/]+)',
            'جهة_الإصدار': r'جهة\sالإصدار\s*\n-+\s*\n(.+)'
        }

    def extract_metadata(self, text):
        """استخراج البيانات الوصفية من الهيكل المنظم"""
        metadata = {}
        for field, pattern in self.metadata_patterns.items():
            match = re.search(pattern, text)
            if match:
                metadata[field] = match.group(1).strip()
        return metadata

    def preprocess_legal_text(self, text):
        """معالجة متقدمة للنصوص القانونية"""
        # التوحيد اللغوي
        replacements = {
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ـ': '', 'ة': 'ه',
            'ى': 'ي', 'ئ': 'ء', 'ؤ': 'ء', 'لا': 'لا'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # التنظيف المتقدم
        text = re.sub(r'[\u064b-\u065f]', '', text)  # إزالة التشكيل
        text = re.sub(r'[ـ،؛:\-\*_]', ' ', text)     # إزالة علامات الترقيم المزعجة
        text = re.sub(r'\s+', ' ', text).strip()      # إزالة المسافات الزائدة
        
        return text

    def detect_document_structure(self, text):
        """الكشف عن الهيكل التنظيمي للوثيقة"""
        structure_markers = [
            'معلومات الوثيقة', 'جهة الإصدار', 
            'فئة الوثيقة', 'نبذة الوثيقة',
            'صحيفة أم القرى'
        ]
        return sum(marker in text for marker in structure_markers) >= 3

    def apply_enhanced_rules(self, text):
        """تطبيق القواعد المعززة مع تحليل الهيكل"""
        # التحقق من الهيكل الرسمي أولاً
        if self.detect_document_structure(text):
            metadata = self.extract_metadata(text)
            if metadata.get('جهة_الإصدار'):
                if 'لائحة' in text[:100] or 'نظام' in text[:100]:
                    return "نظام أو لائحة", 0.97
                if 'قرار' in text[:100]:
                    return "قرار إداري", 0.95
        
        # تطبيق القواعد العادية
        for pattern, (label, confidence) in self.enhanced_rules:
            if re.search(pattern, text, re.IGNORECASE):
                return label, confidence
        return None, 0

    def classify_with_ai(self, text, model_name):
        """التصنيف باستخدام الذكاء الاصطناعي مع تحسينات"""
        try:
            classifier = pipeline(
                "zero-shot-classification",
                model=self.models[model_name],
                device=0 if torch.cuda.is_available() else -1,
                batch_size=4 if torch.cuda.is_available() else 1
            )
            
            # التصنيف مع تحسين السياق
            chunks = self.smart_chunking(text)
            predictions = []
            
            for chunk in chunks:
                result = classifier(
                    chunk,
                    candidate_labels=self.labels,
                    hypothesis_template=self.template,
                    multi_label=False
                )
                
                # تعزيز الثقة للقطع الحاسمة
                if any(kw in chunk for kw in ['المادة', 'ينص', 'يقرر']):
                    result['scores'][0] = min(result['scores'][0] * 1.2, 1.0)
                
                predictions.append((result['labels'][0], result['scores'][0]))
            
            # التجميع الذكي للنتائج
            if not predictions:
                return "غير محدد", 0.0
                
            # حساب المتوسط المرجح
            label_scores = {}
            for label, score in predictions:
                if label in label_scores:
                    label_scores[label].append(score)
                else:
                    label_scores[label] = [score]
            
            avg_scores = {
                label: sum(scores)/len(scores)
                for label, scores in label_scores.items()
            }
            
            best_label = max(avg_scores.items(), key=lambda x: x[1])
            return best_label[0], best_label[1]
            
        except Exception as e:
            print(f"⚠️ خطأ في التصنيف ({model_name}): {str(e)}")
            return "error", 0.0

    def smart_chunking(self, text, max_words=350):
        """تقسيم النص إلى أجزاء ذكية مع الحفاظ على المعنى القانوني"""
        # تقسيم إلى فقرات أولاً
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            words = para.split()
            para_length = len(words)
            
            if para_length > max_words:
                # إذا كانت الفقرة طويلة جداً، تقسيمها إلى جمل
                sentences = [s.strip() for s in re.split(r'[\.\؟\!]', para) if s.strip()]
                for sent in sentences:
                    sent_words = sent.split()
                    if current_length + len(sent_words) <= max_words:
                        current_chunk.append(sent)
                        current_length += len(sent_words)
                    else:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                        current_chunk = [sent]
                        current_length = len(sent_words)
            else:
                if current_length + para_length <= max_words:
                    current_chunk.append(para)
                    current_length += para_length
                else:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [para]
                    current_length = para_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def process_document(self, file_path):
        """معالجة متكاملة للوثيقة الواحدة"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # المعالجة المسبقة
        text = self.preprocess_legal_text(text)
        file_name = os.path.basename(file_path)
        
        # التصنيف التلقائي المعزز
        rule_label, rule_confidence = self.apply_enhanced_rules(text)
        if rule_label and rule_confidence >= 0.9:
            return file_name, rule_label, rule_confidence
        
        # التصنيف باستخدام الذكاء الاصطناعي
        model_results = []
        for model_name in self.models:
            label, confidence = self.classify_with_ai(text, model_name)
            if confidence >= self.min_confidence:
                model_results.append((label, confidence, model_name))
        
        if not model_results:
            return file_name, "غير محدد", 0.0
            
        # اختيار أفضل نتيجة مع مراعاة الثقة والتوافق
        best_result = max(model_results, key=lambda x: x[1])
        return file_name, best_result[0], best_result[1]

    def process_folder(self, input_folder, output_file):
        """معالجة مجلد كامل من الوثائق"""
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"المجلد المحدد غير موجود: {input_folder}")
        
        files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
        print(f" was found {len(files)} legal documenst")
        
        for file_name in tqdm(files, desc="🔎 جاري تصنيف الوثائق"):
            file_path = os.path.join(input_folder, file_name)
            
            try:
                doc_name, label, confidence = self.process_document(file_path)
                
                self.results.append({
                    "اسم_الملف": doc_name,
                    "التصنيف": label,
                    "ثقة_التصنيف": round(confidence, 3),
                    "وقت_التصنيف": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                print(f"✅ {doc_name[:25]}... | {label:15} | {confidence:.2f}")
            
            except Exception as e:
                print(f" خطأ في معالجة {file_name}: {str(e)}")
                continue
        
        self.save_results(output_file)
        self.generate_detailed_report()

    def save_results(self, output_file):
        """Save results with additional data"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df = pd.DataFrame(self.results)
        
        # تحسين تنسيق المخرجات
        df = df.sort_values(by=['ثقة_التصنيف'], ascending=False)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n Results saved in: {output_file}")

    def generate_detailed_report(self):
        """Create a detailed report with analysis"""
        if not self.results:
            print(" لا توجد نتائج لإنشاء التقرير")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n Advanced Analytical Report:")
        print(f"\n Total number {len(df)} documents")
        
        # تحليل التوزيع
        print("\n Categories distribution:")
        dist = df['التصنيف'].value_counts()
        print(dist.to_string())
        
        # تحليل الثقة
        print("\n Average confidence by classification:")
        conf_table = df.groupby('التصنيف')['ثقة_التصنيف'].agg(['mean', 'count'])
        print(conf_table.round(2).to_string())
        
        # تحليل الملفات منخفضة الثقة
        low_conf = df[df['ثقة_التصنيف'] < 0.7]
        if not low_conf.empty:
            print("\n Files with low confidence (<0.7):")
            print(low_conf[['اسم_الملف', 'التصنيف', 'ثقة_التصنيف']].to_string(index=False))

if __name__ == "__main__":
    # تهيئة المسارات
    INPUT_DIR = "../data/txt_files"
    OUTPUT_PATH = "../outputs/classified_docs_enhanced.csv"
    
    # تشغيل النظام
    print("\n Advanced Classification System for Arabic Legal Documents")
    print("----------------------------------------")
    
    classifier = AdvancedArabicLegalClassifier()
    try:
        classifier.process_folder(INPUT_DIR, OUTPUT_PATH)
        print("\n Classification completed successfully!")
    except Exception as e:
        print(f"\n an error occured {str(e)}")