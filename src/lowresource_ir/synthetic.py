"""Synthetic multilingual corpus for low-resource IR and QA research.

All documents, questions, and answers are fictional. The corpus is intended for
reproducible retrieval experiments, not linguistic benchmarking claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class SyntheticIRConfig:
    seed: int = 42
    include_regional_placeholders: bool = True


DOCUMENTS: list[dict[str, str]] = [
    {
        "doc_id": "UR-HEALTH-001", "language": "urdu", "script": "arabic", "topic": "health",
        "title": "بنیادی صحت مرکز کی ہدایات",
        "content": "بنیادی صحت مرکز پیر سے جمعرات صبح نو بجے سے شام پانچ بجے تک کھلا رہتا ہے۔ حاملہ خواتین کے لیے مفت معائنہ دستیاب ہے۔ ہنگامی حالت میں قریبی ضلعی ہسپتال سے رابطہ کیا جائے۔",
        "answer_fact": "بنیادی صحت مرکز پیر سے جمعرات صبح نو بجے سے شام پانچ بجے تک کھلا رہتا ہے۔",
    },
    {
        "doc_id": "UR-WATER-002", "language": "urdu", "script": "arabic", "topic": "water",
        "title": "صاف پانی منصوبہ",
        "content": "صاف پانی منصوبے کے تحت گاؤں میں تین فلٹریشن پوائنٹس لگائے گئے ہیں۔ پانی کے نمونے ہر ماہ لیبارٹری بھیجے جاتے ہیں۔ شکایت کے لیے مقامی کونسل دفتر میں فارم جمع کرائیں۔",
        "answer_fact": "گاؤں میں تین فلٹریشن پوائنٹس لگائے گئے ہیں۔",
    },
    {
        "doc_id": "AR-SCHOOL-001", "language": "arabic", "script": "arabic", "topic": "education",
        "title": "برنامج دعم المدارس",
        "content": "يوفر برنامج دعم المدارس حصص تقوية في الرياضيات والقراءة بعد الدوام. التسجيل متاح للطلاب من الصف السادس إلى الصف التاسع. يجب على ولي الأمر توقيع نموذج الموافقة.",
        "answer_fact": "التسجيل متاح للطلاب من الصف السادس إلى الصف التاسع.",
    },
    {
        "doc_id": "AR-AGRI-002", "language": "arabic", "script": "arabic", "topic": "agriculture",
        "title": "إرشادات الري للمزارعين",
        "content": "تنصح النشرة الزراعية بري المحاصيل في الصباح الباكر لتقليل فقدان المياه. في موجات الحر يجب فحص رطوبة التربة مرتين في الأسبوع. الدعم الفني متاح في مركز الإرشاد الزراعي.",
        "answer_fact": "تنصح النشرة بري المحاصيل في الصباح الباكر لتقليل فقدان المياه.",
    },
    {
        "doc_id": "EN-FLOOD-001", "language": "english", "script": "latin", "topic": "disaster",
        "title": "Community Flood Preparedness",
        "content": "The flood preparedness committee stores sandbags at the union council office. Families should keep drinking water, medicine, and copies of identity documents in a safe bag. The school building is the temporary shelter.",
        "answer_fact": "The school building is the temporary shelter.",
    },
    {
        "doc_id": "EN-JOBS-002", "language": "english", "script": "latin", "topic": "employment",
        "title": "Youth Skills Workshop",
        "content": "The youth skills workshop teaches basic computer use, interview preparation, and online application skills. Registration closes on 15 August. Applicants must bring an identity card copy.",
        "answer_fact": "Registration closes on 15 August.",
    },
    {
        "doc_id": "RU-HEALTH-003", "language": "roman_urdu", "script": "latin", "topic": "health",
        "title": "Sehat camp ka schedule",
        "content": "Sehat camp jumma ke din subah 10 baje shuru hota hai. Bachon ki vaccination aur sugar test muft kiye jate hain. Registration ke liye union council desk par naam likhwain.",
        "answer_fact": "Sehat camp jumma ke din subah 10 baje shuru hota hai.",
    },
    {
        "doc_id": "RU-EDU-004", "language": "roman_urdu", "script": "latin", "topic": "education",
        "title": "Shaam ki taleemi class",
        "content": "Shaam ki class mein English reading, basic math, aur computer practice karai jati hai. Class haftay mein teen din community center mein hoti hai. Fees nahi li jati.",
        "answer_fact": "Class haftay mein teen din community center mein hoti hai.",
    },
]

REGIONAL_PLACEHOLDERS: list[dict[str, str]] = [
    {
        "doc_id": "RG-AGRI-001", "language": "regional_placeholder", "script": "latin", "topic": "agriculture",
        "title": "Regional seed distribution notice",
        "content": "The regional agriculture office distributes drought-resistant seed packets every Tuesday. Farmers must register their village and land size before collection.",
        "answer_fact": "Seed packets are distributed every Tuesday.",
    }
]

QUERIES: list[dict[str, str | int]] = [
    {"query_id": "Q-001", "query_language": "english", "query": "When is the basic health center open?", "expected_doc_id": "UR-HEALTH-001", "expected_language": "urdu"},
    {"query_id": "Q-002", "query_language": "urdu", "query": "گاؤں میں پانی کے کتنے فلٹریشن پوائنٹس ہیں؟", "expected_doc_id": "UR-WATER-002", "expected_language": "urdu"},
    {"query_id": "Q-003", "query_language": "english", "query": "Which grades can register for the school support program?", "expected_doc_id": "AR-SCHOOL-001", "expected_language": "arabic"},
    {"query_id": "Q-004", "query_language": "arabic", "query": "متى يجب ري المحاصيل لتقليل فقدان المياه؟", "expected_doc_id": "AR-AGRI-002", "expected_language": "arabic"},
    {"query_id": "Q-005", "query_language": "urdu", "query": "سیلاب کے دوران عارضی پناہ گاہ کہاں ہے؟", "expected_doc_id": "EN-FLOOD-001", "expected_language": "english"},
    {"query_id": "Q-006", "query_language": "english", "query": "When does registration close for the youth skills workshop?", "expected_doc_id": "EN-JOBS-002", "expected_language": "english"},
    {"query_id": "Q-007", "query_language": "roman_urdu", "query": "Sehat camp kab shuru hota hai?", "expected_doc_id": "RU-HEALTH-003", "expected_language": "roman_urdu"},
    {"query_id": "Q-008", "query_language": "english", "query": "How many days per week is the evening class held?", "expected_doc_id": "RU-EDU-004", "expected_language": "roman_urdu"},
    {"query_id": "Q-009", "query_language": "english", "query": "Where can I renew a passport?", "expected_doc_id": "NONE", "expected_language": "none"},
]


def generate_synthetic_corpus(config: SyntheticIRConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or SyntheticIRConfig()
    docs = list(DOCUMENTS)
    if cfg.include_regional_placeholders:
        docs.extend(REGIONAL_PLACEHOLDERS)
    documents = pd.DataFrame(docs)
    documents["data_origin"] = "synthetic_fictional_multilingual_corpus"
    queries = pd.DataFrame(QUERIES)
    queries["data_origin"] = "synthetic_fictional_multilingual_queries"
    return documents, queries
