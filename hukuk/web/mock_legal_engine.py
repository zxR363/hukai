import asyncio
import random
import os

class MockHit:
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload

class LegalSearchEngine:
    def __init__(self):
        pass

    def connect_db(self):
        return True

    def validate_user_input(self, story, topic):
        return True

    def retrieve_raw_candidates(self, full_query):
        # Return mock candidates that look like Qdrant hits
        return [
            MockHit(0.95, {"source": "Mock_Mevzuat_1.pdf", "page": 1, "type": "MEVZUAT", "page_content": "Mock mevzuat içeriği..."}),
            MockHit(0.88, {"source": "Mock_Karar_1.pdf", "page": 5, "type": "EMSAL KARAR", "page_content": "Mock karar içeriği..."})
        ]

    def close(self):
        pass

class LegalJudge:
    def __init__(self):
        pass

    def generate_expanded_queries(self, story, topic):
        return ["mock sorgu 1", "mock sorgu 2"]

    def evaluate_candidates(self, candidates, story, topic, negatives):
        valid_docs = []
        for i, hit in enumerate(candidates):
            valid_docs.append({
                "source": hit.payload['source'],
                "page": hit.payload['page'],
                "type": hit.payload['type'],
                "role": "[EMSAL İLKE]" if i % 2 == 0 else "[DOĞRUDAN DELİL]",
                "text": hit.payload['page_content'] + "\n(MOCK DATA)",
                "score": hit.score * 100,
                "reason": "Bu bir mock değerlendirme gerekçesidir."
            })
        return valid_docs

    def generate_final_opinion(self, story, topic, context_str):
        return f"MOCK ANALİZ SONUCU:\n\n{story} ve {topic} özelinde yaptığımız simülasyon sonucunda, ekli belgelerin lehinize olduğu değerlendirilmektedir."

class LegalReporter:
    @staticmethod
    def create_report(user_story, valid_docs, advice_text, filename="results/Mock_Rapor.pdf"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"MOCK REPORT\nStory: {user_story}\nAdvice: {advice_text}")
        return True
