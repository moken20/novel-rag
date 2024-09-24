import re

import pandas as pd

from src.utils.preprocess.cleans import replace_symbols

class RetieverEvaluator:
    def __init__(self, evidences_texts: pd.Series, retrieved_texts_list: list[pd.Series]) -> None:
        self.evidence_texts = evidences_texts
        self.retrieved_texts_list = retrieved_texts_list

        if len(self.evidence_texts) != len(self.retrieved_texts_list):
            raise ValueError('number of evidences and number of retrieved_texts do not match.'
                             f'num evidences:{len(self.evidences)}, num retrieved_texts:{len(self.retrieved_texts)}')

    def _extract_evidences(self, text: str) -> list[str]:
        text = replace_symbols(text)
        matches = re.findall(r'「(.*?)」', text)
        matches = [s.replace('『', '「').replace('』', '」') for s in matches]
        return matches
    
    def compute_mean_recall(self):
        recall_score = 0
        num_no_evidence = 0
        self.failed_retrieved_evidence = []

        for evidence_txt, retrieved_texts in zip(self.evidence_texts.values, self.retrieved_texts_list):
            evidences = self._extract_evidences(evidence_txt)
            if len(evidences) == 0:
                num_no_evidence += 1
                continue
            recall_score += self.compute_recall(evidences, retrieved_texts)
        
        return recall_score / (len(self.evidence_texts) - num_no_evidence)

    
    def compute_recall(self, evidences: list[str], retrieved_texts: pd.Series) -> float:
        total_evidences = len(evidences)

        num_retrieved_relevant_text = 0
        for evidence in evidences:
            if retrieved_texts.str.replace('\n', '').str.contains(evidence).any():
                num_retrieved_relevant_text += 1
            else:
                self.failed_retrieved_evidence.append(evidence)
        
        recall_score = num_retrieved_relevant_text / total_evidences
        return recall_score
