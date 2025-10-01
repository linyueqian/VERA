from __future__ import annotations

from typing import List, Tuple
import re


class WERCalculator:
    """Calculate Word Error Rate (WER) between reference and hypothesis text."""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for WER calculation."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into words."""
        normalized = WERCalculator.normalize_text(text)
        return normalized.split() if normalized else []
    
    @staticmethod
    def edit_distance(ref_words: List[str], hyp_words: List[str]) -> Tuple[int, List[List[int]]]:
        """
        Calculate edit distance using dynamic programming.
        Returns (distance, dp_matrix) for traceback.
        """
        m, n = len(ref_words), len(hyp_words)

        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],
                        dp[i][j-1],
                        dp[i-1][j-1]
                    )
        
        return dp[m][n], dp
    
    @staticmethod
    def get_alignment(ref_words: List[str], hyp_words: List[str], dp_matrix: List[List[int]]) -> List[Tuple[str, str, str]]:
        """
        Get alignment between reference and hypothesis using traceback.
        Returns list of (ref_word, hyp_word, operation).
        """
        m, n = len(ref_words), len(hyp_words)
        alignment = []
        
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                if ref_words[i-1] == hyp_words[j-1]:
                    alignment.append((ref_words[i-1], hyp_words[j-1], "MATCH"))
                    i -= 1
                    j -= 1
                elif dp_matrix[i][j] == dp_matrix[i-1][j-1] + 1:
                    alignment.append((ref_words[i-1], hyp_words[j-1], "SUB"))
                    i -= 1
                    j -= 1
                elif dp_matrix[i][j] == dp_matrix[i-1][j] + 1:
                    alignment.append((ref_words[i-1], "*", "DEL"))
                    i -= 1
                else:
                    alignment.append(("*", hyp_words[j-1], "INS"))
                    j -= 1
            elif i > 0:
                alignment.append((ref_words[i-1], "*", "DEL"))
                i -= 1
            else:
                alignment.append(("*", hyp_words[j-1], "INS"))
                j -= 1
        
        return list(reversed(alignment))
    
    @classmethod
    def calculate_wer(cls, reference: str, hypothesis: str, return_details: bool = False) -> dict:
        """
        Calculate Word Error Rate between reference and hypothesis.
        
        Args:
            reference: Ground truth text
            hypothesis: Predicted text (e.g., from ASR)
            return_details: If True, return detailed alignment information
            
        Returns:
            Dictionary with WER metrics and optionally alignment details
        """
        ref_words = cls.tokenize(reference)
        hyp_words = cls.tokenize(hypothesis)

        if len(ref_words) == 0:
            if len(hyp_words) == 0:
                result = {
                    "wer": 0.0,
                    "substitutions": 0,
                    "deletions": 0, 
                    "insertions": 0,
                    "total_words": 0,
                    "reference_length": 0,
                    "hypothesis_length": 0
                }
            else:
                result = {
                    "wer": float('inf'),
                    "substitutions": 0,
                    "deletions": 0,
                    "insertions": len(hyp_words),
                    "total_words": len(hyp_words),
                    "reference_length": 0,
                    "hypothesis_length": len(hyp_words)
                }
        else:
            edit_dist, dp_matrix = cls.edit_distance(ref_words, hyp_words)

            if return_details:
                alignment = cls.get_alignment(ref_words, hyp_words, dp_matrix)
                substitutions = sum(1 for _, _, op in alignment if op == "SUB")
                deletions = sum(1 for _, _, op in alignment if op == "DEL") 
                insertions = sum(1 for _, _, op in alignment if op == "INS")
            else:
                alignment = None
                substitutions = 0
                deletions = 0
                insertions = 0
            
            wer = edit_dist / len(ref_words)
            
            result = {
                "wer": wer,
                "substitutions": substitutions,
                "deletions": deletions,
                "insertions": insertions,
                "total_words": edit_dist,
                "reference_length": len(ref_words),
                "hypothesis_length": len(hyp_words)
            }
        
        if return_details:
            result["alignment"] = alignment
            result["reference_words"] = ref_words
            result["hypothesis_words"] = hyp_words
            
        return result
    
    @classmethod
    def batch_calculate_wer(cls, pairs: List[Tuple[str, str]], return_details: bool = False) -> List[dict]:
        """Calculate WER for multiple reference-hypothesis pairs."""
        return [cls.calculate_wer(ref, hyp, return_details) for ref, hyp in pairs]