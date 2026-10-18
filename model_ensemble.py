"""Multi-model ensemble for improved retrosynthesis predictions."""

import logging
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Single model prediction result."""
    model_name: str
    reactants: List[str]
    confidence: float
    beam_rank: int  # Position in beam
    execution_time: float


@dataclass
class EnsembleResult:
    """Aggregated ensemble prediction."""
    top_predictions: List[Dict]  # Ranked predictions with scores
    consensus_reactants: Optional[List[str]]  # Most voted reactants
    ensemble_confidence: float
    individual_results: List[PredictionResult]
    agreement_score: float  # 0-1, how much models agree


class ModelEnsemble:
    """Manages multiple retrosynthesis models with voting and consensus."""

    def __init__(self, models: Dict, tokenizers: Dict, device: str = "cuda"):
        """
        Args:
            models: Dict of {model_name: model_object}
            tokenizers: Dict of {model_name: tokenizer_object}
            device: torch device
        """
        self.models = models
        self.tokenizers = tokenizers
        self.device = device
        self.num_models = len(models)
        self.executor = ThreadPoolExecutor(max_workers=min(4, len(models)))

    def _predict_single_model(
        self,
        model_name: str,
        smiles: str,
        num_beams: int = 5,
        max_length: int = 128,
    ) -> List[PredictionResult]:
        """Get predictions from a single model."""
        import time
        start = time.perf_counter()
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        try:
            inputs = tokenizer(smiles, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    num_beams=num_beams,
                    num_return_sequences=num_beams,
                    max_length=max_length,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            
            reactants_list = tokenizer.batch_decode(
                outputs.sequences,
                skip_special_tokens=True
            )
            
            # Compute confidence from beam scores
            sequences_scores = outputs.sequences_scores if hasattr(outputs, 'sequences_scores') else None
            
            results = []
            for beam_rank, reactants_str in enumerate(reactants_list):
                reactants = reactants_str.replace(" ", "").rstrip(".").split(".")
                
                # Normalize score to confidence [0, 1]
                confidence = 1.0 / (1.0 + beam_rank) if sequences_scores is None else torch.sigmoid(sequences_scores[beam_rank]).item()
                
                results.append(PredictionResult(
                    model_name=model_name,
                    reactants=reactants,
                    confidence=confidence,
                    beam_rank=beam_rank,
                    execution_time=time.perf_counter() - start
                ))
            
            return results
        except Exception as e:
            logger.error(f"Model {model_name} prediction failed: {e}")
            return []

    def predict_ensemble(
        self,
        smiles: str,
        num_beams: int = 5,
        max_length: int = 128,
        voting_threshold: float = 0.5,
    ) -> EnsembleResult:
        """
        Get predictions from all models and aggregate results.
        
        Args:
            smiles: Target molecule SMILES
            num_beams: Beams per model
            max_length: Max sequence length
            voting_threshold: Minimum agreement for consensus
        
        Returns:
            EnsembleResult with ranked predictions and consensus
        """
        all_results = []
        
        # Parallel model inference
        futures = {}
        for model_name in self.models.keys():
            future = self.executor.submit(
                self._predict_single_model,
                model_name,
                smiles,
                num_beams,
                max_length
            )
            futures[model_name] = future
        
        for model_name, future in futures.items():
            try:
                results = future.result(timeout=60)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Failed to get results from {model_name}: {e}")
        
        if not all_results:
            return EnsembleResult(
                top_predictions=[],
                consensus_reactants=None,
                ensemble_confidence=0.0,
                individual_results=[],
                agreement_score=0.0
            )
        
        # Aggregate predictions
        reactants_votes = defaultdict(list)
        reactants_confidence = defaultdict(list)
        
        for result in all_results:
            reactants_tuple = tuple(sorted(result.reactants))
            reactants_votes[reactants_tuple].append(result.model_name)
            reactants_confidence[reactants_tuple].append(result.confidence)
        
        # Rank predictions by vote count and average confidence
        ranked_predictions = []
        for reactants_tuple, models in sorted(
            reactants_votes.items(),
            key=lambda x: (len(x[1]), np.mean(reactants_confidence[x[0]])),
            reverse=True
        ):
            agreement = len(models) / self.num_models
            avg_confidence = np.mean(reactants_confidence[reactants_tuple])
            
            ranked_predictions.append({
                "reactants": list(reactants_tuple),
                "votes": len(models),
                "agreeing_models": models,
                "agreement_score": agreement,
                "average_confidence": avg_confidence,
                "combined_score": 0.6 * agreement + 0.4 * avg_confidence,
            })
        
        # Determine consensus
        consensus = None
        agreement_score = 0.0
        if ranked_predictions:
            top = ranked_predictions[0]
            agreement_score = top["agreement_score"]
            if agreement_score >= voting_threshold:
                consensus = top["reactants"]
        
        ensemble_confidence = ranked_predictions[0]["combined_score"] if ranked_predictions else 0.0
        
        return EnsembleResult(
            top_predictions=ranked_predictions[:5],  # Top 5 predictions
            consensus_reactants=consensus,
            ensemble_confidence=ensemble_confidence,
            individual_results=all_results,
            agreement_score=agreement_score
        )

    def predict_batch_ensemble(
        self,
        smiles_list: List[str],
        num_beams: int = 5,
        batch_size: int = 16,
    ) -> List[EnsembleResult]:
        """Predict for batch of molecules."""
        results = []
        for smiles in smiles_list:
            result = self.predict_ensemble(smiles, num_beams)
            results.append(result)
        
        return results


class ConfidenceCalibrator:
    """Calibrate model confidence scores."""

    @staticmethod
    def sigmoid_calibration(scores: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
        """Apply sigmoid calibration to raw scores."""
        return 1.0 / (1.0 + np.exp(-alpha * scores - beta))

    @staticmethod
    def temperature_scaling(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Apply temperature scaling to logits."""
        return torch.softmax(torch.tensor(logits) / temperature, dim=-1).numpy()

    @staticmethod
    def is_ensemble_prediction_reliable(
        result: EnsembleResult,
        confidence_threshold: float = 0.6,
        agreement_threshold: float = 0.5,
    ) -> Tuple[bool, str]:
        """Check if ensemble prediction is reliable."""
        if result.ensemble_confidence < confidence_threshold:
            return False, f"Low confidence: {result.ensemble_confidence:.2f}"
        
        if result.agreement_score < agreement_threshold:
            return False, f"Low model agreement: {result.agreement_score:.2f}"
        
        return True, "Reliable prediction"


class PredictionExplainability:
    """Explain why models made specific predictions."""

    @staticmethod
    def generate_explanation(result: EnsembleResult) -> Dict:
        """Generate human-readable explanation."""
        return {
            "top_prediction": result.top_predictions[0] if result.top_predictions else None,
            "consensus": {
                "reactants": result.consensus_reactants,
                "confidence": result.ensemble_confidence,
                "agreement": f"{result.agreement_score:.1%}",
            },
            "alternatives": result.top_predictions[1:4] if len(result.top_predictions) > 1 else [],
            "model_distribution": {
                pred["reactants"][0]: len(pred["agreeing_models"])
                for pred in result.top_predictions[:3]
            },
        }
