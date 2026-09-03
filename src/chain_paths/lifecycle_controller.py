"""Lifecycle orchestration for staged attack prediction."""
from typing import Any, Dict, Iterable, List, Optional

from .apply_ontology_reasoning import apply_ontology_reasoning
from .attack_sequence import AttackSequence
from .extract_features import extract_features
from .fuse_evidence import prioritize_candidates
from .map_to_ttp import map_to_ttp
from .parse_input_data import parse_input_data
from .predict_next_step import predict_next_step
from .preprocess_sequence import preprocess_sequence
from .predictor import Predictor


class AttackLifecycleController:
    """Controller that walks an attack sequence through the staged pipeline."""

    def __init__(self, predictor: Predictor, *, branch_factor: int = 3, max_depth: int = 6,
                 max_paths: int = 10, top_k_tactics: int = 3, verbose: bool = False):
        self.predictor = predictor
        self.branch_factor = max(branch_factor, 1)
        self.max_depth = max_depth
        self.max_paths = max(max_paths, 1)
        self.top_k_tactics = max(top_k_tactics, 1)
        self.verbose = verbose

    def _format_tactic_chain(self, techniques: List[str]) -> List[str]:
        """Return readable TECHNIQUE -> TACTIC strings for the path."""

        chain: List[str] = []
        for tech in techniques:
            tactics = self.predictor.technique_to_tactics.get(tech, [])
            tactic_display = "|".join(tactics) if tactics else "unknown"
            chain.append(f"{tech} -> {tactic_display}")
        return chain

    def simulate_attack(
        self,
        seeds: Iterable[str],
        *,
        group_context: Optional[Any] = None,
    ) -> Dict[str, List[Dict[str, object]]]:
        parsed_seq = parse_input_data(seeds)
        preproc_seq = preprocess_sequence(parsed_seq)

        seen = set()
        deduped_path: List[str] = []
        for tech in preproc_seq.path:
            if tech in seen:
                continue
            seen.add(tech)
            deduped_path.append(tech)
        if len(deduped_path) != len(preproc_seq.path):
            if self.verbose:
                print(f"[Lifecycle] Deduplicated seed: {len(preproc_seq.path)} -> {len(deduped_path)} techniques")
            preproc_seq = AttackSequence(
                path=deduped_path,
                metadata=preproc_seq.metadata,
                branch_id=preproc_seq.branch_id,
                completed=preproc_seq.completed,
            )
        active: List[AttackSequence] = [preproc_seq]
        finished: Dict[str, List[Dict[str, object]]] = {}
        finished_count = 0

        while active and finished_count < self.max_paths:
            current = active.pop(0)
            if current.completed or len(current.path) >= self.max_depth:
                finished[current.branch_id] = [{
                    "techniques": current.path,
                    "technique_tactic_chain": self._format_tactic_chain(current.path),
                }]
                finished_count += 1
                if self.verbose:
                    print(f"[Lifecycle] Completed path {finished_count}/{self.max_paths}: {current.path[:3]}...")
                continue

            features = extract_features(current, self.predictor, group_context=group_context)
            reasoning_output = apply_ontology_reasoning(features, self.predictor)
            
            # Get tactic probabilities and filter to top-K tactics
            tactic_probs = self.predictor.predict_next_tactics(
                current.path,
                top_k=self.top_k_tactics,
                group_context=group_context,
            )
            allowed_tactics = {tactic for tactic, _ in tactic_probs} if tactic_probs else set()
            
            # Filter reasoning output to allowed tactics
            if allowed_tactics:
                reasoning_output['allowed_tactics'] = allowed_tactics
            
            model_output = predict_next_step(
                current.path,
                self.predictor,
                group_context=group_context,
                top_n_candidates=self.branch_factor * 3,
            )
            ranked_predictions = prioritize_candidates(
                model_output,
                reasoning_output,
                max_candidates=self.branch_factor,
            )
            mapped_output = map_to_ttp(ranked_predictions, self.predictor)

            if not mapped_output:
                finished[current.branch_id] = [{
                    "techniques": current.path,
                    "technique_tactic_chain": self._format_tactic_chain(current.path),
                    "reasoning": reasoning_output,
                }]
                finished_count += 1
                if self.verbose:
                    print(f"[Lifecycle] No candidates: completed path {finished_count}/{self.max_paths}")
                continue

            for candidate in mapped_output:
                if candidate["technique_id"] in current.path:
                    continue
                updated_sequence = current.append(
                    candidate["technique_id"],
                    score=candidate.get("score"),
                    source="predictor",
                )
                if len(updated_sequence.path) >= self.max_depth:
                    finished[updated_sequence.branch_id] = [
                        {
                            "techniques": updated_sequence.path,
                            "technique_tactic_chain": self._format_tactic_chain(updated_sequence.path),
                            "metadata": updated_sequence.metadata,
                            "mapped_output": candidate,
                        }
                    ]
                    finished_count += 1
                    if self.verbose:
                        print(f"[Lifecycle] Max depth reached: path {finished_count}/{self.max_paths}")
                else:
                    active.append(updated_sequence)
        
        if self.verbose:
            print(f"[Lifecycle] Simulation complete: {finished_count} paths generated")

        return finished


def simulate_attack(
    predictor: Predictor,
    seeds: Iterable[str],
    *,
    group_context: Optional[Any] = None,
    branch_factor: int = 3,
    max_depth: int = 6,
    max_paths: int = 10,
    top_k_tactics: int = 3,
    verbose: bool = False,
) -> Dict[str, List[Dict[str, object]]]:
    """Convenience wrapper to run the controller in a single call."""

    controller = AttackLifecycleController(
        predictor,
        branch_factor=branch_factor,
        max_depth=max_depth,
        max_paths=max_paths,
        top_k_tactics=top_k_tactics,
        verbose=verbose,
    )
    return controller.simulate_attack(seeds, group_context=group_context)
