from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AttackSequence:
    """Container for attack technique sequences processed by the pipeline."""

    path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    branch_id: str = "root"
    completed: bool = False

    def append(self, technique_id: str, *, score: Optional[float] = None, source: Optional[str] = None) -> "AttackSequence":
        """Create a new sequence with an additional technique step."""

        normalized = str(technique_id).strip().upper()
        next_metadata = {**self.metadata}
        history = list(next_metadata.get("history", []))
        history.append({"technique_id": normalized, "score": score, "source": source})
        next_metadata["history"] = history
        branch_suffix = normalized if not self.branch_id.endswith(normalized) else normalized
        return AttackSequence(
            path=[*self.path, normalized],
            metadata=next_metadata,
            branch_id=f"{self.branch_id}.{branch_suffix}",
            completed=self.completed,
        )

    def mark_complete(self) -> "AttackSequence":
        """Return a copy of the sequence marked as complete."""

        return AttackSequence(
            path=list(self.path),
            metadata={**self.metadata},
            branch_id=self.branch_id,
            completed=True,
        )
