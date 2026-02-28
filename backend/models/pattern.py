from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class FailedApproach(BaseModel):
    approach: str
    why_failed: str


class ReasoningTrace(BaseModel):
    failed_approaches: list[FailedApproach] = []
    key_insight: str


class PatternMetadata(BaseModel):
    source: str  # "nvidia_ocr", "swe_bench", "github_issues", "synthetic"
    verification_status: str = "unverified"  # "verified", "unverified"
    success_rate: float = 0.0
    times_applied: int = 0
    estimated_token_savings: int = 10000
    difficulty: str = "medium"  # "easy", "medium", "hard"


class Pattern(BaseModel):
    pattern_id: str
    problem_class: str
    problem_signature: str
    domain_tags: list[str]
    reasoning_trace: ReasoningTrace
    solution_template: str
    generic_code_template: Optional[str] = None
    metadata: PatternMetadata
    refinements: list[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PatternCreate(BaseModel):
    """Input model for creating a pattern from raw problem/solution"""
    problem: str
    solution: str
    reasoning: Optional[str] = None
    source: str = "synthetic"


class ArenaResult(BaseModel):
    model: str
    solution: str
    chain_of_thought: Optional[str] = None
    tokens_used: int
    time_seconds: float
    cost_estimate: float


class ArenaComparison(BaseModel):
    problem: str
    results: list[ArenaResult]
    best_solution: ArenaResult
    verified: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)
