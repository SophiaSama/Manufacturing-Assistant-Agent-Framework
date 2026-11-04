# Engineering Masterclass: Building Production Knowledge Graphs for Advanced RAG (GraphRAG)

**Instructor**: Principal AI Systems Architect, Enterprise Agentic Systems  
**Domain**: Industrial Automation & Precision Manufacturing (Next-Gen Automation / NGA)  
**Target Audience**: Staff Software Engineers, AI/ML Platform Architects, and Enterprise Tech Leads  
**Prerequisites**: Python 3.10+, NetworkX, Pydantic v2, Vector Embeddings, LLM Tool Calling  

---

## Instructor Welcome & Course Architecture

Welcome, engineers. If you are here today, it is because you have built a standard Retrieval-Augmented Generation (RAG) pipeline—you embedded markdown documents with an off-the-shelf embedding model, dumped the vectors into a vector database, wired top-$k$ cosine similarity to an LLM context window, and demoed it successfully to executive leadership.

And then you deployed it to factory floor operators, field technicians, and compliance auditors.

Within forty-eight hours, the cracks became chasms. When a line technician asked: *"What is the calibrated torque tolerance for Station 144's wheel nutrunner, and which standard operating procedure governs its shift recalibration?"*, your vector retriever handed the model three unrelated chunks about conveyor belt maintenance and oil viscosity. When quality engineers asked for a line-wide summary of every Class A safety defect recorded across two hundred Non-Conformance Records (NCRecords) in Q3, your top-$k$ search dutifully fetched five vivid anecdotes and completely ignored the other one hundred and ninety-five incidents. And worst of all, when your system confused the `TQ-6012` high-speed DC nutrunner with the `TQ-6018` handheld quality audit wrench, an uncertified tool was used to torque critical chassis bolts, triggering an internal audit panic.

Naive vector retrieval is mathematically incapable of solving enterprise engineering questions. Vectors excel at fuzzy conceptual similarity, but they are blind to relational transitivity, helpless at global thematic aggregation, prone to catastrophic subword collisions on alphanumeric part numbers, and indifferent to causal directionality.

In this Masterclass, we are going to dismantle naive RAG and build a production-grade, enterprise-scale **Knowledge Graph + Advanced RAG (GraphRAG)** architecture from the ground up. We will write real, battle-tested Python code matching the architecture of our Next-Gen Automation (NGA) platform. 

Here is our flight plan for this intensive masterclass:
- **Foundation Level**: The mathematical and topological breakdown of naive vector RAG; core property graph theory; typed domain ontologies; triplet extraction primitives; and closed-loop schema enforcement.
- **Intermediate Level**: Anchored chunking; deterministic entity resolution and canonical indexing; equipment conflict guards; NetworkX `MultiDiGraph` construction; edge auto-flipping; Louvain hierarchical community detection; and server-side Role-Based Access Control (RBAC).
- **Advanced Level**: Tri-Hybrid Retrieval (Dense Vector + Lexical BM25 + RBAC-Filtered Local BFS Traversal); Reciprocal Rank Fusion (RRF); System One semantic reranking via TypeSafe AI Jev `Choice` primitives; verbatim evidence verification using Headroom content-addressed hashes; the 4-metric Graph Quality Index (GQI) evaluation framework; and interactive ego-network observability.

Grab your coffee. Open your terminals. Let us begin at the foundation.

---

# Foundation Level: The Limits of Naive RAG & Graph Fundamentals

```
==================================================================================================
                 NAIVE VECTOR RAG vs. KNOWLEDGE GRAPH TRAVERSAL
==================================================================================================

  Query: "What tool & torque spec does Station 144 use for wheel assembly under SOP-OPR-101?"

  [Naive Dense Vector Retrieval]                 [Knowledge Graph Multi-Hop Traversal]
  Query embedding q in R^d (d=1536)              Seed Match: [station-144]
           │                                              │
           ├─► Top-1: Chunk A (Station 144 overview)      ├─► (station-144)
           ├─► Top-2: Chunk B (General chassis fastening) │       │ executes_procedure
           └─► Top-3: Chunk X (Station 131 fluid fill)    │       ▼
                (Semantic drift: missed Hops 3 & 4!)      │   (sop-opr-101)
                                                          │       │ requires_tool
     Result: Incomplete answer or hallucination           │       ▼
                                                          │   (tq-6012)
                                                          │       │ specifies_torque
                                                          │       ▼
                                                          └──►(torque-target-105nm)
                                                          Result: Deterministic 4-Hop Path
==================================================================================================
```

### The 4 Fatal Breakdowns of Pure Dense Vector RAG

To understand why Knowledge Graphs are mandatory in high-consequence enterprise environments, we must examine the linear algebra and topology of dense embedding models. When you compute cosine similarity $S_C(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|\|\mathbf{d}\|}$, you are measuring whether two text passages occupy the same neighborhood of an isotropic semantic manifold. That is wonderful for topic matching, but it collapses under four common industrial query patterns.

#### 1. Multi-Hop Relational Blindspots (Transitivity Failure)
Consider tracing a fastener torque specification at the Northgate Assembly plant. The factual truth is partitioned across four separate engineering artifacts:
1. **Document 1 (Station 144 Profile)**: *"Station 144 executes chassis wheel assembly under SOP-OPR-101."*
2. **Document 2 (SOP-OPR-101 Standard Operating Procedure)**: *"Fasten wheel lug nuts using nutrunner TQ-6012."*
3. **Document 3 (TEC-314 Fastener Tooling Standard)**: *"TQ-6012 DC electric nutrunners must maintain target torque torque-target-105nm."*
4. **Document 4 (SOP-TEC-214 Tooling Calibration Specification)**: *"Target torque tolerance is 105 Nm ± 5% with max dynamic drift 2.0%."*

When an engineer asks: *"What is the calibrated torque tolerance for Station 144's wheel nutrunner?"*, the query vector contains semantic mass around *calibrated*, *torque tolerance*, *Station 144*, and *nutrunner*. 

What happens during vector search? Document 1 matches on *Station 144*. Document 4 matches on *calibrated torque tolerance*. But Documents 2 and 3 have low cosine similarity to the composite query because they discuss specific equipment models and fastener specifications without repeating the words "calibrated tolerance" or "Station 144". 

Because transitive relations $(e_1 \xrightarrow{r_1} e_2 \xrightarrow{r_2} e_3 \xrightarrow{r_3} e_4)$ are non-local in vector space, the cosine similarity between the query and intermediate connecting chunks decays exponentially with path length. The retriever drops Document 2 from the top-$k$ window. Denied the bridge connecting Station 144 to the tool `TQ-6012`, the LLM either hallucinates a fabricated torque specification or falsely asserts that the manual lacks the information.

#### 2. Corpus-Wide Thematic Aggregation Failures (Information Horizon Truncation)
Dense vector search solves the $k$-Nearest Neighbors ($k$-NN) optimization problem:
$$\underset{D' \subset \mathcal{D}, |D'|=k}{\operatorname{argmax}} \sum_{d \in D'} S_C(\mathbf{q}, \mathbf{d})$$
It cannot compute relational database operations like relational selections or group-by aggregations:
$$\sigma_{\text{type}=\text{DefectCode} \land \text{severity}=\text{Class A}}(\mathcal{D})$$

Suppose plant leadership asks: *"List every root cause for Class A safety critical defect codes recorded across plant Non-Conformance Records (NCRecords) in Q3."* 

A vector database searches against the query embedding and returns the top 5 or 10 chunks whose linguistic phrasing most vividly describes safety incidents. If the plant recorded 120 NC reports during Q3, the remaining 110 reports remain completely beyond the vector retriever's information horizon. Vector search is mathematically incapable of exhaustive enumeration, making statistical root-cause clustering, FMEA calculations, and line-wide quality audits impossible.

#### 3. Entity Ambiguity & Alphanumeric Token Fragmentation
Language model tokenizers (BPE, WordPiece) break rare or structured alphanumeric codes into subwords:
- `TQ-6012` becomes `['T', 'Q', '-', '60', '12']`
- `TQ-6018` becomes `['T', 'Q', '-', '60', '18']`

In high-dimensional embedding space, the contextual self-attention weights generated by ubiquitous manufacturing terms ("torque", "tightening", "tool", "fastener", "specification") overwhelm the single subword difference at the tail of the token stream. Consequently, the cosine similarity between an embedding of `TQ-6012` (the assembly DC nutrunner) and `TQ-6018` (the quality audit torque wrench) frequently exceeds $0.97$.

To a vector retriever, these two tools look virtually identical. But on the assembly line, using a quality audit wrench to perform production assembly violates station cycle times and calibration standards. Vector retrieval cannot enforce discrete entity identity.

#### 4. Inability to Follow Causal, Procedural, and Temporal Dependency Chains
Dense vector similarity is symmetric: $\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle$. But enterprise workflows are strictly asymmetric directed acyclic graphs (DAGs) and state machines. Consider this manufacturing failure cascade:
$$\text{FaultCode E-4012} \xrightarrow{\text{causes}} \text{DefectCode Class A Weld Spatter} \xrightarrow{\text{triggers\_recall}} \text{RecallCriteria QCR-501} \xrightarrow{\text{requires}} \text{Procedure FAP-401 8D} \xrightarrow{\text{escalates\_to}} \text{EscalationLevel ESC-402}$$

When an operator asks: *"What procedure must be initiated when FaultCode E-4012 causes weld spatter?"*, an undirected vector search retrieves chunks mentioning general spatter cleanup, robot maintenance, and retrospective 8D reports. It scrambles the causal chain, providing steps out of order or bypassing mandatory escalation gates.

---

### Core Knowledge Graph Concepts & Formal Graph Theory

To eliminate these failure modes, we structure enterprise knowledge into a **Property Multigraph** governed by a formal domain ontology. Let us define our mathematical primitives:

1. **Entities ($V$)**: Unique, discrete domain concepts $u \in V$. Each entity carries a canonical identifier slug $u_{\text{id}}$, a categorical ontology type $\tau(u) \in \mathcal{T}_V$, a human-readable display name, verified aliases, an RBAC clearance rank $\text{rank}(u) \in \{1, 2, 3, 4\}$, and content-addressed document chunk provenance pointers $\text{sources}(u)$.
2. **Directed Relations ($E$)**: Directed edges connecting a source entity to a target entity: $e = (u, v, k) \in V \times V \times \mathcal{R}$. Edge direction establishes semantic polarity: $(u) \xrightarrow{\text{causes}} (v)$ means that $u$ causes $v$, never the inverse.
3. **Typed Domain Ontology**: A formal schema specification defining the allowable entity vocabulary $\mathcal{T}_V$, relation vocabulary $\mathcal{R}$, and directional domain and range constraints $\Omega: \mathcal{R} \to \mathcal{P}(\mathcal{T}_V) \times \mathcal{P}(\mathcal{T}_V)$. In our Next-Gen Automation (NGA) platform, we enforce:
   - **15 Entity Types**: `Machine`, `Station`, `SOP`, `FaultCode`, `DefectCode`, `Part`, `Supplier`, `Personnel`, `Torque`, `Threshold`, `Procedure`, `NCRecord`, `WorkOrder`, `RecallCriteria`, `EscalationLevel`.
   - **12 Relation Types**: `causes`, `indicates`, `prevents`, `monitors`, `threshold_for`, `documented_in`, `escalates_to`, `requires`, `audited_by`, `owned_by`, `certified_for`, `triggers_recall`.
   - *Ontological Validity Invariant*: A relation $(u, v, r)$ is admissible if and only if $\tau(u) \in \Omega_{\text{src}}(r)$ and $\tau(v) \in \Omega_{\text{tgt}}(r)$.
4. **Property Multigraph (`MultiDiGraph`)**: Real enterprise domains contain multiple distinct relationships between the identical pair of entities. For example, between `Station-144` and `TQ-6012`:
   - Edge 1: `(station-144) --requires--> (tq-6012)` (Tooling requirement, Rank 1)
   - Edge 2: `(station-144) --monitors--> (tq-6012)` (Telemetry monitoring, Rank 2)  
   A simple graph ($E \subseteq V \times V$) collapses or overwrites these edges. A `MultiDiGraph` maintains discrete parallel edges indexed by relation key, each with its own provenance, confidence score, and access clearance.

---

### Triplet Extraction Primitives & Production Realities

The atomic building block of a knowledge graph is the semantic triplet:
$$\tau = (s, p, o) \in \mathcal{E} \times \mathcal{R} \times \mathcal{E}$$
where $s$ is the Subject entity, $p$ is the Predicate relation, and $o$ is the Object entity.

In academic tutorials, instructors show you an open-ended prompt: *"Extract all entities and relations as triplets from this text."* 

In production, deploying an unconstrained single-pass prompt against thousands of engineering documents produces a catastrophic disaster. Here are the six failure modes you will encounter within your first hour:

```
==================================================================================================
                             THE 6 EXTRACTION BREAKDOWN MODES
==================================================================================================

  [Raw Text Chunk]
         │
         ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────┐
  │ 1. Entity Fragmentation   : "Station 144", "Stn 144", "Wheel Stn"                          │
  │                             --> Creates 3 isolated singleton nodes; destroys multi-hop     │
  │ 2. Relation Hallucination : "tightened_with", "uses_hardware", "fastens_by"                │
  │                             --> Unbounded predicate vocabulary makes graph search blow up  │
  │ 3. Namespace Clutter      : "machine-tq-6012" vs "tq-6012"                                 │
  │                             --> Naive type prefix duplication fragments hardware nodes     │
  │ 4. Direction Inversion    : (tq-6012) --requires--> (sop-opr-101)                          │
  │                             --> Passive voice inverts edges; backward traversals fail      │
  │ 5. Dangling Endpoints     : (sop-opr-101) --requires--> (ghost-wrench)                     │
  │                             --> Target omitted from entity array; foreign key crash        │
  │ 6. Noise Amplification    : Low-confidence speculative suggestions extracted as hard facts │
  │                             --> Pollutes compliance graph with uncertified procedures      │
  └────────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **Entity Fragmentation**: One technician writes *"Station 144"*, an SOP writes *"Stn-144"*, and a shift log writes *"Wheel Assembly Cell"*. Without deterministic alias mapping, the extractor creates three separate nodes, atomizing your graph topology into disconnected islands.
2. **Relation Hallucination**: Without a fixed schema, the LLM invents sixty synonymous phrases for the same physical relationship (`"is_tightened_using"`, `"utilizes_tool"`, `"fastened_via"`). Graph traversal (BFS/DFS) relies on querying known predicate keys (`requires`). An unbounded vocabulary makes programmatic traversal impossible.
3. **Namespace Clutter**: Extractors prepend entity types into IDs: `"machine-tq-6012"` in chunk 1 and `"tq-6012"` in chunk 2. Without prefix stripping, the same tool exists twice.
4. **Direction Inversion**: Passive voice sentences (*"Tooling TQ-6012 is required by SOP-OPR-101"*) cause the model to emit `(TQ-6012) --requires--> (SOP-OPR-101)`. The dependency arrow is inverted, breaking downstream topological sorting.
5. **Dangling Endpoints**: The LLM outputs an edge referencing an entity that does not exist in the extracted `entities` list, violating referential integrity.
6. **Noise Amplification**: Engineering discussions often include speculative thoughts (*"We could perhaps test with wrench TF-6000"*). Naive extractors record this as a certified relationship, polluting the compliance graph.

---

### Production Pitfall / War Story: The False-Positive Equipment Merge

> **War Story from the Plant Floor**:
> During early deployment of our knowledge graph on the chassis line, our automated entity resolution pipeline relied on standard cosine similarity over entity descriptions. An extraction chunk from an incident log noted: *"The operator picked up the line 3 torque driver to verify fastener torque."* 
> 
> The resolution module saw high linguistic similarity between this description and two canonical tools: `tq-6012` (the production DC nutrunner) and `tq-6018` (the quality inspection audit wrench). Because both tools were described as "torque drivers on line 3", the system merged them into a single canonical entity.
> 
> Three days later, a line technician queried the assistant: *"What tool do I use to verify shift audit torque on Station 144?"* The assistant responded: *"Use TQ-6012."* 
> 
> The technician used a high-speed production nutrunner instead of a calibrated manual audit wrench to audit already-tightened bolts. This applied secondary torque shocks to twenty-eight finished wheel hubs, stripping fastener threads and forcing an immediate four-hour line stoppage that cost the plant \$180,000 in downtime.
> 
> **The Lesson**: In physical systems, equipment codes with identical prefix families (e.g., `tq-6012` vs `tq-6018`) must NEVER be merged based on semantic similarity. Equipment codes demand strict deterministic conflict guards.

---

### Production Pitfall / War Story: The Stale Undirected Legacy Graph & Silent Ontology Holes

> **War Story 2: The Multi-Relation Overwrite Bug & Unmapped Entity Types**:
> When our team audited the initial knowledge graph store (`data/graph_store/graph.json`), we uncovered two silent architecture bugs that decimated downstream retrieval:
> 
> 1. **The Stale Undirected Graph (`nx.Graph` vs. `nx.MultiDiGraph`)**: The disk store was generated using an older undirected `nx.Graph` schema (`282` noisy nodes, `219` undirected edges). Because simple graphs cannot hold multiple edges between the same pair of nodes, when `SOP-OPR-101` both *required* `tq-6012` and *documented* `tq-6012`, the second relation silently overwrote the first! Worse, directional validation became non-deterministic because edge $(u, v)$ was evaluated in arbitrary dictionary hash order. The solution was implementing `sanitize_graph()` to systematically upgrade stores into `nx.MultiDiGraph`, re-orient valid reverse edges, and deduplicate parallel edges by relation key.
> 
> 2. **The Incomplete Ontology Trap (`16.44%` Violation Rate)**: Our ontology declared `owned_by` and `certified_for` in `NGA_RELATION_TYPES`, but our validation table omitted their `(domain, range)` rules. Every single extraction using these relations was flagged as an `unknown_relation` error! Simultaneously, the extraction parser dropped valid entities because LLM synonyms (`"fault"`, `"sensor"`, `"platform"`, `"role"`, `"tool"`) were not mapped to canonical ontology types (`FaultCode`, `Part`, `Personnel`, `Machine`).
> 
> **The Lesson**: Define explicit domain/range tuples for 100% of declared relations, and build a forgiving alias mapping dictionary (`_ENTITY_TYPE_LOOKUP`) that bridges flexible LLM terminology to strict ontology types before validation.

---

### Code Walkthrough 1: Ontology Schema, Pydantic v2 Models & Resilient JSON Extraction Pipeline

Let us look at the production implementation that enforces these invariants. We define our typed ontology, Pydantic v2 schemas with slug validation and referential integrity checks, and a closed-loop extraction pipeline with an automated multi-attempt repair loop.

```python
"""NGA Manufacturing Assistant — Knowledge Graph Extraction & Ontology Validation.

This module implements:
1. Formal NGA domain ontology definitions (15 Entity Types, 12 Relation Types).
2. Pydantic v2 schemas enforcing lowercase slug IDs, prefix stripping, and bounds.
3. Strict referential integrity and ontology domain/range validation.
4. Schema-enforcing prompt templates and resilient multi-tier JSON repair loop.
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any, Callable, Dict, List, Set, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. DOMAIN ONTOLOGY DEFINITIONS (15 Entity Types, 12 Relation Types)
# ==============================================================================

class EntityType(str, Enum):
    """The 15 canonical entity types in the NGA manufacturing ontology."""
    MACHINE = "Machine"
    STATION = "Station"
    SOP = "SOP"
    FAULT_CODE = "FaultCode"
    DEFECT_CODE = "DefectCode"
    PART = "Part"
    SUPPLIER = "Supplier"
    PERSONNEL = "Personnel"
    TORQUE = "Torque"
    THRESHOLD = "Threshold"
    PROCEDURE = "Procedure"
    NC_RECORD = "NCRecord"
    WORK_ORDER = "WorkOrder"
    RECALL_CRITERIA = "RecallCriteria"
    ESCALATION_LEVEL = "EscalationLevel"


class RelationType(str, Enum):
    """The 12 canonical directed relation types in the NGA manufacturing ontology."""
    CAUSES = "causes"
    INDICATES = "indicates"
    PREVENTS = "prevents"
    MONITORS = "monitors"
    THRESHOLD_FOR = "threshold_for"
    DOCUMENTED_IN = "documented_in"
    ESCALATES_TO = "escalates_to"
    REQUIRES = "requires"
    AUDITED_BY = "audited_by"
    OWNED_BY = "owned_by"
    CERTIFIED_FOR = "certified_for"
    TRIGGERS_RECALL = "triggers_recall"


# Formal Domain/Range Constraints: relation -> (admissible source types, admissible target types)
NGA_ONTOLOGY_RULES: Dict[str, Tuple[Set[str], Set[str]]] = {
    "causes": (
        {"faultcode", "defectcode", "machine", "threshold", "part"},
        {"defectcode", "ncrecord", "machine", "faultcode", "part", "escalationlevel"},
    ),
    "indicates": (
        {"machine", "threshold", "sensor", "station", "faultcode", "defectcode", "part"},
        {"faultcode", "defectcode", "ncrecord", "threshold", "torque", "part"},
    ),
    "prevents": (
        {"sop", "procedure", "part", "machine"},
        {"faultcode", "defectcode", "ncrecord"},
    ),
    "monitors": (
        {"machine", "station", "personnel"},
        {"machine", "threshold", "torque", "station", "part", "faultcode", "defectcode"},
    ),
    "threshold_for": (
        {"sop", "procedure", "machine", "station", "part", "torque", "escalationlevel", "defectcode", "faultcode", "personnel"},
        {"threshold", "torque", "part", "machine", "sop", "procedure"},
    ),
    "documented_in": (
        {
            "threshold", "torque", "machine", "faultcode", "sop", "station",
            "procedure", "defectcode", "ncrecord", "workorder", "part",
            "recallcriteria", "escalationlevel",
        },
        {"sop", "station", "procedure", "workorder", "ncrecord", "recallcriteria", "escalationlevel"},
    ),
    "escalates_to": (
        {"defectcode", "ncrecord", "faultcode", "escalationlevel", "personnel", "sop", "procedure", "threshold"},
        {"escalationlevel", "personnel", "role", "procedure", "sop", "defectcode"},
    ),
    "requires": (
        {"sop", "procedure", "defectcode", "ncrecord", "workorder", "escalationlevel", "recallcriteria", "personnel"},
        {"machine", "procedure", "sop", "personnel", "torque", "part", "station", "threshold", "escalationlevel", "ncrecord"},
    ),
    "audited_by": (
        {"machine", "station", "torque", "workorder", "sop", "procedure", "part", "ncrecord"},
        {"machine", "personnel", "sop", "procedure"},
    ),
    "owned_by": (
        {"sop", "procedure", "escalationlevel", "recallcriteria", "workorder", "machine", "station", "ncrecord"},
        {"personnel", "escalationlevel", "procedure", "sop"},
    ),
    "certified_for": (
        {"personnel", "sop", "procedure"},
        {"sop", "procedure", "machine", "station", "torque", "personnel"},
    ),
    "triggers_recall": (
        {"defectcode", "ncrecord", "faultcode", "threshold", "part"},
        {"recallcriteria", "procedure", "escalationlevel"},
    ),
}

# Redundant prefixes that naive extractors produce and must be stripped from IDs
_REDUNDANT_TYPE_PREFIXES = (
    "machine-", "procedure-", "personnel-", "nc-record-", "part-",
    "workorder-", "fault-code-", "defect-code-", "supplier-",
)

def canonical_slug(text: str) -> str:
    """Normalize identifier text into a lowercase hyphen slug and strip redundant prefixes."""
    if not text:
        return ""
    s = str(text).lower().strip()
    s = re.sub(r"[\s_/]+", "-", s)
    s = re.sub(r"[^a-z0-9.%\-]", "", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    for pref in _REDUNDANT_TYPE_PREFIXES:
        if s.startswith(pref) and len(s) > len(pref):
            s = s[len(pref):]
            break
    return s


# ==============================================================================
# 2. PYDANTIC SCHEMAS WITH RIGOROUS VALIDATORS
# ==============================================================================

class Entity(BaseModel):
    """Extracted knowledge graph node with typed metadata and provenance."""
    id: str = Field(description="Canonical entity identifier (slugified)")
    type: EntityType = Field(description="Typed ontology category")
    name: str = Field(description="Human-readable display name")
    aliases: List[str] = Field(default_factory=list, description="Alternative codes/names")
    description: str = Field(default="", description="Brief technical definition or role")
    level_rank: int = Field(default=1, ge=1, le=4, description="RBAC security clearance rank")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score")
    evidence: str = Field(default="", description="Verbatim quote from source chunk")

    @field_validator("id")
    @classmethod
    def validate_id_slug(cls, v: str) -> str:
        clean = canonical_slug(v)
        if not clean:
            raise ValueError("Entity ID cannot be empty after slugification")
        return clean


class Relation(BaseModel):
    """Extracted knowledge graph directed edge connecting two entities."""
    source: str = Field(description="Source entity ID (slugified)")
    relation: RelationType = Field(description="Directed semantic predicate")
    target: str = Field(description="Target entity ID (slugified)")
    description: str = Field(default="", description="Contextual relationship note")
    level_rank: int = Field(default=1, ge=1, le=4, description="RBAC clearance level")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    evidence: str = Field(default="", description="Verbatim supporting text snippet")

    @field_validator("source", "target")
    @classmethod
    def validate_endpoints(cls, v: str) -> str:
        clean = canonical_slug(v)
        if not clean:
            raise ValueError("Relation endpoint ID cannot be empty")
        return clean


class ExtractionPayload(BaseModel):
    """Complete document extraction payload containing entities and relations."""
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_referential_integrity(self) -> ExtractionPayload:
        """Ensure every relation connects entities defined within the payload."""
        entity_ids = {e.id for e in self.entities}
        dangling_sources = [r.source for r in self.relations if r.source not in entity_ids]
        dangling_targets = [r.target for r in self.relations if r.target not in entity_ids]
        if dangling_sources or dangling_targets:
            raise ValueError(
                f"Referential integrity failure: dangling relation endpoints. "
                f"Missing sources: {dangling_sources}, Missing targets: {dangling_targets}"
            )
        return self


# ==============================================================================
# 3. EXTRACTION PROMPTS WITH ONTOLOGY CONSTRAINTS
# ==============================================================================

EXTRACTION_SYSTEM_PROMPT = """Extract manufacturing knowledge entities and directed relations strictly adhering to the NGA domain ontology.

Entity Types (15):
{entity_types}

Relation Types (12):
{relation_types}

Allowed Direction and Endpoint Constraints (source_type --relation--> target_type):
{ontology_rules}

Rules:
1. Entity IDs MUST be lowercase slugs without redundant prefixes (e.g. 'tq-6012', 'sop-opr-101', 'station-144').
2. Relations are strictly directed: source --relation--> target. Every source and target ID MUST be present in 'entities'.
3. Assign confidence (0.0 to 1.0) based strictly on explicit textual evidence; never speculate.
4. Output MUST be valid JSON only, conforming to this schema:
{{
  "entities": [
    {{"id": "sop-opr-101", "type": "SOP", "name": "SOP-OPR-101 Wheel Lug Nut Tightening", "aliases": ["SOP 101"], "description": "Lug nut fastening procedure", "level_rank": 1, "confidence": 0.95, "evidence": "SOP-OPR-101 requires nutrunner TQ-6012"}}
  ],
  "relations": [
    {{"source": "sop-opr-101", "relation": "requires", "target": "tq-6012", "description": "Fastening equipment requirement", "level_rank": 1, "confidence": 0.95, "evidence": "SOP-OPR-101 requires nutrunner TQ-6012"}}
  ]
}}
Do NOT output markdown commentary or wrapping other than valid JSON.
"""

REPAIR_SYSTEM_PROMPT = """The previous output failed validation.
Error encountered:
{error_message}

Please repair the JSON output to strictly conform to the schema and resolve all validation errors.
Return ONLY the raw JSON object.
"""

# ==============================================================================
# 4. EXTRACTION & REPAIR ENGINE
# ==============================================================================

def format_ontology_rules() -> str:
    """Format ontology domain and range rules into readable lines for LLM prompts."""
    lines = []
    for rel, (sources, targets) in sorted(NGA_ONTOLOGY_RULES.items()):
        s_str = ", ".join(sorted(sources))
        t_str = ", ".join(sorted(targets))
        lines.append(f"- [{s_str}] --{rel}--> [{t_str}]")
    return "\n".join(lines)


def parse_json_safely(raw_text: str) -> dict[str, Any]:
    """Parse raw LLM output into a dictionary, handling code fences and JSON bounds."""
    text = raw_text.strip()
    fence_match = re.search(r"(?:`{3}|~{3})(?:json)?\s*([\s\S]*?)\s*(?:`{3}|~{3})", text)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as err:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise ValueError(f"Unable to parse valid JSON payload: {err}") from err


def validate_ontology_semantics(payload: ExtractionPayload) -> Tuple[ExtractionPayload, List[str]]:
    """Validate extracted relations against NGA ontology directional type rules."""
    entity_map = {e.id: e for e in payload.entities}
    valid_relations: List[Relation] = []
    violations: List[str] = []

    for rel in payload.relations:
        src_ent = entity_map.get(rel.source)
        tgt_ent = entity_map.get(rel.target)
        if not src_ent or not tgt_ent:
            violations.append(f"Dangling edge: {rel.source} --{rel.relation}--> {rel.target}")
            continue

        rel_key = rel.relation.value.lower()
        if rel_key in NGA_ONTOLOGY_RULES:
            valid_srcs, valid_tgts = NGA_ONTOLOGY_RULES[rel_key]
            src_type = src_ent.type.value.lower()
            tgt_type = tgt_ent.type.value.lower()
            if src_type not in valid_srcs or tgt_type not in valid_tgts:
                violations.append(
                    f"Ontology violation: '{src_ent.type.value}' cannot have relation "
                    f"'{rel.relation.value}' to '{tgt_ent.type.value}'."
                )
                continue
        valid_relations.append(rel)

    payload.relations = valid_relations
    return payload, violations


class ExtractionRepairPipeline:
    """Orchestrates LLM extraction with automatic schema repair and ontology validation."""

    def __init__(self, llm_invoke_fn: Callable[[str], str], max_retries: int = 2):
        self.llm_invoke_fn = llm_invoke_fn
        self.max_retries = max_retries

    def extract(self, document_text: str, level_rank: int = 1) -> ExtractionPayload:
        """Extract structured knowledge from text with automated multi-attempt repair."""
        prompt = EXTRACTION_SYSTEM_PROMPT.format(
            entity_types=", ".join(e.value for e in EntityType),
            relation_types=", ".join(r.value for r in RelationType),
            ontology_rules=format_ontology_rules(),
        ) + f"\nDocument to extract:\n{document_text}\n"

        current_prompt = prompt
        last_error = ""

        for attempt in range(self.max_retries + 1):
            raw_response = self.llm_invoke_fn(current_prompt)
            try:
                parsed_dict = parse_json_safely(raw_response)
                # Ensure default level_rank propagates if omitted
                for ent in parsed_dict.get("entities", []):
                    if isinstance(ent, dict):
                        ent.setdefault("level_rank", level_rank)
                for rel in parsed_dict.get("relations", []):
                    if isinstance(rel, dict):
                        rel.setdefault("level_rank", level_rank)

                # Validate Pydantic schema (including referential integrity)
                payload = ExtractionPayload.model_validate(parsed_dict)
                # Validate semantic domain/range ontology constraints
                validated_payload, violations = validate_ontology_semantics(payload)
                if violations:
                    logger.warning("Ontology violations dropped: %s", violations)
                return validated_payload
            except Exception as exc:
                last_error = str(exc)
                logger.warning("Extraction attempt %d failed: %s", attempt + 1, last_error)
                if attempt < self.max_retries:
                    current_prompt = REPAIR_SYSTEM_PROMPT.format(
                        error_message=last_error
                    ) + f"\nPrevious raw response:\n{raw_response}\n"

        raise RuntimeError(
            f"Extraction failed after {self.max_retries + 1} attempts. Last error: {last_error}"
        )
```

---

### Foundation Level Review & Key Takeaways

1. **Cosine similarity measures linguistic proximity, not truth or transitivity**: In multi-hop reasoning, intermediate hops are lost because they lack lexical overlap with the query.
2. **Global queries require graph structures**: Aggregation over hundreds of reports cannot be solved with $k$-NN search; it requires ontology-typed entity clustering.
3. **Subwords are dangerous for hardware identifiers**: High-dimensional attention drowns out 1-character subword differences, requiring deterministic alias indices and conflict guards.
4. **Directionality matters**: Causal and procedural workflows are asymmetric DAGs. Models must enforce directional domain/range constraints.
5. **Schema enforcement must be closed-loop**: Never rely on zero-shot LLM formatting. Wrap extractions in Pydantic v2 schemas with automated JSON repair loops.

Now that we have established our extraction engine and ontology bounds, let us advance to the **Intermediate Level** and examine how to ingest messy documents, resolve entities, construct multigraphs, and partition them into community topologies.


---

# Intermediate Level: Production Ingestion, Normalization & Graph Topology

Welcome back, team. In the Foundation Level, we dissected why naive vector retrieval fails and established our mathematical property multigraph and ontology bounds. Now we enter the engineering trenches: taking raw, messy, unformatted industrial PDF manuals, shift logs, and quality reports, and converting them into a sanitized, canonical, community-clustered NetworkX `MultiDiGraph`.

Let us look at our ingestion architecture on the whiteboard:

```
+-----------------------------------------------------------------------------------------+
|                  DIAGRAM 1: INGESTION & NORMALIZATION PIPELINE                          |
+-----------------------------------------------------------------------------------------+
| [Raw SOP / Spec Documents] (Doc ID, Folder Path -> Level Rank 1..4)                     |
|         │                                                                               |
|         ▼                                                                               |
| [Anchored Windowing] (3,000 chars, 300 char overlap, zero headless fragmenting)         |
|         │                                                                               |
|         ▼                                                                               |
| [LLM Extraction (Structured JSON)] -> entities + relations with confidence scores       |
|         │                                                                               |
|         +---> Confidence < 0.5?  ---[YES]---> [Human-in-the-Loop Review Queue]          |
|         │                                                                               |
|        [NO] (Confidence >= 0.5)                                                         |
|         ▼                                                                               |
| [Normalization & Canonical Indexing]                                                    |
|   * Lowercase slugification: canonical_id()                                             |
|   * Prefix stripping: _REDUNDANT_TYPE_PREFIXES ("machine-tq-6012" -> "tq-6012")          |
|   * Reverse alias lookup: _ALIAS_TO_CANONICAL["TorqMaster 6012"] -> "tq-6012"           |
|   * Equipment Conflict Guard: _has_conflicting_codes("tq-6012", "tq-6018") -> FORBID     |
|         │                                                                               |
|         ▼                                                                               |
| [MultiDiGraph Builder & sanitize_graph]                                                 |
|   * Deduplicate nodes, merge descriptions & union aliases                               |
|   * Edge direction validation & auto-flip per NGA_ONTOLOGY_RULES                         |
|   * Server-side RBAC propagation: max(existing_rank, source_doc_rank)                   |
|         │                                                                               |
|         ▼                                                                               |
| [Louvain Community Detection] -> Undirected projection -> node["community"] attribute   |
+-----------------------------------------------------------------------------------------+
```

### Anchored Chunking, Structured Extraction & Confidence Gating

The first failure point in naive graph ingestion is chunking. Standard naive chunkers split blindly by token count (e.g., 512 tokens with 50-token overlap). When an arbitrary boundary cuts through an engineering table or standard operating procedure, it creates "headless fragments"—passages containing tables of torque numbers or safety codes that lack the subject entity introduced three paragraphs earlier.

In production GraphRAG, we employ **Anchored Windowing**:
1. **Window Size**: 3,000 characters with a 300-character sliding overlap (`WINDOW_CHARS = 3000`, `WINDOW_OVERLAP = 300`).
2. **Context Anchor Injection**: Every chunk header prepends the source document ID, parent section title, and folder-derived RBAC classification level (`level_rank`).
3. **Confidence Score Gating**: The model evaluates each extracted entity and relation, assigning an extraction confidence score $c \in [0.0, 1.0]$.
   - If $c \ge 0.5$: The item proceeds through canonical normalization and enters the active knowledge graph.
   - If $c < 0.5$: The item is filtered out of the graph and routed to the `review_queue` for Human-in-the-Loop (HITL) triage. This prevents low-confidence speculations from poisoning automated reasoning.

---

### Entity Resolution, Canonical Indexing & Equipment Conflict Guards

Once entities are extracted, you will face the **Entity Resolution** crisis. 

In industrial manufacturing, equipment, procedures, and fault codes are referenced under a dozen distinct aliases. For example, the primary DC electric nutrunner on Station 144 is referenced as:
- `"TorqMaster TQ-6012 DC Electric Nutrunner"`
- `"TQ-6012"`
- `"TorqMaster 6012"`
- `"machine-tq-6012"`
- `"TQ6012"`

If these strings enter your graph as separate nodes, your graph degree distribution is destroyed, and multi-hop paths break.

#### 1. Slugification and Prefix Stripping
First, every mention is normalized to a lowercase alphanumeric hyphen-slug (`canonical_id`). Second, redundant type prefixes generated by naive extractors (`machine-`, `procedure-`, `personnel-`, `nc-record-`, `part-`, `workorder-`, `fault-code-`, `defect-code-`, `supplier-`) are stripped:
$$\text{"machine-tq-6012"} \xrightarrow{\text{strip}} \text{"tq-6012"}$$

#### 2. Reverse Canonical Alias Index (`_ALIAS_TO_CANONICAL`)
Rather than running expensive, non-deterministic LLM alignment or heuristic embedding searches on every extracted mention, we maintain a precomputed hash index `_ALIAS_TO_CANONICAL`. We index:
- Canonical ID $\to$ Canonical ID (`"tq-6012"` $\to$ `"tq-6012"`)
- Official Display Name $\to$ Canonical ID (`"torqmaster-tq-6012-dc-electric-nutrunner"` $\to$ `"tq-6012"`)
- Every documented alias $\to$ Canonical ID (`"torqmaster-6012"` $\to$ `"tq-6012"`)

Lookups execute in $O(1)$ time, guaranteeing that all known synonyms converge to the authoritative canonical slug.

#### 3. Equipment Code Conflict Guards (`_has_conflicting_codes`)
Here is where naive ML engineers commit fatal mistakes. They compute embedding similarity between two entity descriptions. If the similarity is above 0.85, they merge the nodes.

In manufacturing, equipment models follow structured code families:
- `TQ-6012`: High-speed DC electric nutrunner (Station 144 assembly)
- `TQ-6018`: Handheld manual digital audit wrench (Quality lab inspection)

Their descriptions are almost identical ("Torque tool for fastener tightening on Line 3"). But merging them is catastrophic. 

Our conflict guard parses alpha-numeric equipment codes via regex:
$$\text{Family Pattern: } \texttt{\textasciicircum([a-z]\{1,4\})-?([0-9]\{2,5\}[a-z]?)\$}$$
If two candidate entities belong to the same prefix family (`"tq"`) but possess distinct numeric identifiers (`"6012"` vs `"6018"`), the function `has_conflicting_codes()` returns `True`, strictly forbidding pairing or merging under any circumstances.

#### 4. The Shortest-Key Merging Trap & Canonical Authority
In early graph implementations, when a cluster of synonyms was merged, the canonical identifier was selected using:
$$\text{key} = \min(\text{cluster}, \text{key}=\lambda n: (\text{len}(n), n))$$
This innocent heuristic produced a catastrophic failure: canonical vehicle platform entity `aurora-au-2025` was merged into `"au25"` simply because `"au25"` had four characters! 
To fix this, `merge_entities()` and `enrich_canonical_entity()` must strictly check membership in `CANONICAL_ENTITY_REGISTRY`. If any member of a merge cluster is in the authoritative registry, its registered canonical ID and official display name take absolute precedence. Furthermore, any alias belonging to a *different* canonical entity is stripped to prevent cross-entity alias contamination.

#### 5. Provenance Coverage Contracts (`sources` / `source_docs`)
A knowledge graph without document provenance is unauditable. In our initial deployment, Atomic Business Fact Coverage scored a dismal **`0.0%`** despite edges being present, because neither nodes nor edges carried structured `sources`.
Every node and edge must maintain:
- `sources`: A structured list of dictionaries: `[{"document": "SOP-OPR-101", "chunk": 4, "evidence": "Torque target 105 Nm"}]`
- `source_docs`: Deduplicated document identifiers enabling instant cross-referencing with dense vector chunks.
- `community`: Modularity community integer assigned via Louvain partition detection, guaranteeing that local BFS search can summarize high-level thematic context.

---

### Code Walkthrough 2: Entity Resolution, Alias Reverse Index & Conflict Guard

Let us examine the exact implementation of our canonical registry, reverse alias index, and prefix conflict guards:

```python
"""NGA Manufacturing Assistant — Canonical Entity Resolution & Conflict Guards.

Implements deterministic slug normalization, reverse alias lookup, redundant prefix
stripping, and equipment code conflict prevention matching production behavior.
"""

from __future__ import annotations

import re
from typing import Any

# Redundant prefixes that naive extractors prepend to entity identifiers
_REDUNDANT_TYPE_PREFIXES = (
    "machine-", "procedure-", "personnel-", "nc-record-",
    "part-", "workorder-", "fault-code-", "defect-code-",
)

# Canonical Registry: canonical_id -> (EntityType, DisplayName, Aliases)
CANONICAL_ENTITY_REGISTRY: dict[str, tuple[str, str, list[str]]] = {
    "tq-6012": (
        "Machine",
        "TorqMaster TQ-6012 DC Electric Nutrunner",
        ["TQ-6012", "TorqMaster 6012", "TQ6012", "machine-tq-6012"],
    ),
    "tq-6018": (
        "Machine",
        "TorqMaster TF-6000 (TQ-6018) Audit Wrench",
        ["TQ-6018", "TorqMaster TF-6000", "TF-6000", "audit wrench TQ-6018", "tq6018"],
    ),
    "sop-opr-101": (
        "SOP",
        "SOP-OPR-101 Wheel Lug Nut Tightening",
        ["SOP-OPR-101", "SOP 101", "Lug nut torque SOP"],
    ),
    "station-144": (
        "Station",
        "Station 144 Chassis Assembly",
        ["Station 144", "Stn 144", "Station-144", "Wheel Assembly Station"],
    ),
    "torque-target-105nm": (
        "Threshold",
        "105 Nm Lug Nut Torque Specification",
        ["105 Nm", "105.0 Nm", "105Nm target", "torque-target-105nm"],
    ),
}

def canonical_id(text: Any) -> str:
    """Normalize text into a lowercase hyphenated slug."""
    if text is None:
        return ""
    slug = re.sub(r"[\s_/]+", "-", str(text).lower().strip())
    slug = re.sub(r"[^a-z0-9.%\-]", "", slug)
    return re.sub(r"-{2,}", "-", slug).strip("-")


# Precompute reverse lookup index from all aliases and display names to canonical slug
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for cid, (_ctype, cname, aliases) in CANONICAL_ENTITY_REGISTRY.items():
    _ALIAS_TO_CANONICAL[cid] = cid
    for label in [cname, *aliases]:
        norm = canonical_id(label)
        if norm:
            _ALIAS_TO_CANONICAL[norm] = cid


def resolve_canonical_id(raw_id: Any, name: Any = None, aliases: list[str] | None = None) -> str:
    """Resolve an entity identifier or alias to its canonical ID, stripping type prefixes."""
    candidates = [raw_id, name, *(aliases or [])]
    for cand in candidates:
        slug = canonical_id(cand)
        if not slug:
            continue
        if slug in _ALIAS_TO_CANONICAL:
            return _ALIAS_TO_CANONICAL[slug]
        for pref in _REDUNDANT_TYPE_PREFIXES:
            if slug.startswith(pref):
                stripped = slug[len(pref):]
                if stripped in _ALIAS_TO_CANONICAL:
                    return _ALIAS_TO_CANONICAL[stripped]

    slug = canonical_id(raw_id) or canonical_id(name)
    for pref in _REDUNDANT_TYPE_PREFIXES:
        if slug.startswith(pref) and len(slug) > len(pref):
            slug = slug[len(pref):]
            break
    return _ALIAS_TO_CANONICAL.get(slug, slug)


def extract_equipment_codes(text: str) -> set[str]:
    """Extract alphanumeric equipment codes (e.g. 'tq-6012', 'sop-opr-101')."""
    if not text:
        return set()
    return {
        m.replace("_", "-")
        for m in re.findall(r"[a-zA-Z]{1,4}[-_]?[0-9]{2,5}[a-zA-Z]?", text.lower())
    }


def has_conflicting_codes(codes_a: set[str], codes_b: set[str]) -> bool:
    """Return True if distinct numeric codes share the same prefix family (e.g. tq-6012 vs tq-6018)."""
    if not codes_a or not codes_b or (codes_a & codes_b):
        return False

    def parse_family(codes: set[str]) -> dict[str, set[str]]:
        fams: dict[str, set[str]] = {}
        for c in codes:
            m = re.match(r"^([a-z]{1,4})-?([0-9]{2,5}[a-z]?)$", c)
            if m:
                fams.setdefault(m.group(1), set()).add(m.group(2))
        return fams

    fams_a = parse_family(codes_a)
    fams_b = parse_family(codes_b)

    for prefix in set(fams_a) & set(fams_b):
        if fams_a[prefix] != fams_b[prefix] and not (fams_a[prefix] & fams_b[prefix]):
            return True  # Conflict: distinct equipment numbers within identical prefix family
    return False
```

---

### Multigraph Structure, Auto-Flipping & Louvain Community Detection

Now let us look at the structure of our property graph on the whiteboard:

```
+-----------------------------------------------------------------------------------------+
|                  DIAGRAM 2: GRAPH SCHEMA / MULTIGRAPH STRUCTURE                         |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  (Node: sop-opr-101)                                      (Node: tq-6012)               |
|  +------------------------+                               +--------------------------+  |
|  | type: "SOP"            |                               | type: "Machine"          |  |
|  | name: "SOP-OPR-101"    |                               | name: "TorqMaster TQ-6012|  |
|  | level_rank: 1          |                               | level_rank: 1            |  |
|  | community: 0           |====== multi-edge: "requires" =======> | community: 0     |  |
|  +------------------------+  key="requires", rank=1, conf=0.95   +--------------------------+  |
|             │                sources=[SOP-OPR-101, chunk 0]                 ▲     ▲     |
|             │                                                               │     │     |
|             │ multi-edge: "threshold_for"                                   │     │     |
|             │ key="threshold_for", rank=1, conf=0.92                        │     │     |
|             ▼                                                               │     │     |
|  (Node: torque-target-105nm)                                                │     │     |
|  +------------------------+                                                 │     │     |
|  | type: "Threshold"      |                                                 │     │     |
|  | name: "105 Nm Target"  |<===== multi-edge: "monitors" ===================+     │     |
|  | level_rank: 1          |       key="monitors", rank=1, conf=0.90               │     |
|  | community: 0           |                                                       │     |
|  +------------------------+                                                       │     |
|             ▲                                                                     │     |
|             │ multi-edge: "threshold_for" (rank=2)                                │     |
|             │ key="threshold_for", rank=2, conf=0.94                              │     |
|             │                                                                     │     |
|  (Node: sop-tec-214)                                                              │     |
|  +------------------------+  multi-edge: "audited_by"                             │     |
|  | type: "SOP"            |  key="audited_by", rank=2, conf=0.91                  │     |
|  | name: "SOP-TEC-214"    |-------------------------------------------------------+     |
|  | level_rank: 2          |                                                             |
|  | community: 1           |  * Node/Edge level_rank <= user_level enforced server-side  |
|  +------------------------+  * Parallel edges preserved via typed relation keys         |
+-----------------------------------------------------------------------------------------+
```

Notice the critical architectural features of this topology:
1. **Multi-Edge Preservation**: `SOP-OPR-101` requires `TQ-6012` (Rank 1). `SOP-TEC-214` audits `TQ-6012` (Rank 2). Both edges coexist between the identical nodes without collisions or data loss.
2. **Edge Auto-Flipping**: When an extractor outputs an inverted relationship (e.g. `(tq-6012) --requires--> (sop-opr-101)`), our builder checks `NGA_ONTOLOGY_RULES`. The rule permits `SOP --requires--> Machine`, but prohibits `Machine --requires--> SOP`. The builder detects that the target and source types are transposed and automatically flips the edge direction `(sop-opr-101) --requires--> (tq-6012)`, healing the graph topology.
3. **Server-Side RBAC Clearance**: Every node and edge is tagged with `level_rank` (1=Operator, 2=Technician, 3=Engineer, 4=Director). If a node appears in both an Operator SOP (`rank=1`) and an Engineering Standard (`rank=3`), the node clearance is calculated as $\max(\text{existing}, \text{new}) = 3$. During retrieval, any entity or edge where `level_rank > user_level` is pruned before traversal begins.
4. **Louvain Hierarchical Community Detection**: To answer high-level thematic questions (*"What are the major functional subsystems in the chassis plant?"*), we project the directed multigraph into a simple undirected graph (`simple = nx.Graph(g.to_undirected())`) and execute the Louvain modularity optimization algorithm. Each node is assigned a discrete `community` integer ID, enabling the retrieval engine to construct community-level thematic summaries.

---

### Code Walkthrough 3: NetworkX MultiDiGraph Construction & Louvain Community Detection

Here is the complete, self-contained Python builder that enforces confidence gating, node deduplication, edge auto-flipping, RBAC hierarchy tracking, and Louvain community partitioning:

```python
"""NGA Manufacturing Assistant — MultiDiGraph Builder & Community Partitioning.

Constructs an attributed NetworkX MultiDiGraph with confidence gating, referential
integrity checks, ontology-directed auto-flipping, RBAC propagation, and Louvain
community clustering.
"""

from __future__ import annotations

from typing import Any
import networkx as nx


def build_and_sanitize_graph(
    extracted_items: list[dict[str, Any]],
    min_confidence: float = 0.5,
    ontology_rules: dict[str, tuple[set[str], set[str]]] | None = None,
) -> tuple[nx.MultiDiGraph, list[dict[str, Any]]]:
    """Build a validated MultiDiGraph with confidence gating, RBAC propagation, and Louvain partitions."""
    g = nx.MultiDiGraph()
    review_queue: list[dict[str, Any]] = []

    for item in extracted_items:
        conf = float(item.get("confidence", 0.0))
        if conf < min_confidence:
            review_queue.append({
                "item": item,
                "reason": f"Confidence {conf:.2f} < {min_confidence}",
            })
            continue

        item_type = item.get("item_type")
        if item_type == "entity":
            eid = item["id"]
            lvl = int(item.get("level_rank", 1))
            desc = str(item.get("description", "")).strip()
            aliases = set(item.get("aliases", []))

            if eid not in g:
                g.add_node(
                    eid,
                    id=eid,
                    type=item.get("type", "Unknown"),
                    name=item.get("name", eid),
                    description=desc,
                    descriptions=[desc] if desc else [],
                    aliases=sorted(aliases),
                    level_rank=lvl,
                    confidence=conf,
                    sources=item.get("sources", []),
                )
            else:
                node = g.nodes[eid]
                if desc and desc not in node.get("descriptions", []):
                    node.setdefault("descriptions", []).append(desc)
                    node["description"] = " | ".join(node["descriptions"])[:600]
                node["aliases"] = sorted(set(node.get("aliases", [])) | aliases)
                node["level_rank"] = max(node.get("level_rank", 1), lvl)
                node["confidence"] = max(node.get("confidence", 0.0), conf)
                node.setdefault("sources", []).extend(item.get("sources", []))

        elif item_type == "relation":
            src, tgt = item["source"], item["target"]
            rel = item.get("relation", "related_to").lower()
            lvl = int(item.get("level_rank", 1))
            desc = str(item.get("description", "")).strip()

            if src not in g or tgt not in g or src == tgt:
                review_queue.append({
                    "item": item,
                    "reason": "Missing or self-loop endpoint",
                })
                continue

            # Validate orientation and auto-flip if inverted per ontology
            if ontology_rules and rel in ontology_rules:
                valid_src, valid_tgt = ontology_rules[rel]
                s_type = str(g.nodes[src].get("type", "")).lower()
                t_type = str(g.nodes[tgt].get("type", "")).lower()
                if s_type not in valid_src or t_type not in valid_tgt:
                    if t_type in valid_src and s_type in valid_tgt:
                        src, tgt = tgt, src  # Auto-repaired inverted relation
                    else:
                        review_queue.append({
                            "item": item,
                            "reason": f"Ontology mismatch: {s_type} -[{rel}]-> {t_type}",
                        })
                        continue

            if g.has_edge(src, tgt, key=rel):
                edge = g.edges[src, tgt, rel]
                if desc and desc not in edge.get("description", ""):
                    edge["description"] = (edge.get("description", "") + " | " + desc).strip(" |")
                edge["level_rank"] = max(edge.get("level_rank", 1), lvl)
                edge["confidence"] = max(edge.get("confidence", 0.0), conf)
                edge.setdefault("sources", []).extend(item.get("sources", []))
            else:
                g.add_edge(
                    src,
                    tgt,
                    key=rel,
                    relation=rel,
                    description=desc,
                    level_rank=lvl,
                    confidence=conf,
                    sources=item.get("sources", []),
                )

    # Louvain hierarchical community detection on simple undirected projection
    if g.number_of_nodes() > 0:
        simple = nx.Graph(g.to_undirected())
        try:
            from community import best_partition
            partition = best_partition(simple)
        except ImportError:
            import networkx.algorithms.community as nx_comm
            communities = list(nx_comm.greedy_modularity_communities(simple))
            partition = {node: i for i, comm in enumerate(communities) for node in comm}

        for nid, comm_id in partition.items():
            g.nodes[nid]["community"] = int(comm_id)

    return g, review_queue
```

---

### Production Pitfall / War Story: The Inverted Escalation Edge

> **War Story from the Quality Department**:
> When processing Non-Conformance reports, technicians often write passively: *"Escalation to Engineering Tier 3 was triggered by Defect Class A Weld Spatter."*
> 
> Without ontology direction validation, a single-pass extraction pipeline generated:
> `(escalation-level-3) --triggers_recall--> (defect-class-a-weld-spatter)`
> 
> When our automated safety monitoring system traversed outgoing edges from `defect-class-a-weld-spatter` to identify required actions, the outgoing edge list was empty because the arrow was reversed. The system concluded that no escalation was necessary. 
> 
> For forty-eight hours, twenty-four vehicles with out-of-spec weld spatter proceeded down the assembly line without supervisor notification until an auditor caught the issue on the final rollout pad.
> 
> **The Lesson**: Never assume LLMs understand edge polarity in complex grammatical sentences. The graph builder MUST check relation domain and range against ontology rules and automatically flip inverted edges.

---

### Intermediate Level Review & Key Takeaways

1. **Anchored chunking preserves context**: Always inject document headers and RBAC levels into chunks to prevent headless fragments.
2. **Reverse alias indices eliminate redundancy**: Resolve synonyms in $O(1)$ time with canonical lookup tables before attempting expensive LLM alignment.
3. **Prefix conflict guards protect hardware**: Entities sharing identical prefix families with distinct numeric suffixes (e.g., `tq-6012` vs `tq-6018`) must never be merged.
4. **MultiDiGraph is mandatory for enterprise schemas**: Entities frequently sustain parallel relations with different clearance levels and provenance.
5. **Ontology rules must heal inversions**: Auto-flip reversed edges at ingestion time to keep downstream graph traversals intact.
6. **Louvain community detection enables thematic summaries**: Project to an undirected graph to partition nodes into cohesive global clusters.

We now have a clean, topologically sound, access-controlled knowledge graph. In the **Advanced Level**, we will explore how to retrieve from this graph alongside dense vectors and BM25 indices, fuse scores using Reciprocal Rank Fusion, distill evidence with System One primitives, and evaluate the graph using quantitative quality indices.
# Advanced Level: Tri-Hybrid Retrieval, Reranking & Quantitative Evaluation

Now that we have constructed a topologically sound, normalized, access-controlled knowledge graph partitioned by Louvain community structures, we face the operational core of GraphRAG: **retrieval, fusion, distillation, and governance**.

Many teams assume that once a graph is loaded into memory or a graph database, retrieval is simple: just query the graph. But in production, isolated graph retrieval is just as flawed as isolated vector retrieval. 

If an engineer asks: *"What is the general philosophy behind our high-voltage battery cell isolation barriers?"*, a graph traversal will flounder unless exact entity seeds are mentioned, whereas a dense vector embedding immediately retrieves the relevant introductory standard. Conversely, if the engineer asks: *"What is the calibrated torque tolerance for Station 144's wheel nutrunner under SOP-OPR-101?"*, dense vectors hallucinate or drift across tool models, whereas graph traversal navigates the exact four-hop path from Station 144 to the 105 Nm torque target. And if the technician enters an alphanumeric equipment code like `TQ-6012` with non-standard spacing or punctuation, an inverted lexical index (BM25) with sub-token decomposition finds the exact match in microseconds where dense vector tokenizers break.

To achieve production reliability across factory floors and engineering organizations, we must deploy a **Tri-Hybrid Retrieval Engine** that fuses three orthogonal modalities:
1. **Dense Vector Embeddings**: Continuous semantic similarity across continuous latent space for fuzzy, thematic, and conceptual questions.
2. **Lexical BM25 Search**: Discrete term-frequency / inverse-document-frequency matching with sub-token decomposition for precise alphanumeric equipment codes, part serials, and SOP numbers.
3. **RBAC-Filtered Local BFS Graph Traversal**: Relational multi-hop path traversal following causal, procedural, and tooling dependencies across entity neighborhoods, bounded by strict role-based access clearance.

Let us map out how these three streams interact and how their disparate scoring distributions are unified.

---

### Whiteboard Diagram 3: Tri-Hybrid Reciprocal Rank Fusion & System One Evidence Distillation

```
========================================================================================================================
 WHITEBOARD DIAGRAM 3: TRI-HYBRID RECIPROCAL RANK FUSION (RRF) & SYSTEM ONE EVIDENCE DISTILLATION FLOW
========================================================================================================================

                                       +------------------------------------------+
                                       |            User Query Input              |
                                       |   e.g. "What torque spec and procedure   |
                                       |       apply to Station-144 on Line 3?"   |
                                       +------------------------------------------+
                                                            |
                                                            v
                                   +----------------------------------------------------+
                                   |         Security & RBAC Context Injection          |
                                   |        User: Line Operator | Clearance: Level 1    |
                                   +----------------------------------------------------+
                                                            |
                     +--------------------------------------+--------------------------------------+
                     |                                      |                                      |
                     v                                      v                                      v
    +----------------------------------+   +----------------------------------+   +----------------------------------+
    |       STREAM 1: DENSE VECTOR     |   |       STREAM 2: LEXICAL BM25     |   |    STREAM 3: GRAPH BFS TRAVERSAL |
    +----------------------------------+   +----------------------------------+   +----------------------------------+
    | 1. Query Embedding (d=1536)      |   | 1. Query Tokenization:           |   | 1. Query Code & Alias Extraction:|
    |    q_emb = embed_query(query)    |   |    ["station", "144", "torque",  |   |    resolve_canonical_id("144")   |
    | 2. Vector ANN Index Search       |   |     "spec", "line", "3"]         |   |    -> hits: {"station-144"}      |
    |    (Cosine Similarity):          |   | 2. Inverted Index Lookup         |   | 2. Multi-Tier Seed Scoring:      |
    |    S_dense = cos(q_emb, doc_emb) |   |    (TF-IDF / Okapi BM25):        |   |    Tier 1 (Canonical Hit): 1.0   |
    | 3. RBAC Chunk Filtering:         |   |    S_bm25 = sum(IDF * TF_norm)   |   |    Tier 2 (Alias Hit):     0.95  |
    |    chunk.level_rank <= 1         |   | 3. RBAC Chunk Filtering:         |   |    Tier 3 (Token Overlap): 0.85  |
    | 4. Top-K Candidates (Ranked):    |   |    chunk.level_rank <= 1         |   | 3. RBAC Seed Filter: level <= 1  |
    |    Rank 1: chunk_402 (0.832)     |   | 4. Top-K Candidates (Ranked):    |   | 4. Bidirectional Local BFS:      |
    |    Rank 2: chunk_101 (0.810)     |   |    Rank 1: chunk_101 (18.42)     |   |    successors | predecessors     |
    |    Rank 3: chunk_505 (0.785)     |   |    Rank 2: chunk_789 (15.10)     |   |    max_hops = 2, level <= 1      |
    |    ...                           |   |    Rank 3: chunk_402 (12.65)     |   | 5. Top-K Graph Subgraph Passages:|
    |    Rank 20: chunk_912 (0.642)    |   |    ...                           |   |    Rank 1: chunk_101 (sop-opr)   |
    |                                  |   |    Rank 20: chunk_314 (4.12)     |   |    Rank 2: chunk_144 (station)   |
    +----------------------------------+   +----------------------------------+   +----------------------------------+
                     |                                      |                                      |
                     |  Rank List 1                         |  Rank List 2                         |  Rank List 3
                     |  pi_dense(d)                         |  pi_bm25(d)                          |  pi_graph(d)
                     +--------------------------------------+--------------------------------------+
                                                            |
                                                            v
                                  +----------------------------------------------------+
                                  |       RECIPROCAL RANK FUSION (RRF) ENGINE          |
                                  |                 k = 60 Smoothing                   |
                                  +----------------------------------------------------+
                                  | Formula:                                           |
                                  |   RRF_Score(d) = sum_m [ w_m / (k + pi_m(d)) ]     |
                                  |                                                    |
                                  | Multi-Modal Scoring Computation:                   |
                                  | - chunk_101: 1/(60+2) + 1/(60+1) + 1/(60+1)        |
                                  |   = 0.01613 + 0.01639 + 0.01639 = 0.04891  [Rank 1]|
                                  | - chunk_402: 1/(60+1) + 1/(60+3) + 0               |
                                  |   = 0.01639 + 0.01587 + 0       = 0.03226  [Rank 2]|
                                  | - chunk_144: 0        + 0        + 1/(60+2)        |
                                  |   = 0       + 0       + 0.01613 = 0.01613  [Rank 3]|
                                  +----------------------------------------------------+
                                                            |
                                                            v Unified Candidate Pool (15-30 Passages)
                                  +----------------------------------------------------+
                                  |     SYSTEM ONE SEMANTIC RERANKING (TYPESAFE JEV)   |
                                  |        Model: jev-1.13.0 | Primitive: Choice       |
                                  +----------------------------------------------------+
                                  | State: User Query + Top Candidates                 |
                                  | Question: "Which candidate directly specifies the  |
                                  |           actionable torque value and procedure?"  |
                                  | Criteria: Candidate criteria descriptions          |
                                  |                                                    |
                                  | Calibrated Output Probabilities & Confidence:      |
                                  | - chunk_101: P = 0.884, Confidence = 0.94          |
                                  | - chunk_144: P = 0.081, Confidence = 0.94          |
                                  | - chunk_402: P = 0.025, Confidence = 0.94          |
                                  |                                                    |
                                  | Action: Prune low-probability candidates (< 0.05)  |
                                  | Output: Distilled to Top 3 High-Precision Passages |
                                  +----------------------------------------------------+
                                                            |
                                                            v Top 3 Verified Passages
                                  +----------------------------------------------------+
                                  |      CONTEXT COMPRESSION & EVIDENCE AUDITING       |
                                  |           (HEADROOM MCP / COMPRESSION)             |
                                  +----------------------------------------------------+
                                  | 1. SHA-256 Digest Calculation:                     |
                                  |    hash = sha256(raw_chunk_text)[:8]               |
                                  |    -> Stored in SessionRawContentStore (In-Memory) |
                                  | 2. Semantic Context Compression:                   |
                                  |    target_ratio = 0.70 | min_tokens = 250          |
                                  |    headroom.compress(payload)                      |
                                  |    -> 42% token reduction, preserving key facts    |
                                  | 3. Verbatim Traceability Marker:                   |
                                  |    [Compressed retrieval | hash=a4f9b2c1 | -184 tok|
                                  | 4. Lossless Audit Seam:                            |
                                  |    headroom_retrieve(hash="a4f9b2c1") -> Raw Text  |
                                  +----------------------------------------------------+
                                                            |
                                                            v Clean Synthesized Context Block
                                  +----------------------------------------------------+
                                  |          GENERATIVE REASONING AGENT (LLM)          |
                                  |    Context: RBAC-Cleared, Graph-Grounded Evidence  |
                                  |    Output: Deterministic Answer + Valid Citation:  |
                                  |      "Per SOP-OPR-101 (Station-144), target torque |
                                  |       is 105.0 Nm +/- 2.0 Nm using TQ-6012."       |
                                  +----------------------------------------------------+
========================================================================================================================
```

---

### The Mechanics of Reciprocal Rank Fusion (RRF)

Why do we need Reciprocal Rank Fusion instead of a weighted linear combination?

In naive systems, engineers frequently attempt linear score fusion:
$$S_{\text{hybrid}}(d) = \alpha \cdot S_{\text{dense}}(d) + \beta \cdot S_{\text{bm25}}(d) + \gamma \cdot S_{\text{graph}}(d)$$

This formula looks intuitive on a slide, but in production software, it collapses because the underlying score distributions are mathematically incompatible:
1. **Dense Vector Cosine Similarity**: $S_{\text{dense}} \in [-1.0, 1.0]$. Due to embedding space anisotropy (the cone effect), real-world document embeddings are typically clustered tightly between $0.65$ and $0.85$. The dynamic range is tiny ($\Delta \approx 0.20$).
2. **Lexical BM25**: $S_{\text{bm25}} \in [0, +\infty)$. BM25 scores are unbounded. For a long query containing rare terms, a document might score $28.5$. For a short query with common terms, the top document might score $3.2$.
3. **Graph Traversal Proximity**: $S_{\text{graph}} \in [0.0, 1.0]$ or discrete hop counts $\{0, 1, 2\}$.

If you attempt min-max normalization:
$$\tilde{S}(d) = \frac{S(d) - S_{\min}}{S_{\max} - S_{\min}}$$
a single statistical outlier with a massive BM25 score compresses all other BM25 scores toward zero, completely wiping out the lexical signal across the rest of the candidate pool. Furthermore, whenever the underlying index expands, maximum scores shift, requiring constant recalibration of $\alpha, \beta, \gamma$.

#### The Invariant Power of Ordinal Ranks
Reciprocal Rank Fusion (Cormack, Clarke, and Büttcher, SIGIR 2009) solves this by discarding arbitrary score magnitudes entirely. It converts each retrieval stream into a sorted permutation $\pi_m: \mathcal{D} \to \{1, 2, \dots, N\}$ and scores candidates based solely on their 1-based ordinal rank:

$$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{w_m}{k + \pi_m(d)}$$

where:
- $M = \{\text{Dense}, \text{BM25}, \text{Graph}\}$ is the set of retrieval modalities.
- $\pi_m(d)$ is the 1-based rank position of document $d$ in modality $m$.
- $k \ge 1$ is a rank-smoothing constant (standard literature default: $k = 60$).
- $w_m \ge 0$ is the modality weight (defaulting to uniform $1.0$).

#### Why Does $k = 60$ Work?
The smoothing parameter $k$ acts as a **low-pass filter on rank variance**:
- If $k = 0$, the top-ranked item receives $\frac{1}{1} = 1.0$, while rank 20 receives $\frac{1}{20} = 0.05$. The ratio is $20:1$. A single modality placing an irrelevant passage at Rank 1 due to keyword stuffing would completely overpower a highly relevant passage ranked #2 across all three modalities.
- With $k = 60$:
  - Rank 1 receives $\frac{1}{60 + 1} = \frac{1}{61} \approx 0.01639$
  - Rank 2 receives $\frac{1}{60 + 2} = \frac{1}{62} \approx 0.01613$
  - Rank 20 receives $\frac{1}{60 + 20} = \frac{1}{80} \approx 0.01250$
  - The ratio between Rank 1 and Rank 20 is only $\frac{80}{61} \approx 1.31$.

Because the curve decays gently, a candidate cannot dominate merely by topping one stream. Instead, **multi-stream consensus dominates**. 

Let us look at our whiteboard example:
- `chunk_101` was ranked #2 in Dense ($\frac{1}{62}$), #1 in BM25 ($\frac{1}{61}$), and #1 in Graph ($\frac{1}{61}$). Its fused score is:
  $$\text{RRF}(\text{chunk\_101}) = 0.01613 + 0.01639 + 0.01639 = \mathbf{0.04891}$$
- `chunk_402` was ranked #1 in Dense ($\frac{1}{61}$), #3 in BM25 ($\frac{1}{63}$), but was missing from the Graph traversal ($r = \infty$). Its fused score is:
  $$\text{RRF}(\text{chunk\_402}) = 0.01639 + 0.01587 + 0 = \mathbf{0.03226}$$

Even though `chunk_402` took first place in vector search, `chunk_101` decisively claims Rank 1 overall because both the lexical engine and the relational knowledge graph confirmed its relevance. That is robustness in action.

---

> ### Production Pitfall / War Story 2: The RRF Constant Catastrophe ($k=0$ vs $k=60$)
>
> During our Phase 2 pilot at the transmission plant, an engineer decided that $k=60$ was "arbitrary academic baggage" and simplified the fusion function to $k=0$ (pure reciprocal rank $\frac{1}{r}$) to "make the best matches stand out more."
>
> Two days later, a maintenance technician submitted a query containing a slang typo: *"nutrunner torque drift fault 44"*. 
>
> In the BM25 index, a legacy 2017 draft memo contained the single sentence: *"Nutrunner torque drift fault 44 recorded on test bench."* Because the phrase matched four rare words identically, BM25 assigned it Rank 1. Under $k=0$, that single Rank 1 hit contributed a score of $1.0$.
>
> Meanwhile, the true governing standard—`SOP-OPR-101`—was ranked #2 in Dense ($0.50$), #2 in BM25 ($0.50$), and #1 in Graph ($1.0$). Its score was $2.0$. But an unrelated safety notice happened to score Rank 1 in Dense, yielding $1.0 + 0 = 1.0$.
>
> Worse, for queries where the target chunk ranked #3 across all streams, its score under $k=0$ was $\frac{1}{3} + \frac{1}{3} + \frac{1}{3} = 1.0$. It tied exactly with a completely irrelevant document that happened to hit Rank 1 in one stream due to keyword stuffing! The agent presented the obsolete 2017 draft memo as authoritative, recommending an outdated torque tolerance that had been superseded three years prior.
>
> **The Lesson**: Never set $k=0$. The constant $k=60$ is mathematically required to dampen outlier variance from single modalities and reward true multi-modal agreement.

---

### Code Walkthrough 4: RBAC-Filtered Local BFS Graph Traversal

Let us examine the production implementation of our local graph search engine, matching `src/nga/graphrag/search.py`. 

This module executes seed discovery via canonical alias indexing, handles bidirectional multi-hop BFS traversal, enforces role-based clearance at every hop, and provides a graceful fallback to pure lexical graph scoring if dense vector embedding services fail:

```python
"""RBAC-Filtered Local BFS Knowledge Graph Traversal Engine.

Provides multi-hop evidence extraction over NetworkX MultiDiGraph instances
with strict Role-Based Access Control (RBAC) filtering and graceful
lexical fallback when embeddings are absent.
"""

from __future__ import annotations

import re
from typing import Any
import networkx as nx
import numpy as np


def normalize_id(text: str) -> str:
    """Generate a clean alphanumeric slug for canonical matching."""
    if not text:
        return ""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(text).strip().lower()).strip("-")
    return slug


# Conversational and syntactic stopwords that cause false-positive lexical collisions
_QUERY_STOPWORDS = {
    "what", "which", "where", "when", "who", "whom", "whose", "why", "how",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "to", "at", "in", "for",
    "on", "by", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "from", "up", "down",
    "in", "out", "over", "under", "again", "further", "then", "once",
    "here", "there", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "can", "will", "just", "should", "now",
    "used", "uses", "use", "using", "under", "per", "and", "or", "if",
}


def tokenize_query(text: str, filter_stopwords: bool = True) -> set[str]:
    """Extract alphanumeric tokens and decompose hyphenated identifiers.
    
    Preserves equipment codes (e.g. 'tq-6012' -> {'tq-6012', 'tq', '6012'}).
    Filters conversational stopwords to prevent query noise collisions.
    """
    if not text:
        return set()
    cleaned = re.sub(r"[^a-zA-Z0-9%\.\s\-]", " ", text.lower())
    tokens: set[str] = set()
    for word in cleaned.split():
        clean_word = word.strip("-")
        if len(clean_word) > 1:
            if filter_stopwords and clean_word in _QUERY_STOPWORDS and not any(c.isdigit() for c in clean_word):
                continue
            tokens.add(clean_word)
            for part in clean_word.split("-"):
                if len(part) > 1:
                    if filter_stopwords and part in _QUERY_STOPWORDS and not any(c.isdigit() for c in part):
                        continue
                    tokens.add(part)
    return tokens


def compute_lexical_node_score(
    node_id: str,
    node_data: dict[str, Any],
    query_lower: str,
    query_slug: str,
    query_tokens: set[str],
    canonical_hits: set[str],
) -> float:
    """Compute calibrated lexical and alias matching score in [0.05, 1.0]."""
    nid_str = str(node_id).lower()
    nid_slug = normalize_id(nid_str)

    # Tier 1: Exact canonical hit or query slug substring
    if nid_str in canonical_hits or (len(nid_slug) >= 3 and nid_slug in query_slug):
        return 1.0

    name = str(node_data.get("name", "")).lower()
    aliases = [str(a).lower() for a in (node_data.get("aliases") or []) if a]

    # Tier 2: Verbatim name or alias hit in query text
    for label in [name, *aliases]:
        if not label:
            continue
        label_slug = normalize_id(label)
        if (len(label) >= 3 and label in query_lower) or (
            len(label_slug) >= 3 and label_slug in query_slug
        ):
            return 0.95

    # Tier 3: Jaccard-scaled token overlap across node text, aliases and source evidence
    if not query_tokens:
        return 0.05

    description = str(node_data.get("description", "")).lower()
    evidence_snippets = [
        str(s.get("evidence", "")).lower()
        for s in (node_data.get("sources") or [])
        if isinstance(s, dict)
    ]
    haystack = " ".join([nid_str, name, *aliases, description, *evidence_snippets])
    node_tokens = tokenize_query(haystack, filter_stopwords=False)

    if not node_tokens:
        return 0.05

    overlap = len(query_tokens & node_tokens)
    if overlap == 0:
        return 0.05

    # Scale overlap: 1 overlap -> ~0.22, 4 overlaps -> ~0.60, max 0.85
    denom = max(min(len(query_tokens), 6), 1)
    return min(0.85, 0.10 + 0.75 * (overlap / denom))


def extract_doc_bridge_signals(
    graph: nx.MultiDiGraph,
    retrieved_docs: list[dict[str, Any]] | None,
) -> dict[str, float]:
    """Extract graph node activation signals from dense vector retrieval chunks.
    
    Bridges conceptual and symptom matches from vector search (e.g. 'wheel separation')
    into graph seed nodes by inspecting retrieved doc_ids and chunk_ids.
    """
    if not retrieved_docs:
        return {}

    bridge_scores: dict[str, float] = {}
    for rank_idx, doc in enumerate(retrieved_docs[:5]):
        doc_id = str(doc.get("doc_id") or "").lower()
        chunk_id = str(doc.get("chunk_id") or "").lower()
        # Rank-discounted activation boost in [0.45, 0.90]
        boost = 0.90 / (1.0 + 0.20 * rank_idx)

        for nid, ndata in graph.nodes(data=True):
            nid_str = str(nid).lower()
            sources = ndata.get("sources") or []
            source_docs = [str(d).lower() for d in (ndata.get("source_docs") or [])]

            matched = False
            if doc_id and (doc_id in source_docs or doc_id in nid_str):
                matched = True
            else:
                for s in sources:
                    if isinstance(s, dict):
                        s_doc = str(s.get("document", "")).lower()
                        s_chunk = str(s.get("chunk", "")).lower()
                        if (doc_id and doc_id in s_doc) or (chunk_id and chunk_id in s_chunk):
                            matched = True
                            break
            if matched:
                bridge_scores[nid_str] = max(bridge_scores.get(nid_str, 0.0), boost)

    return bridge_scores


def rbac_local_bfs_traversal(
    graph: nx.MultiDiGraph,
    query: str,
    embeddings_provider: Any | None = None,
    *,
    retrieved_docs: list[dict[str, Any]] | None = None,
    k_seeds: int = 5,
    max_hops: int = 2,
    user_level: int = 1,
) -> dict[str, Any]:
    """Execute RBAC-filtered local BFS graph traversal with Vector-to-Graph bridge.

    Args:
        graph: NetworkX MultiDiGraph representing the domain knowledge base.
        query: User prompt or diagnostic question.
        embeddings_provider: Optional embedding client with `.embed_query(str)`.
        retrieved_docs: Optional top dense vector passages from ChromaDB search.
        k_seeds: Number of highest-scoring seed entities to initiate BFS from.
        max_hops: Maximum traversal radius around seed nodes.
        user_level: Security clearance rank (1=operator, 4=director).

    Returns:
        Structured dictionary containing RBAC-cleared entities, relations,
        and Louvain community thematic summaries.
    """
    if graph is None or graph.number_of_nodes() == 0:
        return {"entities": [], "relations": [], "community_summaries": []}

    query_lower = (query or "").lower().strip()
    query_slug = normalize_id(query_lower)
    query_tokens = tokenize_query(query, filter_stopwords=True)

    # Resolve equipment codes and pattern identifiers
    code_matches = re.findall(r"[a-zA-Z]{1,4}[-_ ]?[0-9]{2,5}[a-zA-Z]?", query_lower)
    canonical_hits: set[str] = set()
    for candidate in [*code_matches, *query_tokens]:
        cand_slug = normalize_id(candidate)
        if cand_slug in graph:
            canonical_hits.add(cand_slug)

    # Vector-to-Graph Bridge: Extract activation signals from vector search docs
    bridge_scores = extract_doc_bridge_signals(graph, retrieved_docs)

    # Generate query embedding if provider is functional
    query_embedding = None
    if embeddings_provider is not None:
        try:
            query_embedding = embeddings_provider.embed_query(query)
        except Exception:
            query_embedding = None

    # Step 1: Score all candidate nodes permitted under user_level
    scored_candidates: list[tuple[str, float]] = []
    score_map: dict[str, float] = {}

    for node_id, node_data in graph.nodes(data=True):
        node_rank = int(node_data.get("level_rank", 1))
        if node_rank > user_level:
            continue  # RBAC security filter

        nid_str = str(node_id)
        lex_score = compute_lexical_node_score(
            nid_str, node_data, query_lower, query_slug, query_tokens, canonical_hits
        )
        bridge_score = bridge_scores.get(nid_str.lower(), 0.0)

        node_emb = node_data.get("embedding")
        if node_emb is None or query_embedding is None:
            # Fallback combining lexical score and vector bridge signal
            combined = max(lex_score, bridge_score)
            scored_candidates.append((nid_str, combined))
            score_map[nid_str] = combined
        else:
            # Upper-bound hybrid combination across embeddings, lexical, and bridge
            q_vec = np.array(query_embedding, dtype=np.float32)
            n_vec = np.array(node_emb, dtype=np.float32)
            denom = (np.linalg.norm(q_vec) * np.linalg.norm(n_vec)) + 1e-9
            cos_sim = float(np.dot(q_vec, n_vec) / denom)
            combined = max(cos_sim, lex_score, bridge_score)
            scored_candidates.append((nid_str, combined))
            score_map[nid_str] = combined

    scored_candidates.sort(key=lambda item: item[1], reverse=True)
    seeds = [nid for nid, _ in scored_candidates[:k_seeds]]

    # Step 2: Bidirectional BFS Expansion
    visited: set[str] = set()
    frontier = list(seeds)
    current_hop = 0

    while frontier and current_hop < max_hops:
        next_frontier: list[str] = []
        for current_node in frontier:
            if current_node in visited:
                continue
            visited.add(current_node)

            # Bidirectional neighbors: successors (outgoing) + predecessors (incoming)
            if graph.is_directed():
                neighbors = set(graph.successors(current_node)) | set(
                    graph.predecessors(current_node)
                )
            else:
                neighbors = set(graph.neighbors(current_node))

            # Sort candidate neighbors by seed score to maintain deterministic traversal
            sorted_neighbors = sorted(
                neighbors,
                key=lambda nb: score_map.get(str(nb), 0.0),
                reverse=True,
            )

            for neighbor in sorted_neighbors:
                neighbor_str = str(neighbor)
                if neighbor_str in visited:
                    continue
                neighbor_data = graph.nodes[neighbor]
                neighbor_rank = int(neighbor_data.get("level_rank", 1))
                if neighbor_rank <= user_level:
                    next_frontier.append(neighbor_str)

        frontier = next_frontier
        current_hop += 1

    # Step 3: Format Extracted Subgraph Elements
    ordered_visited = [n for n in seeds if n in visited] + [
        n for n in visited if n not in seeds
    ]

    extracted_entities: list[dict[str, Any]] = []
    communities_grouped: dict[int, list[str]] = {}

    for nid in ordered_visited:
        data = dict(graph.nodes[nid])
        data.pop("embedding", None)  # Omit dense vector arrays from payload
        extracted_entities.append({"id": nid, **data})

        comm_id = data.get("community")
        if isinstance(comm_id, int):
            display_name = data.get("name") or nid
            communities_grouped.setdefault(comm_id, []).append(str(display_name))

    extracted_relations: list[dict[str, Any]] = []
    for u, v, k, edata in graph.edges(keys=True, data=True):
        u_str, v_str = str(u), str(v)
        if u_str in visited and v_str in visited:
            edge_rank = int(edata.get("level_rank", 1))
            if edge_rank <= user_level:
                extracted_relations.append({
                    "source": u_str,
                    "target": v_str,
                    "key": k,
                    "relation": edata.get("relation", "related"),
                    "description": edata.get("description", ""),
                    "level_rank": edge_rank,
                    "confidence": float(edata.get("confidence", 1.0)),
                })

    # Step 4: Synthesize Community Thematic Summaries
    community_summaries = [
        f"Community {cid}: {', '.join(members[:6])}"
        for cid, members in sorted(communities_grouped.items())
        if len(members) >= 2
    ]

    return {
        "entities": extracted_entities[:20],
        "relations": extracted_relations[:30],
        "community_summaries": community_summaries[:5],
    }
```

Notice four critical engineering choices in this implementation:
1. **Bidirectional Expansion (`_neighbors`)**: In NetworkX, calling `graph.neighbors(n)` on a directed graph returns only successors. But in manufacturing, if the relationship was extracted as `(sop-opr-101, specifies_torque, torque-target-105nm)` and your seed is `torque-target-105nm`, a successor-only search stops dead! Bidirectional traversal (`set(graph.successors) | set(graph.predecessors)`) traverses both upstream causes and downstream effects regardless of polarity.
2. **Graceful Fallback**: If an embedding model times out, returns HTTP 504, or is omitted in an air-gapped factory environment, `rbac_local_bfs_traversal` does not throw an unhandled exception. It falls back smoothly to lexical and canonical alias matching.
3. **The Vector-to-Graph Bridge Pattern**: In real plants, operators ask symptom queries without equipment codes (*"One customer reports wheel separation at highway speed"*). Pure lexical graph search is completely blind here because `"QCR-501"` and `"Class A"` do not appear in the text. The Vector-to-Graph bridge inspects the dense vector chunks retrieved by ChromaDB, extracts their document and chunk IDs, and injects rank-discounted activation scores directly into the graph seed nodes.
4. **Deterministic Neighbor Sorting**: Expanding neighbors in raw Python `set` order causes non-deterministic output due to randomized hash seeds across Python processes. Sorting next-frontier nodes by their computed activation score (`score_map.get(str(nb), 0.0)`) guarantees reproducible, deterministic traversal paths.

---

### Production Pitfall / War Story: The 3 Search Engine Bugs That Destroyed Retrieval Recall

#### War Story 3: The Constant 0.1 Seed-Score Bug (Issue 6)
In our initial production audit, `ContextualRecallMetric` averaged an abysmal **`5.4%`** across all 85 benchmark questions. When we traced the execution of `build_graph_evidence()`, we discovered an unbelievable bug:
To save disk space, our graph serializer popped `embedding` arrays (`data.pop("embedding", None)`). In `search.py`, the fallback branch had:

```python
# The fatal fallback bug:
if node_emb is None:
    scored.append((node_id, 0.1))  # Flat 0.1 score for all nodes!
```

Every single node in the graph received an identical score of `0.1`! Python's stable sort (`sort(reverse=True)`) preserved insertion order, returning the **exact same five arbitrary nodes** (`['wo-2025-0417', 'weld-robot-07', 'station-42', 'fault-code-e-4012', 'sop-tec-201']`) for every single query in the benchmark!
Replacing this with `compute_lexical_node_score()` immediately activated true entity and alias discovery.

#### War Story 4: Query Stopword Noise & The Wrong Audit Wrench (`R4` Case Study, Issue 7)
Benchmark question `R4` asked: *"What tool is used for the torque audit?"* (Expected: `TQ-6018` TorqMaster TF-6000, calibrated weekly).
Instead of retrieving `tq-6018`, our lexical scorer ranked `tq-6019` as the top seed! Why?
Node `tq-6019`'s description contained the words *"tool"*, *"used"*, *"is"*, and *"torque"*. Because our tokenizer did not filter English conversational stopwords, it accumulated 4 token matches against the query string. `TQ-6018`, titled *"TorqMaster TF-6000 Audit Wrench"*, used the word *"wrench"* rather than *"tool"*, losing the token match race!
DeepEval's safety gate clamped the final grounded score to `0.20` because an uncertified tool appeared in the context. Introducing `_QUERY_STOPWORDS` filtering completely eliminated stopword pollution, promoting `tq-6018` to the top seed and boosting Contextual Recall from `0.023` to **`0.967`**.

#### War Story 5: Pure Symptom Queries & The Vector-to-Graph Bridge (`E1` Case Study, Issue 8)
In question `E1`: *"One customer reports wheel separation at highway speed. No other reports."*
The prompt mentions neither `QCR-501` (Quality Recall Standard) nor `ESC-402` (Level 4 Escalation). Pure graph search matched only the single token *"wheel"* on `station-144`, missing all recall and containment protocols.
By implementing `extract_doc_bridge_signals()`, dense vector search first retrieved the `QCR-501` and `ESC-402` documentation chunks. Those chunk signals were passed into the graph search, which automatically activated `qcr-501` and `class-a-defect` as high-priority seeds, expanding 1 hop to `esc-402` and `fap-401`.
The result on `E1`: Contextual Recall jumped **21x** (`0.044` $\to$ **`0.927`**), Context Faithfulness jumped **7.5x** (`0.109` $\to$ **`0.817`**), and `JevEval` jumped from `0.307` to **`0.949`**.

---

---

### System One Semantic Reranking with TypeSafe Jev Choice

Once the Tri-Hybrid engine evaluates Dense Vector search, BM25 keyword search, and the RBAC Graph traversal, Reciprocal Rank Fusion produces a unified candidate pool of 15 to 30 candidate passages.

What do most naive architectures do next? They dump all 30 passages directly into the LLM prompt.

This introduces three fatal production defects:
1. **The "Lost in the Middle" Effect**: LLM self-attention mechanisms heavily attend to tokens at the very beginning and very end of long contexts. Placing critical torque tolerances in chunk 14 of 30 leads to frequent retrieval hallucinations or omission.
2. **Context Window Inflation & TTFT Latency**: Ingesting 30 uncompressed chunks consumes 10,000 to 15,000 tokens. Generating the first token takes 2 to 4 seconds, and token billing explodes by an order of magnitude.
3. **Cross-Encoder Hosting Overhead**: Using a traditional BERT/RoBERTa cross-encoder (`bge-reranker-large`) requires maintaining dedicated GPU infrastructure, PyTorch runtimes, and dealing with cold-start latency.

#### The System One Primitive Paradigm
In cognitive science (Kahneman), **System One** represents fast, intuitive, subconscious pattern recognition, while **System Two** represents slow, deliberate, autoregressive reasoning.

Traditional LLMs are System Two engines: they generate tokens sequentially, one after another, costing time and compute. But reranking does not require generating paragraphs of text. Reranking requires a **structured judgment**: *"Given this state and these candidates, what is the probability distribution of relevance?"*

TypeSafe AI's **Jev** model (`jev-1.13.0`) is purpose-built as an agentic System One primitive. It evaluates typed application state against discrete options and returns normalized softmax probabilities and confidence scores in under 120 milliseconds.

Using the `Choice` primitive:
```python
from typesafe_sdk import Choice, TypeSafeClient

client.system_one(
    state={"query": query_text, "candidates": candidate_pool},
    questions={
        "evidence_selection": Choice(
            instructions="Which candidate passage directly provides the technical standard, torque specification, or procedural step required to answer the query?",
            criteria={
                "candidate_0": "Specifies SOP-OPR-101 torque calibration and tolerance parameters.",
                "candidate_1": "Describes general safety gear and PPE requirements.",
                "none_of_the_above": "None of the candidates contain direct actionable specifications.",
            },
        )
    },
)
```

Jev evaluates the entire option set in parallel and emits a calibrated probability distribution:
```json
{
  "model": "jev-1.13.0",
  "answers": {
    "evidence_selection": {
      "type": "choice",
      "choice": "candidate_0",
      "confidence": 0.942,
      "probabilities": {
        "candidate_0": 0.885,
        "candidate_1": 0.082,
        "none_of_the_above": 0.033
      }
    }
  }
}
```

By filtering candidates where $P(\text{candidate}_i) \ge \tau$ (e.g. $\tau = 0.15$), we distill the 30-passage candidate pool down to the 3 to 5 highest-precision evidence passages before any generative LLM is invoked.

---

### Code Walkthrough 5: Tri-Hybrid Retrieval Fusion with Reciprocal Rank Fusion (RRF) & TypeSafe Jev Choice Semantic Reranking

Here is the complete implementation of our fusion and distillation pipeline, integrating RRF with the TypeSafe AI System One `Choice` primitive:

```python
"""Tri-Hybrid Retrieval Engine with Reciprocal Rank Fusion (RRF) and TypeSafe Distillation.

Combines Dense Embeddings, BM25 Lexical Match, and Graph Subgraph Traversal into
a unified candidate pool, distilled down to 3-5 high-precision evidence passages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedPassage:
    """Represents a unified evidence passage from any retrieval stream."""
    passage_id: str
    doc_id: str
    text: str
    level_rank: int
    metadata: dict[str, Any] = field(default_factory=dict)
    rrf_score: float = 0.0
    stream_ranks: dict[str, int] = field(default_factory=dict)


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[RetrievedPassage]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[RetrievedPassage]:
    """Execute Reciprocal Rank Fusion across multiple retrieval modalities.

    Formula:
        RRF_Score(d) = sum_m [ w_m / (k + rank_m(d)) ]

    Args:
        ranked_lists: Map of modality name -> ordered list of RetrievedPassage.
        k: Smoothing constant preventing outlier rank domination (default: 60).
        weights: Optional importance weights per modality (default: 1.0 each).

    Returns:
        Consolidated, deduplicated list of passages sorted by RRF_Score descending.
    """
    weights = weights or {mod: 1.0 for mod in ranked_lists}
    passage_registry: dict[str, RetrievedPassage] = {}
    rrf_scores: dict[str, float] = {}

    for modality, passages in ranked_lists.items():
        w_m = weights.get(modality, 1.0)
        for rank_0, passage in enumerate(passages):
            rank_1 = rank_0 + 1  # 1-based ordinal rank
            pid = passage.passage_id

            if pid not in passage_registry:
                passage_registry[pid] = passage

            # Record rank per stream for provenance and explainability
            passage_registry[pid].stream_ranks[modality] = rank_1

            # Accumulate reciprocal rank score
            reciprocal_value = w_m / (k + rank_1)
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + reciprocal_value

    # Assign final RRF scores
    fused_passages: list[RetrievedPassage] = []
    for pid, score in rrf_scores.items():
        passage = passage_registry[pid]
        passage.rrf_score = round(score, 6)
        fused_passages.append(passage)

    fused_passages.sort(key=lambda p: p.rrf_score, reverse=True)
    return fused_passages


class SystemOneSemanticDistiller:
    """Uses TypeSafe AI Jev System One Choice primitive to distill candidates."""

    def __init__(self, typesafe_client: Any | None = None, confidence_threshold: float = 0.15):
        self.client = typesafe_client
        self.threshold = confidence_threshold

    def distill(
        self,
        query: str,
        candidates: list[RetrievedPassage],
        target_count: int = 5,
    ) -> list[RetrievedPassage]:
        """Distill candidate pool down to top-N high-precision passages.

        If TypeSafe client is not configured, gracefully falls back to top RRF order.
        """
        if not candidates:
            return []
        if self.client is None or len(candidates) <= target_count:
            return candidates[:target_count]

        # Construct discrete criteria mapping for top candidates
        eval_pool = candidates[: min(len(candidates), 12)]
        criteria: dict[str, str] = {}
        for idx, passage in enumerate(eval_pool):
            key = f"candidate_{idx}"
            snippet = (passage.text[:120] + "...").replace("\n", " ")
            criteria[key] = f"Document {passage.doc_id}: {snippet}"
        criteria["none_of_the_above"] = "None of the candidates contain direct actionable specifications."

        try:
            from typesafe_sdk import Choice

            response = self.client.system_one(
                state={"query": query, "candidate_count": len(eval_pool)},
                questions={
                    "evidence_relevance": Choice(
                        instructions=(
                            "Which candidate passage provides the most direct, authoritative, "
                            "and actionable procedural specifications to answer the query?"
                        ),
                        criteria=criteria,
                    )
                },
            )

            answer = response.answers.get("evidence_relevance")
            probabilities = getattr(answer, "probabilities", {})

            # Rescore and sort candidates by System One probability
            rescored: list[tuple[RetrievedPassage, float]] = []
            for idx, passage in enumerate(eval_pool):
                prob = float(probabilities.get(f"candidate_{idx}", 0.0))
                rescored.append((passage, prob))

            rescored.sort(key=lambda item: item[1], reverse=True)
            distilled = [
                passage for passage, prob in rescored if prob >= self.threshold
            ]

            # If threshold pruned too aggressively, retain top matches
            if len(distilled) < 2:
                distilled = [p for p, _ in rescored[:target_count]]

            return distilled[:target_count]
        except Exception:
            # Fallback cleanly to RRF order on API error
            return candidates[:target_count]
```

With our retrieval, fusion, and semantic distillation pipeline in place, we now turn to the critical discipline of **evidence auditing and quantitative graph governance**.

---

### Context Compression & Verbatim Lossless Evidence Verification

In regulated enterprise environments—governed by automotive quality standards such as IATF 16949, ISO 9001, or aerospace AS9100—an automated agent cannot merely present a synthesized recommendation. Every single numerical threshold, calibration procedure, and escalation pathway must be back-referenced to an authoritative, verifiable source document.

At the same time, enterprise conversations are multi-turn workflows. A line technician diagnosing a persistent weld fault may exchange six or seven turns with the agent:
1. Reporting the defect code and station number.
2. Checking recent maintenance logs.
3. Querying torque transducer calibrations.
4. Reviewing ultrasonic inspection readings.
5. Formulating an 8D root-cause corrective action.

If the agent injects uncompressed, multi-paragraph document chunks into the context window at every turn, the context history rapidly explodes past 30,000 tokens. This causes severe latency spikes, attention distraction, and ballooning inference costs.

To solve this, our production architecture implements a **Dual-Layer Evidence Storage & Compression Engine** (`src/enterprise_agent/compression/manager.py`):
1. **Semantic Compression Layer**: Large retrieval payloads and tool outputs exceeding 250 tokens are compressed by 30% to 50% using Headroom compression (`target_ratio=0.7`). Redundant formatting, generic boilerplate, and repetitive narrative phrasing are stripped, preserving the distilled technical facts.
2. **Lossless Content-Addressed Verification Seam**: Before any compression occurs, the raw, byte-for-byte text payload is hashed using SHA-256:
   $$\text{hash} = \text{SHA-256}(\text{raw\_content})[:8]$$
   The uncompressed text is registered in an in-memory, thread-safe LRU store (`SessionRawContentStore`). The compressed text is then injected into the prompt prefixed with a cryptographic traceability header:
   ```text
   [Compressed document_retrieval | hash=a4f9b2c1 | saved 342 tokens]
   SOP-OPR-101 specifies target torque 105.0 Nm +/- 2.0 Nm for Station-144 using TQ-6012 nutrunner.
   ```
3. **On-Demand Verbatim Recovery**: When an auditor, compliance officer, or human-in-the-loop (HITL) supervisor clicks "Verify Source Citation", the system does not ask the LLM to reproduce the source. The application executes `headroom_retrieve(hash="a4f9b2c1")` or calls `store.get(session_id, "a4f9b2c1")`, returning the original, unedited engineering document with zero hallucination drift in $O(1)$ time.

---

> ### Production Pitfall / War Story 3: Context Amnesia & Aerospace AS9100 Audit Rejection
>
> In our aerospace customer deployment, our agent assisted assembly technicians in setting turbine blade shroud fastener torques. To conserve context window space during lengthy diagnostic sessions, an early engineer applied a lossy LLM summarizer that compressed incoming technical bulletins down to two-sentence bullet points before injecting them into the agent's scratchpad.
>
> The LLM summarizer read:
> *"Torque bolts to 85 Nm in cross-pattern. For titanium-coated grade 5 fasteners, apply anti-seize lubricant and reduce torque by 15% (72.2 Nm)."*
>
> The summarizer compressed this to:
> *"Fasteners require 85 Nm torque in cross-pattern with anti-seize lubricant."*
>
> It completely dropped the 15% torque reduction clause!
>
> The assembly agent instructed the technician to torque titanium-coated bolts to 85 Nm with anti-seize. Under actual assembly, the over-torqued titanium fasteners experienced micro-fractures in the bolt threads. During a routine AS9100 quality audit, the lead auditor demanded to see the exact text citation justifying the 85 Nm recommendation with lubricant. 
>
> When the engineering team looked at the agent transcript, the raw source text was gone—overwritten by the lossy summary. The plant was issued a Major Non-Conformance finding, halting shroud assembly for three weeks.
>
> **The Lesson**: Never discard raw source text. Context compression must be strictly paired with content-addressed, cryptographic provenance (`SessionRawContentStore` with SHA-256 digest headers). The LLM reasons over the compressed representation, but the system must be capable of retrieving the verbatim, uncompressed source passage on demand.

---

### The Quantitative Graph Quality Index (GQI) Evaluation Framework

How do you know if your knowledge graph is production-ready? 

Most software teams rely on vanity metrics: *"We have 15,000 nodes and 45,000 edges!"* Or they rely on manual spot checks: *"I asked about Station 144 and it looked fine."*

In high-consequence industrial engineering, vanity metrics are dangerous. A graph with 50,000 nodes where 30% of the edges violate domain ontology rules (e.g. connecting tools to defect codes via causal links) will poison every downstream RAG prompt.

Our platform defines and enforces the **Graph Quality Index (GQI)** framework (`src/nga/evaluation/graph_eval.py`). The GQI evaluates four orthogonal, quantitative dimensions against a validated gold benchmark:

```
+----------------------------------------------------------------------------------------------------+
|                               GRAPH QUALITY INDEX (GQI) FRAMEWORK                                  |
+-----------------------------+---------+--------------------+---------------------------------------+
| Metric                      | Weight  | Target Benchmark   | Operational Definition                |
+-----------------------------+---------+--------------------+---------------------------------------+
| 1. Entity Extraction F1     |  30%    | F1 >= 0.90         | Precision & Recall against gold nodes |
| 2. Relation Accuracy        |  25%    | Acc >= 0.85        | Ontology schema compliance rate       |
| 3. Entity Alignment Rate    |  25%    | Rate >= 0.95       | Canonical synonym resolution rate     |
| 4. Knowledge Coverage Rate  |  20%    | Rate >= 0.90       | Gold atomic fact path reachability    |
+-----------------------------+---------+--------------------+---------------------------------------+
| Composite GQI Target:  GQI = 0.30*M1 + 0.25*M2 + 0.25*M3 + 0.20*M4 >= 0.90                         |
+----------------------------------------------------------------------------------------------------+
```

Let us dissect each metric mathematically:

#### 1. Entity Extraction Accuracy ($F_1 \ge 0.90$)
Measures whether the ingestion pipeline extracts valid domain entities while rejecting conversational noise:
$$\text{Precision} = \frac{TP}{TP + FP_{\text{negatives}}}, \quad \text{Recall} = \frac{TP}{|V_{\text{gold}}|}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall} + 10^{-9}}$$
Entities are matched through canonical slugs and alias indexes (`build_alias_index`). Hallucinated or non-existent equipment codes explicitly listed in `gold_negative_entities` are heavily penalized as false positives.

#### 2. Relation Extraction Accuracy ($\text{Acc}_{\text{rel}} \ge 0.85$)
Measures whether extracted relationships conform strictly to the domain ontology:
$$\text{Relation Accuracy} = \frac{|E_{\text{valid}}|}{|E_{\text{total}}|}$$
Every edge $(u, \text{rel}, v)$ is evaluated against `check_relation_compliance(rel, u_type, v_type)`. If an extractor asserts that an `SOP` *causes* a `DefectCode` (violating domain semantics where SOPs *mitigate* defects or *specify* procedures), the edge is flagged as an ontology violation.

#### 3. Entity Alignment Success Rate ($\text{Rate}_{\text{align}} \ge 0.95$)
Measures whether known domain synonyms converge into unified canonical entities:
$$\text{Alignment Success Rate} = \frac{\sum_{(c, a) \in \text{Pairs}} \mathbb{I}(\text{resolve}(a) == \text{resolve}(c))}{|\text{Pairs}|}$$
If `TQ-6012`, `Station 144 Nutrunner`, and `tq6012` remain as three separate, disconnected nodes in the graph, graph traversals fragment and fail. This metric verifies that all known aliases resolve to the single canonical entity.

#### 4. Atomic Fact Knowledge Coverage Rate ($\text{Rate}_{\text{cov}} \ge 0.90$)
Measures whether atomic engineering propositions $(S, P, O)$ from governing documents are physically traversable in the graph:
$$\text{Knowledge Coverage Rate} = \frac{\sum_{f \in \text{Facts}} \mathbb{I}(S_f \in V \land O_f \in V \land \text{dist}_{\text{undirected}}(S_f, O_f) \le 3)}{|\text{Facts}|}$$
A fact is covered if both subject and object exist in the graph and are connected via an undirected path within $\le 3$ hops.

---

### Code Walkthrough 6: Graph Quality Index (GQI) Evaluation Metric Calculation

Here is the complete, self-contained Python implementation of the GQI evaluation engine, matching `src/nga/evaluation/graph_eval.py`:

```python
"""Graph Quality Index (GQI) Quantitative Evaluation Framework.

Calculates the Four Quantitative Graph Quality Metrics:
1. Entity Extraction F1
2. Relation Extraction Accuracy (Ontology Compliance)
3. Entity Alignment Success Rate (Synonym Resolution)
4. Atomic Fact Knowledge Coverage Rate (Graph Reachability)
"""

from __future__ import annotations

import re
from typing import Any
import networkx as nx


def canonical_slug(text: str) -> str:
    """Normalize text into an entity ID slug."""
    if not text:
        return ""
    return re.sub(r"[^a-zA-Z0-9]+", "-", str(text).strip().lower()).strip("-")


def build_alias_index(graph: nx.MultiDiGraph) -> dict[str, Any]:
    """Build reverse lookup index mapping canonical slugs of names/aliases to nodes."""
    index: dict[str, Any] = {}
    if graph is None:
        return index

    for node, data in graph.nodes(data=True):
        labels = [data.get("name", ""), *(data.get("aliases", []) or [])]
        for label in labels:
            key = canonical_slug(str(label))
            if key:
                index.setdefault(key, node)

    # Node IDs take absolute precedence
    for node in graph.nodes:
        key = canonical_slug(str(node))
        if key:
            index[key] = node

    return index


def check_relation_compliance(
    rel_type: str,
    source_type: str,
    target_type: str,
    ontology_rules: dict[str, set[tuple[str, str]]],
) -> str | None:
    """Verify whether (source_type, relation, target_type) is ontology compliant."""
    rel = rel_type.lower().strip()
    src = source_type.lower().strip()
    tgt = target_type.lower().strip()

    if rel not in ontology_rules:
        return f"Unknown relation type: '{rel}'"

    valid_pairs = ontology_rules[rel]
    if (src, tgt) not in valid_pairs:
        return f"Prohibited relation: ({src}) --[{rel}]--> ({tgt})"

    return None


def calculate_graph_quality_index(
    graph: nx.MultiDiGraph,
    benchmark_payload: dict[str, Any],
    ontology_rules: dict[str, set[tuple[str, str]]],
) -> dict[str, Any]:
    """Calculate the 4 quantitative metrics and composite Graph Quality Index (GQI).

    Formula:
        GQI = 0.30 * Entity_F1 + 0.25 * Relation_Acc + 0.25 * Alignment_Rate + 0.20 * Coverage_Rate
    """
    alias_index = build_alias_index(graph)
    total_nodes = graph.number_of_nodes() if graph else 0
    total_edges = graph.number_of_edges() if graph else 0

    # ── Metric 1: Entity Extraction F1 ────────────────────────────────────────
    gold_entities = benchmark_payload.get("gold_entities", [])
    gold_negatives = benchmark_payload.get("gold_negative_entities", [])
    tp_entities = 0
    matched_nodes: set[Any] = set()

    for gold in gold_entities:
        gid = canonical_slug(gold.get("id", ""))
        gname = canonical_slug(gold.get("name", ""))
        resolved = alias_index.get(gid) or alias_index.get(gname)
        if resolved is not None and graph.has_node(resolved):
            tp_entities += 1
            matched_nodes.add(resolved)

    negative_hits = sum(
        1 for neg in gold_negatives
        if canonical_slug(neg.get("id", "")) and graph.has_node(canonical_slug(neg.get("id", "")))
    )

    precision = round(len(matched_nodes) / max(len(matched_nodes) + negative_hits, 1), 4)
    recall = round(tp_entities / max(len(gold_entities), 1), 4)
    entity_f1 = round((2 * precision * recall) / max(precision + recall, 1e-9), 4)

    # ── Metric 2: Relation Extraction Accuracy ───────────────────────────────
    valid_edges = 0
    violations: list[dict[str, Any]] = []

    for u, v, k, data in graph.edges(keys=True, data=True):
        rel = str(data.get("relation", "related")).lower().strip()
        u_type = str(graph.nodes[u].get("type", "")).lower()
        v_type = str(graph.nodes[v].get("type", "")).lower()

        violation_reason = check_relation_compliance(rel, u_type, v_type, ontology_rules)
        if violation_reason is None:
            valid_edges += 1
        else:
            violations.append({"edge": f"({u})--[{rel}]-->({v})", "reason": violation_reason})

    relation_accuracy = round(valid_edges / max(total_edges, 1), 4)

    # ── Metric 3: Entity Alignment Success Rate ──────────────────────────────
    gold_synonyms = benchmark_payload.get("gold_synonyms", [])
    total_alias_pairs = 0
    successful_pairs = 0

    for syn in gold_synonyms:
        canonical_node = alias_index.get(canonical_slug(syn["canonical_id"]))
        for alias in syn.get("aliases", []):
            total_alias_pairs += 1
            resolved_node = alias_index.get(canonical_slug(alias))
            if canonical_node is not None and resolved_node == canonical_node:
                successful_pairs += 1

    alignment_success_rate = round(successful_pairs / max(total_alias_pairs, 1), 4)

    # ── Metric 4: Atomic Fact Knowledge Coverage Rate ────────────────────────
    gold_facts = benchmark_payload.get("gold_atomic_facts", [])
    undirected_view = graph.to_undirected(as_view=True) if graph else None
    covered_facts = 0

    for fact in gold_facts:
        sub_node = alias_index.get(canonical_slug(fact.get("subject", "")))
        obj_node = alias_index.get(canonical_slug(fact.get("object", "")))

        if sub_node is not None and obj_node is not None and undirected_view is not None:
            if sub_node == obj_node:
                covered_facts += 1
            else:
                try:
                    if nx.shortest_path_length(undirected_view, sub_node, obj_node) <= 3:
                        covered_facts += 1
                except nx.NetworkXNoPath:
                    pass

    knowledge_coverage_rate = round(covered_facts / max(len(gold_facts), 1), 4)

    # ── Composite Graph Quality Index (GQI) ──────────────────────────────────
    gqi = round(
        0.30 * entity_f1
        + 0.25 * relation_accuracy
        + 0.25 * alignment_success_rate
        + 0.20 * knowledge_coverage_rate,
        4,
    )

    targets = {
        "entity_extraction_f1": 0.90,
        "relation_accuracy": 0.85,
        "alignment_success_rate": 0.95,
        "knowledge_coverage_rate": 0.90,
        "graph_quality_index": 0.90,
    }

    status = {
        "entity_f1_pass": entity_f1 >= targets["entity_extraction_f1"],
        "relation_acc_pass": relation_accuracy >= targets["relation_accuracy"],
        "alignment_pass": alignment_success_rate >= targets["alignment_success_rate"],
        "coverage_pass": knowledge_coverage_rate >= targets["knowledge_coverage_rate"],
        "overall_pass": gqi >= targets["graph_quality_index"],
    }

    return {
        "node_count": total_nodes,
        "edge_count": total_edges,
        "graph_quality_index": gqi,
        "status": status,
        "targets": targets,
        "metrics": {
            "entity_f1": entity_f1,
            "entity_precision": precision,
            "entity_recall": recall,
            "relation_accuracy": relation_accuracy,
            "alignment_success_rate": alignment_success_rate,
            "knowledge_coverage_rate": knowledge_coverage_rate,
        },
        "violations_sample": violations[:5],
    }
```

---

### Empirical Production Benchmark: From 47.07% to 100.0% GQI

To demonstrate the real-world impact of these architectural patterns, let us examine the empirical benchmark data from our Next-Gen Automation (NGA) plant deployment.

When our engineering team initially evaluated our baseline Knowledge Graph (`data/graph_store/graph.json`) against the gold benchmark (`eval-graph/gold_graph_benchmark.json`), the system failed catastrophically across every pillar. Overall GQI was **`0.4707` (47.07%)**, and DeepEval Contextual Recall was stuck at **`5.4%`**.

By systematically diagnosing and fixing the **8 root-cause issues** discussed throughout this Masterclass—without altering a single line of the evaluation benchmark—we achieved a clean sweep across all metrics:

| Benchmark / Evaluation Dimension | Before Fixes (Baseline) | After Fixes (Final) | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Overall Graph Quality Index (GQI)** | `0.4707` (47.07%) ❌ | **`1.0000` (100.0%)** ✅ | **`+52.93%`** |
| **1. Entity Extraction F1** | `69.57%` (`16/23` recall) | **`100.0%` (`23/23` recall, `0` false pos)** | **`+30.43%`** |
| **2. Relation Extraction & Ontology Compliance** | `16.67%` recall / `83.56%` valid | **`100.0%` recall (`12/12`) / `100.0%` valid (`0` violations)** | **`+83.33%` recall** |
| **3. Entity Alignment (Synonym Resolution)** | `65.71%` (`23/35`) | **`100.0%` (`35/35`)** | **`+34.29%`** |
| **4. Atomic Business Fact Coverage** | `0.00%` (`0/8` facts, `0%` provenance) | **`100.0%` (`8/8` facts, `100%` provenance)** | **`+100.0%`** |
| **DeepEval `ContextualRecallMetric` (`jev-1.13.0`)** | `0.0543` (5.4%) | **`0.5541` (55.4%)** | **10.2x higher (`+50.0%`)** |
| **DeepEval `ContextualPrecisionMetric` (`jev-1.13.0`)** | `0.3036` (30.4%) | **`0.4426` (44.3%)** | **+45.8% (`+13.9%`)** |
| **DeepEval `FaithfulnessMetric` (`jev-1.13.0`)** | `0.1971` (19.7%) | **`0.5391` (53.9%)** | **2.7x higher (`+34.2%`)** |
| **DeepEval `Manufacturing Quality JevEval` (Mean)** | `0.4696` (47.0%) | **`0.9203` (92.0%)** | **`+45.1%`** |
| **DeepEval `JevEval` Pass Rate (`>= 0.70`)** | `3 / 16` (**18.8%**) | **`16 / 16` (100.0%)** | **`+81.2%` (5.3x pass rate)** |
| **Standalone `evaluate_with_jev` Rubric Score** | **`1.40 / 5.0`** (`28.0%`) | **`4.94 / 5.0`** (`98.8%`, `16/16` at `5/5`) | **`+3.54 / 5.0` (`+70.8%`)** |

Notice the causal chain:
1. Elevating graph quality (GQI from 47% to 100%) and repairing the seed scoring bug increased the relevant facts delivered to the LLM.
2. Contextual Recall jumped **10.2x** (from 5.4% to 55.4%).
3. Because the model no longer had to hallucinate missing specifications, Faithfulness surged **2.7x** (from 19.7% to 53.9%).
4. The DeepEval industrial safety pass rate skyrocketed from **18.8% to a perfect 100.0%**.

---

### Evaluation Trap: Static Candidate Judge vs. Live Graph Context Evaluation

During our benchmarking, we uncovered a deceptive trap that leads many AI teams to overestimate the quality of their RAG pipelines:

1. **The Static Judge Trap (`side_by_side_eval.py`)**:
   Many RAG benchmark frameworks evaluate candidates by calling `to_deep_eval_test_case(question_data=row, actual_output=candidate, tool_outputs=None)`.
   When `tool_outputs` is omitted, the harness falls back to a synthetic context string: `f"[{source_docs}] {expected_answer}"`.
   **This never queries the knowledge graph or vector store!** It tests whether the judge can verify an answer when handed the perfect cheat sheet, telling you nothing about whether your retrieval pipeline actually works in production.
2. **Live Graph Context Evaluation (`tool_outputs` with `graph_evidence`)**:
   Real evaluation must execute the full retrieval pipeline and pass real `tool_outputs` containing `graph_evidence`. The adapter injects:
   - `[GRAPH ENTITIES]`: Matched entities, types, and descriptions.
   - `[GRAPH RELATIONS]`: Multi-hop relational edges discovered during BFS traversal.
   - `[GRAPH COMMUNITY SUMMARY]`: Louvain community summaries providing high-level thematic context.
   Evaluating with live payloads using TypeSafe Jev (`jev-1.13.0`) provides calibrated, non-generative metrics with zero token generation cost and sub-50ms latency.

---

### Production Checklist: The 4 Golden Rules for GraphRAG Updates

Before promoting any knowledge graph build or search algorithm update to production, enforce this four-point release gate:

1. **Always run `sanitize_graph()` after extraction**: Ensure the saved graph is an `nx.MultiDiGraph` with zero ontology violations, valid edge directions, populated `sources` and `source_docs`, and modularity `community` IDs.
2. **Guard candidate pairing with `_has_conflicting_codes()`**: Strictly forbid pairing or merging distinct equipment or SOP codes sharing a prefix family (`tq-6012` vs `tq-6018`, `sop-opr-101` vs `sop-opr-158`).
3. **Bridge vector hits into graph search (`extract_doc_bridge_signals`)**: Ensure that semantic vector hits for symptom queries automatically activate their corresponding knowledge graph neighborhoods.
4. **Verify both GQI (`1.0000`) and live RAG metrics**: Run both `graph_eval.py` and DeepEval with live `tool_outputs`, verifying that Contextual Recall, Precision, and Faithfulness pass all target thresholds.

---

---

### Interactive Graph Observability: Ego-Network API & Node-Link Canvas

Even with automated GQI checks, production systems require human-in-the-loop observability. When an edge validation alert fires or an engineer investigates a weird retrieval result, they need to visually inspect the graph topology.

However, rendering a complete 50,000-node graph in a browser canvas freezes the UI thread and creates an unreadable "hairball".

Our platform architecture (`src/enterprise_agent/ui/server.py` and `src/enterprise_agent/ui/static/app.js`) solves this through **Contextual Ego-Network Projections**:

#### 1. FastAPI Ego-Network Topology Endpoint (`/api/graph/topology`)
The backend projects a localized sub-network centered on the user's query or inspection target:
- **RBAC Filtering**: Filters out any node or relation where `level_rank > user_level`.
- **Ego-Network Discovery**: Seeds expansion from query terms or equipment slugs, running BFS outward up to `max_hops` (default 2).
- **Hub Preservation**: Truncates candidate nodes by NetworkX degree (`deg(u)`), ensuring structural hubs remain visible.
- **Live Violation Flagging**: Every edge is evaluated against `check_relation`. Valid edges are marked `valid: true`; invalid edges are marked `valid: false` with the verbatim ontology error string.

```python
@app.get("/api/graph/topology")
def get_graph_topology(
    user_level: int = Query(4, ge=1, le=4, description="RBAC access level (1-4)"),
    entity_type: str | None = Query(None, description="Optional entity type filter"),
    query: str | None = Query(None, description="Optional search query for ego-network filtering"),
    max_hops: int = Query(2, ge=1, le=4, description="BFS hop radius when query is provided"),
    max_nodes: int = Query(150, ge=10, le=500, description="Max nodes to return"),
) -> dict[str, Any]:
    """Interactive Knowledge Graph Topology endpoint for node-link canvas exploration."""
    from nga.ingestion.build_graph import load_graph
    from nga.ingestion.normalize import canonical_id, check_relation

    graph = ctx.graph_data or load_graph("data/graph_store")
    if graph is None or graph.number_of_nodes() == 0:
        return {"nodes": [], "edges": [], "entity_types": [], "relation_types": [], "total_nodes": 0, "total_edges": 0}

    degrees = dict(graph.degree())
    candidate_nodes: set[str] = {
        str(nid) for nid, ndata in graph.nodes(data=True)
        if int(ndata.get("level_rank", 1)) <= user_level
    }

    # Query-seeded ego-network expansion
    if query and query.strip():
        q_lower = query.strip().lower()
        q_slug = canonical_id(q_lower)
        seeds = {
            nid for nid in candidate_nodes
            if q_lower in str(graph.nodes[nid].get("name", "")).lower()
            or q_slug in canonical_id(str(nid))
        }
        if seeds:
            visited = set(seeds)
            frontier = list(seeds)
            for _ in range(max_hops):
                next_f = []
                for cur in frontier:
                    nbrs = set(graph.successors(cur)) | set(graph.predecessors(cur)) if graph.is_directed() else set(graph.neighbors(cur))
                    for nb in nbrs:
                        nb_s = str(nb)
                        if nb_s in candidate_nodes and nb_s not in visited:
                            visited.add(nb_s)
                            next_f.append(nb_s)
                frontier = next_f
            candidate_nodes = visited

    # Degree-prioritized truncation to preserve network hubs
    sorted_nodes = sorted(candidate_nodes, key=lambda n: (degrees.get(n, 0), str(n)), reverse=True)[:max_nodes]
    selected_set = set(sorted_nodes)

    nodes_out = [
        {
            "id": str(nid),
            "name": graph.nodes[nid].get("name") or str(nid),
            "type": graph.nodes[nid].get("type") or "Unknown",
            "level_rank": int(graph.nodes[nid].get("level_rank", 1)),
            "degree": int(degrees.get(nid, 0)),
            "community": int(graph.nodes[nid].get("community", 0)),
            "sources": (graph.nodes[nid].get("sources") or [])[:3],
        }
        for nid in sorted_nodes
    ]

    edges_out = []
    for u, v, edata in graph.edges(data=True):
        u_s, v_s = str(u), str(v)
        if u_s in selected_set and v_s in selected_set:
            if int(edata.get("level_rank", 1)) <= user_level:
                rel = str(edata.get("relation") or "related")
                u_type = str(graph.nodes[u].get("type", "")).lower()
                v_type = str(graph.nodes[v].get("type", "")).lower()
                violation = check_relation(rel, u_type, v_type)
                edges_out.append({
                    "source": u_s,
                    "target": v_s,
                    "relation": rel,
                    "level_rank": int(edata.get("level_rank", 1)),
                    "valid": violation is None,
                    "violation": violation,
                })

    return {"nodes": nodes_out, "edges": edges_out, "total_nodes": graph.number_of_nodes(), "total_edges": graph.number_of_edges()}
```

#### 2. HTML5 Canvas Force Physics Simulation (`app.js`)
On the frontend, client-side rendering uses an iterative physical particle simulation with electrostatic repulsion, spring tension, and center-mass gravity:
- **Coulomb Repulsion**: Pairs of nodes repel inversely to distance squared ($F_{\text{repel}} = \frac{k}{d^2}$, $k=1600$), preventing overlaps.
- **Hooke's Spring Tension**: Connected edges pull nodes toward equilibrium rest length ($L_0 = 95\text{px}$, $k_s = 0.035$).
- **Center Gravity & Damping**: Damps velocity by $0.78$ per frame to smoothly stabilize the layout.
- **Visual Alerting**: Valid edges render as crisp directed vectors; invalid edges render as pulsing red dashed lines (`rgba(239, 68, 68, 0.75)`), enabling engineers to spot extraction hallucinations instantly.

```javascript
function stepForcePhysics(nodes, edges, width, height) {
  const kRepel = 1600;    // Coulomb electrostatic repulsion constant
  const kSpring = 0.035;   // Hooke's elastic spring stiffness constant
  const restLen = 95;      // Equilibrium edge rest length (pixels)
  const cx = width / 2;
  const cy = height / 2;

  // 1. N^2 All-Pairs Electrostatic Repulsion
  for (let i = 0; i < nodes.length; i++) {
    const a = nodes[i];
    for (let j = i + 1; j < nodes.length; j++) {
      const b = nodes[j];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const distSq = dx * dx + dy * dy + 1;
      if (distSq > 90000) continue; // Spatial cutoff: 300px radius
      const dist = Math.sqrt(distSq);
      const force = kRepel / distSq;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      a.vx -= fx;
      a.vy -= fy;
      b.vx += fx;
      b.vy += fy;
    }
  }

  // 2. Spring Contraction / Expansion Along Edges
  for (const e of edges) {
    const s = e.sourceNode;
    const t = e.targetNode;
    const dx = t.x - s.x;
    const dy = t.y - s.y;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
    const disp = dist - restLen;
    const fx = (dx / dist) * disp * kSpring;
    const fy = (dy / dist) * disp * kSpring;
    s.vx += fx;
    s.vy += fy;
    t.vx -= fx;
    t.vy -= fy;
  }

  // 3. Center Gravity & Velocity Damping
  for (const n of nodes) {
    if (n === kgSimState.draggingNode) continue;
    n.vx += (cx - n.x) * 0.006;
    n.vy += (cy - n.y) * 0.006;
    n.vx *= 0.78; // Friction factor (damping)
    n.vy *= 0.78;
    n.x += Math.max(-12, Math.min(12, n.vx));
    n.y += Math.max(-12, Math.min(12, n.vy));
  }
}
```

---

### Masterclass Conclusion & Final Review Summary

Over the course of this Masterclass, we have taken you from the mathematical breakdown of naive vector search to an industrial-strength, enterprise-grade GraphRAG architecture.

Let us review the core architectural pillars you must carry back to your engineering teams:

```
========================================================================================================================
                                     PRODUCTION GRAPHRAG ARCHITECTURE BLUEPRINT
========================================================================================================================

 1. FOUNDATION LEVEL (INGESTION & ONTOLOGY)
    - Naive Vector Breakdown: Blind to multi-hop transitivity, helpless at corpus-wide aggregations, prone to
      alphanumeric subword collisions, and indifferent to causal directionality.
    - Closed-Loop Property Graph Schema: Strict Pydantic v2 schemas defining discrete entity types, relation types,
      confidence gating (>= 0.5), and RBAC clearance metadata (Levels 1-4).
    - Source Chunk Provenance: Every node and edge retains document chunk origin, section header, and byte offsets.

 2. INTERMEDIATE LEVEL (NORMALIZATION & TOPOLOGY)
    - Anchored Chunking: Inject document titles and section context into every chunk to prevent headless fragments.
    - Deterministic Entity Resolution: Canonical slug normalization, reverse alias lookup indexing, and prefix
      conflict guards preventing cross-tool collisions (e.g., TQ-6012 vs TQ-6018).
    - MultiDiGraph Modeling: NetworkX MultiDiGraph preserving parallel edges with distinct provenance and security levels.
    - Automatic Inversion Healing: Ontology-driven domain/range checks that auto-flip inverted triplet edges.
    - Louvain Community Detection: Undirected graph projections generating hierarchical clusters and thematic summaries.

 3. ADVANCED LEVEL (RETRIEVAL, FUSION & GOVERNANCE)
    - Tri-Hybrid Retrieval: Fusing Dense Vector embeddings (conceptual), BM25 lexical search (alphanumeric codes),
      and RBAC-filtered local BFS graph traversal (relational paths) with graceful offline fallback.
    - Reciprocal Rank Fusion (RRF): Score-scale invariant fusion with k=60 dampening single-stream noise and
      rewarding multi-modal consensus.
    - System One Semantic Distillation: TypeSafe AI Jev (jev-1.13.0) Choice primitives distilling 30 candidate passages
      down to 3-5 high-precision evidence blocks in < 120ms without generative token overhead.
    - Lossless Verbatim Traceability: Headroom semantic context compression paired with 8-character SHA-256 digests
      enabling O(1) verbatim text recovery for AS9100/ISO 9001 compliance audits.
    - Quantitative Graph Quality Index (GQI): Four-metric evaluation framework (Entity F1, Relation Accuracy,
      Entity Alignment, Knowledge Coverage) guaranteeing GQI >= 0.90 before production release.
    - Interactive Observability: FastAPI /api/graph/topology ego-networks rendered on an HTML5 physics canvas with
      real-time ontology violation alerting.
========================================================================================================================
```

Build your graphs with strict schemas. Fuse your retrieval streams with Reciprocal Rank Fusion. Distill your evidence with fast System One judgment. Verify your claims with cryptographic provenance. And evaluate your knowledge base with quantitative rigor.

Class dismissed. Go build your systems.
