from __future__ import annotations

import math
import re
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any, Union, Callable
from pathlib import Path
from enum import Enum

import numpy as np
from rapidfuzz import fuzz

# Import data types

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cleaners'))

from data_cleaningv2 import (
    RegionLine,
    RegionBlock,
    ClusterLine,
    Rectangle,
    Contour,
    DataCleaningInputsV2,
    CandidateAnchor,
    BBox,
)

# Import bootstrap Module B types

try:
    from bootstrap.NER import Entity, EntityType, EntityCollection
    from bootstrap.proto_entities import ResolvedEntity, generate_proto_entities
    from bootstrap.entity_merge import merge_entities, create_module_b_output
    from bootstrap.config.config import load_rule_pack, RulePack
    BOOTSTRAP_AVAILABLE = True
except ImportError:
    BOOTSTRAP_AVAILABLE = False
    Entity = None
    EntityType = None
    EntityCollection = None
    ResolvedEntity = None
    RulePack = None


# CONFIG


@dataclass(frozen=True)
class ResolverConfigV2:
    """Configuration for v2 relationship resolution."""
    similarity_min: float = 0.50
    band_y_factor: float = 0.60
    band_x_factor: float = 0.35
    band_x_min_px: float = 12.0
    distance_decay_px: float = 150.0
    align_tolerance_px: float = 10.0
    prefer_structured: bool = True
    max_candidates_per_block: int = 200
    topk_anchors_per_block: int = 10
    kv_split_regex: str = r"\s*[:=\u2013\u2014\-–]\s*"
    series_mode: str = "first_only"
    series_max_hops: int = 50
    gap_stop_multiplier: float = 1.8
    unit_catalog: Tuple[str, ...] = (
        "mm", "cm", "m", "in", "ft", "psi", "bar", "ph", "nm", "kg", "g", "lb", "°c", "°f",
    )
    alias_exact_boost: float = 0.10


DEFAULT_CONFIG = ResolverConfigV2()


# CONTEXT GRAPH BUILDER


@dataclass
class ContextNode:
    """A node in the context graph representing an entity."""
    entity_id: str
    entity_type: str
    canonical_value: str
    text: str
    region_id: Optional[str] = None
    page_id: str = ""
    span: Tuple[int, int] = (0, 0)
    confidence: float = 0.5
    source: str = "unknown"
    bbox: Optional[Tuple[float, float, float, float]] = None
    
    @classmethod
    def from_entity(cls, entity: "Entity", entity_id: str = None) -> "ContextNode":
        """Create ContextNode from bootstrap Entity."""
        if entity_id is None:
            entity_id = f"{entity.page_id}_{entity.region_id}_{hash(entity.text) % 10000:04d}"
        
        return cls(
            entity_id=entity_id,
            entity_type=entity.entity_type.value if hasattr(entity.entity_type, "value") else str(entity.entity_type),
            canonical_value=entity.canonical or entity.text,
            text=entity.text,
            region_id=entity.region_id,
            page_id=entity.page_id,
            span=entity.span,
            confidence=entity.confidence,
            source=entity.source,
        )
    
    @classmethod
    def from_resolved_entity(cls, entity: "ResolvedEntity", entity_id: str = None) -> "ContextNode":
        """Create ContextNode from proto ResolvedEntity."""
        if entity_id is None:
            entity_id = f"{entity.page_id}_{entity.region_id}_{hash(entity.value) % 10000:04d}"
        
        return cls(
            entity_id=entity_id,
            entity_type=entity.entity_type.value if hasattr(entity.entity_type, "value") else str(entity.entity_type),
            canonical_value=entity.canonical_value,
            text=entity.value,
            region_id=entity.region_id,
            page_id=entity.page_id,
            span=entity.span,
            confidence=entity.confidence,
            source=entity.rule_id.replace("proto.", "") if entity.rule_id.startswith("proto.") else entity.rule_id,
        )


@dataclass
class ContextEdge:
    """An edge in the context graph representing a potential relationship."""
    node_a: str
    node_b: str
    score: float
    features: Dict[str, float] = field(default_factory=dict)
    relationship_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.node_a,
            "to": self.node_b,
            "score": self.score,
            "features": self.features,
            "type": self.relationship_type,
        }


@dataclass
class ContextGraph:
    """Entity-level context graph for a region or document."""
    nodes: Dict[str, ContextNode] = field(default_factory=dict)
    edges: List[ContextEdge] = field(default_factory=list)
    region_id: Optional[str] = None
    page_id: str = ""
    
    def add_node(self, node: ContextNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.entity_id] = node
    
    def add_edge(self, edge: ContextEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
    
    def get_nodes_by_type(self, entity_type: str) -> List[ContextNode]:
        """Get all nodes of a specific type."""
        return [n for n in self.nodes.values() if n.entity_type == entity_type]
    
    def get_edges_for_node(self, node_id: str) -> List[ContextEdge]:
        """Get all edges connected to a node."""
        return [e for e in self.edges if e.node_a == node_id or e.node_b == node_id]
    
    def get_related_nodes(self, node_id: str, min_score: float = 0.0) -> List[Tuple[ContextNode, float]]:
        """Get nodes related to a given node with their scores."""
        related = []
        for edge in self.edges:
            if edge.score < min_score:
                continue
            if edge.node_a == node_id and edge.node_b in self.nodes:
                related.append((self.nodes[edge.node_b], edge.score))
            elif edge.node_b == node_id and edge.node_a in self.nodes:
                related.append((self.nodes[edge.node_a], edge.score))
        return sorted(related, key=lambda x: x[1], reverse=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "page_id": self.page_id,
            "nodes": {k: {
                "entity_type": v.entity_type,
                "canonical_value": v.canonical_value,
                "text": v.text,
                "confidence": v.confidence,
                "source": v.source,
            } for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
        }



# RELATIONSHIP SCORING ENGINE


@dataclass
class ScoringWeights:
    """Configurable weights for relationship scoring."""
    same_row: float = 0.35
    same_region: float = 0.25
    type_prior: float = 0.20
    distance_decay: float = 0.10
    confidence_product: float = 0.10
    
    # Type-pair priors (entity_type_a, entity_type_b) -> prior score
    type_priors: Dict[Tuple[str, str], float] = field(default_factory=lambda: {
        ("TAG", "PART"): 0.9,
        ("TAG", "SIZE"): 0.85,
        ("TAG", "MATERIAL"): 0.8,
        ("TAG", "PRESSURE"): 0.75,
        ("TAG", "TEMPERATURE"): 0.75,
        ("TAG", "UNIT"): 0.6,
        ("TAG", "ACTION"): 0.7,
        ("TAG", "DESCRIPTION"): 0.65,
        ("PART", "MATERIAL"): 0.85,
        ("PART", "SIZE"): 0.8,
        ("PART", "UNIT"): 0.7,
        ("MATERIAL", "SIZE"): 0.6,
        ("ACTION", "TAG"): 0.7,
        ("ACTION", "PART"): 0.65,
    })
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoringWeights":
        """Load weights from dictionary (e.g., from JSON config)."""
        type_priors = {}
        if "type_priors" in data:
            for key, value in data["type_priors"].items():
                # Handle both "TAG,PART" and ("TAG", "PART") formats
                if isinstance(key, str):
                    parts = key.split(",")
                    if len(parts) == 2:
                        type_priors[(parts[0].strip(), parts[1].strip())] = value
                else:
                    type_priors[tuple(key)] = value
        
        return cls(
            same_row=data.get("same_row", 0.35),
            same_region=data.get("same_region", 0.25),
            type_prior=data.get("type_prior", 0.20),
            distance_decay=data.get("distance_decay", 0.10),
            confidence_product=data.get("confidence_product", 0.10),
            type_priors=type_priors if type_priors else cls.__dataclass_fields__["type_priors"].default_factory(),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "same_row": self.same_row,
            "same_region": self.same_region,
            "type_prior": self.type_prior,
            "distance_decay": self.distance_decay,
            "confidence_product": self.confidence_product,
            "type_priors": {f"{k[0]},{k[1]}": v for k, v in self.type_priors.items()},
        }


DEFAULT_SCORING_WEIGHTS = ScoringWeights()


class RelationshipScorer:
    """Scores relationships between entities based on contextual features."""
    
    def __init__(
        self,
        weights: ScoringWeights = None,
        distance_decay_px: float = 150.0,
        stats_pack: Any = None,  # StatsPack from NER_learning
    ):
        self.weights = weights or DEFAULT_SCORING_WEIGHTS
        self.distance_decay_px = distance_decay_px
        self.stats_pack = stats_pack
        
        # Extract co-occurrence stats from StatsPack if available
        self._cooccurrence_priors: Dict[Tuple[str, str], float] = {}
        self._load_stats_pack()
    
    def _load_stats_pack(self):
        """Load co-occurrence priors from StatsPack if available."""
        if not self.stats_pack:
            return
        
        # Try to extract token_frequencies for co-occurrence stats
        token_freq = None
        if hasattr(self.stats_pack, "token_frequencies"):
            token_freq = self.stats_pack.token_frequencies
        elif hasattr(self.stats_pack, "token_frequency"):
            token_freq = self.stats_pack.token_frequency
        
        if not token_freq:
            return
        
        # Build co-occurrence priors from token frequencies
        # If token_freq is a TokenFrequencies object, extract by_type
        if hasattr(token_freq, "by_type"):
            by_type = token_freq.by_type
        elif isinstance(token_freq, dict):
            by_type = token_freq.get("by_type", token_freq)
        else:
            return
        
        # Calculate co-occurrence priors based on frequency ratios
        total_tokens = sum(by_type.values()) if by_type else 1
        for type_a, count_a in by_type.items():
            for type_b, count_b in by_type.items():
                if type_a != type_b:
                    # Co-occurrence prior based on relative frequencies
                    # Higher frequency pairs get higher priors
                    prior = math.sqrt((count_a / total_tokens) * (count_b / total_tokens))
                    self._cooccurrence_priors[(type_a, type_b)] = min(0.9, prior * 10)
    
    def get_cooccurrence_prior(self, type_a: str, type_b: str) -> Optional[float]:
        """Get co-occurrence prior from StatsPack if available."""
        if not self._cooccurrence_priors:
            return None
        
        return self._cooccurrence_priors.get((type_a, type_b)) or \
               self._cooccurrence_priors.get((type_b, type_a))
    
    def adjust_confidence_for_ocr(self, confidence: float, text: str) -> float:
        """Adjust entity confidence based on OCR confusion matrix."""
        if not self.stats_pack:
            return confidence
        
        confusion_matrix = None
        if hasattr(self.stats_pack, "confusion_matrix"):
            confusion_matrix = self.stats_pack.confusion_matrix
        
        if not confusion_matrix:
            return confidence
        
        # Check if text contains characters with high confusion rates
        if hasattr(confusion_matrix, "substitutions"):
            subs = confusion_matrix.substitutions
        elif isinstance(confusion_matrix, dict):
            subs = confusion_matrix.get("substitutions", confusion_matrix)
        else:
            return confidence
        
        # Calculate OCR risk factor based on confusable characters in text
        ocr_risk = 0.0
        for char in text:
            if char in subs:
                ocr_risk += 0.02
        
        ocr_risk = min(0.2, ocr_risk)
        
        return confidence * (1.0 - ocr_risk)
    
    def compute_features(
        self,
        node_a: ContextNode,
        node_b: ContextNode,
        lines: List["UnifiedLine"] = None,
    ) -> Dict[str, float]:
        """Compute relationship features between two entities."""
        features = {}
        
        # Same region check
        features["same_region"] = 1.0 if (
            node_a.region_id and node_b.region_id and 
            node_a.region_id == node_b.region_id
        ) else 0.0
        
        # Same row check (using bbox y-overlap or span proximity)
        features["same_row"] = self._compute_same_row(node_a, node_b, lines)
        
        # Token/span distance
        features["token_distance"] = self._compute_distance_decay(node_a, node_b)
        
        # Type prior - use StatsPack co-occurrence if available, else default weights
        type_pair = (node_a.entity_type, node_b.entity_type)
        type_pair_rev = (node_b.entity_type, node_a.entity_type)
        
        # Try StatsPack co-occurrence prior first (learned from data)
        cooccurrence_prior = self.get_cooccurrence_prior(node_a.entity_type, node_b.entity_type)
        if cooccurrence_prior is not None:
            # Blend learned prior with default prior (don't blindly follow learned rules)
            default_prior = self.weights.type_priors.get(
                type_pair, 
                self.weights.type_priors.get(type_pair_rev, 0.3)
            )
            # Weight: 60% learned, 40% default - ensures we don't blindly follow
            features["type_prior"] = 0.6 * cooccurrence_prior + 0.4 * default_prior
        else:
            features["type_prior"] = self.weights.type_priors.get(
                type_pair, 
                self.weights.type_priors.get(type_pair_rev, 0.3)
            )
        
        # Confidence product - adjust for OCR if StatsPack available
        conf_a = self.adjust_confidence_for_ocr(node_a.confidence, node_a.text)
        conf_b = self.adjust_confidence_for_ocr(node_b.confidence, node_b.text)
        features["confidence_product"] = math.sqrt(conf_a * conf_b)
        
        return features
    
    def _compute_same_row(
        self,
        node_a: ContextNode,
        node_b: ContextNode,
        lines: List["UnifiedLine"] = None,
    ) -> float:
        """Check if entities are on the same row."""
        # If we have bboxes, use y-overlap
        if node_a.bbox and node_b.bbox:
            y_a_mid = (node_a.bbox[1] + node_a.bbox[3]) / 2
            y_b_mid = (node_b.bbox[1] + node_b.bbox[3]) / 2
            height = max(
                node_a.bbox[3] - node_a.bbox[1],
                node_b.bbox[3] - node_b.bbox[1],
                10  # minimum
            )
            if abs(y_a_mid - y_b_mid) < height * 0.5:
                return 1.0
        
        # Fallback: check if spans are close (same logical line)
        if node_a.span and node_b.span:
            span_gap = abs(node_a.span[1] - node_b.span[0])
            if span_gap < 50:  # Within ~50 characters
                return 0.8
        
        return 0.0
    
    def _compute_distance_decay(self, node_a: ContextNode, node_b: ContextNode) -> float:
        """Compute distance-based decay score."""
        # Use bbox distance if available
        if node_a.bbox and node_b.bbox:
            x_dist = abs((node_a.bbox[0] + node_a.bbox[2]) / 2 - (node_b.bbox[0] + node_b.bbox[2]) / 2)
            y_dist = abs((node_a.bbox[1] + node_a.bbox[3]) / 2 - (node_b.bbox[1] + node_b.bbox[3]) / 2)
            dist = math.sqrt(x_dist**2 + y_dist**2)
            return math.exp(-dist / self.distance_decay_px)
        
        # Fallback: use span distance
        if node_a.span and node_b.span:
            span_dist = abs(node_a.span[0] - node_b.span[0])
            return math.exp(-span_dist / 100)  # Decay over ~100 characters
        
        return 0.5  # Default moderate score
    
    def score_relationship(
        self,
        node_a: ContextNode,
        node_b: ContextNode,
        lines: List["UnifiedLine"] = None,
        query_context: Dict[str, Any] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Score the relationship between two entities."""
        features = self.compute_features(node_a, node_b, lines)
        
        # Apply weights
        weights = self.weights
        
        # Allow query context to adjust weights dynamically
        if query_context:
            # Example: boost same_row for table queries
            if query_context.get("format") == "table":
                features["same_row"] *= 1.2
        
        score = (
            weights.same_row * features["same_row"] +
            weights.same_region * features["same_region"] +
            weights.type_prior * features["type_prior"] +
            weights.distance_decay * features["token_distance"] +
            weights.confidence_product * features["confidence_product"]
        )
        
        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        
        return score, features
    
    def infer_relationship_type(
        self,
        node_a: ContextNode,
        node_b: ContextNode,
    ) -> Optional[str]:
        """Infer the relationship type based on entity types."""
        type_a, type_b = node_a.entity_type, node_b.entity_type
        
        # Define relationship mappings
        relationship_map = {
            ("TAG", "PART"): "HAS_PART",
            ("TAG", "MATERIAL"): "HAS_MATERIAL",
            ("TAG", "SIZE"): "HAS_SIZE",
            ("TAG", "PRESSURE"): "HAS_PRESSURE",
            ("TAG", "TEMPERATURE"): "HAS_TEMPERATURE",
            ("TAG", "UNIT"): "HAS_UNIT",
            ("TAG", "ACTION"): "HAS_ACTION",
            ("TAG", "DESCRIPTION"): "HAS_DESCRIPTION",
            ("PART", "MATERIAL"): "MADE_OF",
            ("PART", "SIZE"): "HAS_SIZE",
            ("ACTION", "TAG"): "ACTS_ON",
            ("ACTION", "PART"): "ACTS_ON",
        }
        
        return relationship_map.get((type_a, type_b)) or relationship_map.get((type_b, type_a))

# QUERY INTERPRETER

@dataclass
class EntityAnchor:
    """An anchor specification derived from query interpretation."""
    entity_type: str
    canonical_value: Optional[str] = None
    original_text: str = ""
    confidence: float = 0.8
    source: str = "query"
    
    def matches(self, node: ContextNode, fuzzy_threshold: float = 0.8) -> bool:
        """Check if this anchor matches a context node."""
        if self.entity_type != node.entity_type:
            return False
        
        if self.canonical_value:
            # Exact match on canonical
            if node.canonical_value.lower() == self.canonical_value.lower():
                return True
            # Fuzzy match
            ratio = fuzz.ratio(node.canonical_value.lower(), self.canonical_value.lower()) / 100
            return ratio >= fuzzy_threshold
        
        return True  # Type match only


class QueryInterpreter:
    """ Interprets user queries into entity anchors. """
    
    def __init__(self, rule_pack: "RulePack" = None):
        self.rule_pack = rule_pack
        self._build_lookup_tables()
    
    def _build_lookup_tables(self):
        """Build reverse lookup tables from RulePack."""
        self.term_to_entity: Dict[str, Tuple[str, str]] = {}  # term -> (entity_type, canonical)
        self.synonyms: Dict[str, str] = {}  # synonym -> canonical
        
        if not self.rule_pack:
            return
        
        # Build from dictionaries
        dict_to_type = {
            "tags": "TAG",
            "parts": "PART",
            "units": "UNIT",
            "materials": "MATERIAL",
            "actions": "ACTION",
        }
        
        for dict_name, entity_type in dict_to_type.items():
            if dict_name in self.rule_pack.dictionaries:
                for term in self.rule_pack.dictionaries[dict_name]:
                    self.term_to_entity[term.lower()] = (entity_type, term)
        
        # Build from synonyms
        for dict_name, entity_type in dict_to_type.items():
            if dict_name in self.rule_pack.synonyms:
                for synonym, canonical in self.rule_pack.synonyms[dict_name].items():
                    self.synonyms[synonym.lower()] = canonical
                    self.term_to_entity[synonym.lower()] = (entity_type, canonical)
    
    def interpret(self, query: str) -> List[EntityAnchor]:
        """Interpret a query string into entity anchors."""
        anchors = []
        query_lower = query.lower()
        tokens = query_lower.split()
        
        # Try to match known terms
        matched_indices = set()
        
        # Multi-word matching (longest first)
        for n in range(min(4, len(tokens)), 0, -1):
            for i in range(len(tokens) - n + 1):
                if any(j in matched_indices for j in range(i, i + n)):
                    continue
                
                phrase = " ".join(tokens[i:i + n])
                
                # Check direct dictionary match
                if phrase in self.term_to_entity:
                    entity_type, canonical = self.term_to_entity[phrase]
                    anchors.append(EntityAnchor(
                        entity_type=entity_type,
                        canonical_value=canonical,
                        original_text=phrase,
                        confidence=0.9,
                        source="dictionary",
                    ))
                    matched_indices.update(range(i, i + n))
                
                # Check synonym match
                elif phrase in self.synonyms:
                    canonical = self.synonyms[phrase]
                    # Find entity type from canonical
                    if canonical.lower() in self.term_to_entity:
                        entity_type, _ = self.term_to_entity[canonical.lower()]
                        anchors.append(EntityAnchor(
                            entity_type=entity_type,
                            canonical_value=canonical,
                            original_text=phrase,
                            confidence=0.85,
                            source="synonym",
                        ))
                        matched_indices.update(range(i, i + n))
        
        # Pattern-based detection for unmatched tokens
        for i, token in enumerate(tokens):
            if i in matched_indices:
                continue
            
            # SIZE pattern 
            if re.match(r'^DN\d{1,4}$', token, re.IGNORECASE) or re.match(r'^\d{1,2}["\']$', token):
                anchors.append(EntityAnchor(
                    entity_type="SIZE",
                    canonical_value=token.upper(),
                    original_text=token,
                    confidence=0.8,
                    source="pattern",
                ))
                matched_indices.add(i)
            
            # MATERIAL pattern 
            elif re.match(r'^(SS|CS)\d{2,3}[A-Z]?$', token, re.IGNORECASE) or re.match(r'^[AB]\d{3}$', token, re.IGNORECASE):
                anchors.append(EntityAnchor(
                    entity_type="MATERIAL",
                    canonical_value=token.upper(),
                    original_text=token,
                    confidence=0.8,
                    source="pattern",
                ))
                matched_indices.add(i)
            
            # TAG pattern 
            elif re.match(r'^[A-Z]{1,4}[-–]\d{2,5}[A-Z]?$', token, re.IGNORECASE) or \
                 re.match(r'^[A-Z]{2,4}\d{3,5}[A-Z]?$', token, re.IGNORECASE):
                anchors.append(EntityAnchor(
                    entity_type="TAG",
                    canonical_value=token.upper(),
                    original_text=token,
                    confidence=0.85,
                    source="pattern",
                ))
                matched_indices.add(i)
        
        return anchors
    
    def interpret_with_fallback(self, query: str) -> Tuple[List[EntityAnchor], List[str]]:
        """ Interpret query with fallback to unmatched terms. """
        anchors = self.interpret(query)
        
        # Find unmatched terms
        query_lower = query.lower()
        tokens = query_lower.split()
        matched_texts = {a.original_text.lower() for a in anchors}
        
        unmatched = []
        for token in tokens:
            if token not in matched_texts and not any(token in mt for mt in matched_texts):
                unmatched.append(token)
        
        return anchors, unmatched


# ENTITY OVERLAY


class EntityOverlay:
    """Overlays Module B entities onto resolver regions."""
    
    # Format type to region ID type mapping
    FORMAT_TO_REGION_TYPE = {
        "TABLE": "poly_id",
        "INFO_CLUSTER": "poly_id",
        "CLUSTER": "poly_id",
        "KEY_VALUE": "block_id",
        "LIST": "block_id",
        "PARAGRAPH": "block_id",
        "STRING": "line_id",
    }
    
    def __init__(
        self,
        entities: List["Entity"] = None,
        proto_entities: List["ResolvedEntity"] = None,
        rule_pack: "RulePack" = None,
        # Optional region mapping for explicit ID translation
        region_mapping: Dict[str, str] = None,
    ):
        self.entities = entities or []
        self.proto_entities = proto_entities or []
        self.rule_pack = rule_pack
        self.region_mapping = region_mapping or {}
        
        # Build indices
        self._by_region: Dict[str, List["Entity"]] = defaultdict(list)
        self._by_page: Dict[str, List["Entity"]] = defaultdict(list)
        self._by_type: Dict[str, List["Entity"]] = defaultdict(list)
        # Additional index by cluster_block for TABLE/CLUSTER lookups
        self._by_cluster: Dict[str, List["Entity"]] = defaultdict(list)
        
        self._build_indices()
    
    def _build_indices(self):
        """Build lookup indices for entities."""
        # Merge entities if we have both regex and proto
        if self.entities and self.proto_entities:
            merged = merge_entities(self.entities, self.proto_entities) if BOOTSTRAP_AVAILABLE else self.entities
        elif self.entities:
            merged = self.entities
        elif self.proto_entities and BOOTSTRAP_AVAILABLE:
            # Convert proto to Entity
            from bootstrap.entity_merge import resolved_entity_to_entity
            merged = [resolved_entity_to_entity(pe) for pe in self.proto_entities]
        else:
            merged = []
        
        for entity in merged:
            # Index by region_id (block_id from Module B)
            if entity.region_id:
                self._by_region[entity.region_id].append(entity)
                
                # Also check if there's a mapping to a different ID
                if entity.region_id in self.region_mapping:
                    mapped_id = self.region_mapping[entity.region_id]
                    self._by_region[mapped_id].append(entity)
            
            # Index by page
            if entity.page_id:
                self._by_page[entity.page_id].append(entity)
            
            # Index by entity type
            entity_type = entity.entity_type.value if hasattr(entity.entity_type, "value") else str(entity.entity_type)
            self._by_type[entity_type].append(entity)
            
            # Index by cluster_block if available in metadata
            if hasattr(entity, "metadata") and entity.metadata:
                cluster_id = entity.metadata.get("cluster_block") or entity.metadata.get("cluster_id")
                if cluster_id:
                    self._by_cluster[cluster_id].append(entity)
    
    def map_region_id(self, region_id: str, format_type: str = None) -> str:
        """Map a region_id based on format type."""
        # Check explicit mapping first
        if region_id in self.region_mapping:
            return self.region_mapping[region_id]
        
        # No mapping needed if format type not specified
        return region_id
    
    def get_entities_for_region(
        self,
        region_id: str,
        page_id: str = None,
        entity_types: List[str] = None,
        min_confidence: float = 0.0,
        format_type: str = None,
    ) -> List["Entity"]:
        """Get entities for a specific region."""
        # Map region_id if needed
        mapped_id = self.map_region_id(region_id, format_type)
        
        # Try primary lookup
        entities = list(self._by_region.get(mapped_id, []))
        
        # For TABLE/CLUSTER formats, also check cluster index
        if format_type in ("TABLE", "INFO_CLUSTER", "CLUSTER") and not entities:
            entities = list(self._by_cluster.get(region_id, []))
        
        if page_id:
            entities = [e for e in entities if e.page_id == page_id]
        
        if entity_types:
            entities = [e for e in entities 
                       if (e.entity_type.value if hasattr(e.entity_type, "value") else str(e.entity_type)) in entity_types]
        
        if min_confidence > 0:
            entities = [e for e in entities if e.confidence >= min_confidence]
        
        return entities
    
    def get_entities_for_page(
        self,
        page_id: str,
        entity_types: List[str] = None,
        min_confidence: float = 0.0,
    ) -> List["Entity"]:
        """Get all entities for a page."""
        entities = self._by_page.get(page_id, [])
        
        if entity_types:
            entities = [e for e in entities 
                       if (e.entity_type.value if hasattr(e.entity_type, "value") else str(e.entity_type)) in entity_types]
        
        if min_confidence > 0:
            entities = [e for e in entities if e.confidence >= min_confidence]
        
        return entities
    
    def get_entities_by_type(
        self,
        entity_type: str,
        min_confidence: float = 0.0,
    ) -> List["Entity"]:
        """Get all entities of a specific type."""
        entities = self._by_type.get(entity_type, [])
        
        if min_confidence > 0:
            entities = [e for e in entities if e.confidence >= min_confidence]
        
        return entities
    
    def get_entities_in_span(
        self,
        region_id: str,
        span_start: int,
        span_end: int,
        overlap_threshold: float = 0.5,
    ) -> List["Entity"]:
        """Get entities whose spans overlap with given range."""
        entities = self._by_region.get(region_id, [])
        
        result = []
        for entity in entities:
            e_start, e_end = entity.span
            overlap_start = max(span_start, e_start)
            overlap_end = min(span_end, e_end)
            
            if overlap_start < overlap_end:
                overlap_len = overlap_end - overlap_start
                entity_len = e_end - e_start
                if entity_len > 0 and overlap_len / entity_len >= overlap_threshold:
                    result.append(entity)
        
        return result

# ENTITY-AWARE TEMPLATES

@dataclass
class ResolvedRegion:
    """A resolved region with attached entities. """
    region_id: str
    page_id: str
    format_type: str  # table, kv, list, paragraph, string, cluster
    text: str = ""
    values: List[str] = field(default_factory=list)
    entities: List["Entity"] = field(default_factory=list)
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_entities_by_type(self, entity_type: str) -> List["Entity"]:
        """Get entities of a specific type in this region."""
        return [e for e in self.entities 
                if (e.entity_type.value if hasattr(e.entity_type, "value") else str(e.entity_type)) == entity_type]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "page_id": self.page_id,
            "format_type": self.format_type,
            "text": self.text,
            "values": self.values,
            "entities": [e.to_dict() for e in self.entities] if self.entities else [],
            "score": self.score,
            "metadata": self.metadata,
        }


class EntityTemplates:
    """ Entity-aware template functions for common relationship queries."""
    
    def __init__(
        self,
        entity_overlay: EntityOverlay,
        scorer: RelationshipScorer = None,
        min_score: float = 0.5,
    ):
        self.overlay = entity_overlay
        self.scorer = scorer or RelationshipScorer()
        self.min_score = min_score
    
    def get_equipment_spec(
        self,
        tag: str,
        page_id: str = None,
        min_score: float = None,
    ) -> Optional[ResolvedRegion]:
        """Get equipment specification """
        min_score = min_score if min_score is not None else self.min_score
        
        # Find TAG entity
        tag_entities = self.overlay.get_entities_by_type("TAG")
        tag_entity = None
        for e in tag_entities:
            if e.canonical and e.canonical.upper() == tag.upper():
                tag_entity = e
                break
            if e.text.upper() == tag.upper():
                tag_entity = e
                break
        
        if not tag_entity:
            return None
        
        # Get entities in same region
        region_entities = self.overlay.get_entities_for_region(
            tag_entity.region_id,
            page_id=page_id,
        )
        
        # Build context graph and score relationships
        tag_node = ContextNode.from_entity(tag_entity) if BOOTSTRAP_AVAILABLE else None
        
        related_entities = []
        for entity in region_entities:
            if entity == tag_entity:
                continue
            
            if tag_node and BOOTSTRAP_AVAILABLE:
                entity_node = ContextNode.from_entity(entity)
                score, _ = self.scorer.score_relationship(tag_node, entity_node)
                if score >= min_score:
                    related_entities.append(entity)
            else:
                # Fallback: include all entities in region
                related_entities.append(entity)
        
        return ResolvedRegion(
            region_id=tag_entity.region_id or "",
            page_id=tag_entity.page_id,
            format_type="equipment_spec",
            text=tag,
            values=[tag],
            entities=[tag_entity] + related_entities,
            score=tag_entity.confidence,
            metadata={"tag": tag},
        )
    
    def get_actions_for_tag(
        self,
        tag: str,
        page_id: str = None,
        min_score: float = None,
    ) -> List[ResolvedRegion]:
        """Get actions associated"""
        min_score = min_score if min_score is not None else self.min_score
        
        # Find TAG entity
        tag_entities = self.overlay.get_entities_by_type("TAG")
        tag_entity = None
        for e in tag_entities:
            if (e.canonical and e.canonical.upper() == tag.upper()) or e.text.upper() == tag.upper():
                tag_entity = e
                break
        
        if not tag_entity:
            return []
        
        # Find ACTION entities
        action_entities = self.overlay.get_entities_by_type("ACTION")
        
        results = []
        tag_node = ContextNode.from_entity(tag_entity) if BOOTSTRAP_AVAILABLE and tag_entity else None
        
        for action in action_entities:
            if page_id and action.page_id != page_id:
                continue
            
            # Score relationship
            if tag_node and BOOTSTRAP_AVAILABLE:
                action_node = ContextNode.from_entity(action)
                score, _ = self.scorer.score_relationship(tag_node, action_node)
                if score < min_score:
                    continue
            else:
                score = 0.5
            
            # Get region entities
            region_entities = self.overlay.get_entities_for_region(action.region_id)
            
            results.append(ResolvedRegion(
                region_id=action.region_id or "",
                page_id=action.page_id,
                format_type="action",
                text=action.text,
                values=[action.canonical or action.text],
                entities=region_entities,
                score=score,
                metadata={"tag": tag, "action": action.text},
            ))
        
        return sorted(results, key=lambda r: r.score, reverse=True)
    
    def get_parts_by_material(
        self,
        material: str,
        page_id: str = None,
        min_score: float = None,
    ) -> List[ResolvedRegion]:
        """Get parts made of a specific material."""
        min_score = min_score if min_score is not None else self.min_score
        
        # Find MATERIAL entity
        material_entities = self.overlay.get_entities_by_type("MATERIAL")
        material_entity = None
        for e in material_entities:
            if (e.canonical and e.canonical.upper() == material.upper()) or e.text.upper() == material.upper():
                material_entity = e
                break
        
        if not material_entity:
            return []
        
        # Find PART entities in same regions
        part_entities = self.overlay.get_entities_by_type("PART")
        
        results = []
        material_node = ContextNode.from_entity(material_entity) if BOOTSTRAP_AVAILABLE else None
        
        for part in part_entities:
            if page_id and part.page_id != page_id:
                continue
            
            # Score relationship
            if material_node and BOOTSTRAP_AVAILABLE:
                part_node = ContextNode.from_entity(part)
                score, _ = self.scorer.score_relationship(material_node, part_node)
                if score < min_score:
                    continue
            else:
                # Fallback: same region check
                if part.region_id != material_entity.region_id:
                    continue
                score = 0.5
            
            # Get region entities
            region_entities = self.overlay.get_entities_for_region(part.region_id)
            
            results.append(ResolvedRegion(
                region_id=part.region_id or "",
                page_id=part.page_id,
                format_type="part",
                text=part.text,
                values=[part.canonical or part.text],
                entities=region_entities,
                score=score,
                metadata={"material": material, "part": part.text},
            ))
        
        return sorted(results, key=lambda r: r.score, reverse=True)


# GRAPH EXPORT

@dataclass
class GraphEdge:
    """An edge in the entity relationship graph."""
    from_entity: str  # canonical value or ID
    to_entity: str
    from_type: str
    to_type: str
    relationship_type: str
    score: float
    features: Dict[str, float] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.from_entity,
            "to": self.to_entity,
            "from_type": self.from_type,
            "to_type": self.to_type,
            "type": self.relationship_type,
            "score": self.score,
            "features": self.features,
            "sources": self.sources,
            "metadata": self.metadata,
        }


class GraphExporter:
    """ Exports entity relationships as graph edges."""
    
    def __init__(self, scorer: RelationshipScorer = None, min_score: float = 0.3):
        self.scorer = scorer or RelationshipScorer()
        self.min_score = min_score
    
    def build_edges_from_context_graph(
        self,
        graph: ContextGraph,
    ) -> List[GraphEdge]:
        """Build graph edges from a context graph."""
        result = []
        
        for edge in graph.edges:
            if edge.score < self.min_score:
                continue
            
            node_a = graph.nodes.get(edge.node_a)
            node_b = graph.nodes.get(edge.node_b)
            
            if not node_a or not node_b:
                continue
            
            result.append(GraphEdge(
                from_entity=node_a.canonical_value,
                to_entity=node_b.canonical_value,
                from_type=node_a.entity_type,
                to_type=node_b.entity_type,
                relationship_type=edge.relationship_type or self.scorer.infer_relationship_type(node_a, node_b) or "RELATED_TO",
                score=edge.score,
                features=edge.features,
                sources=[node_a.source, node_b.source],
                metadata={
                    "region_id": graph.region_id,
                    "page_id": graph.page_id,
                },
            ))
        
        return result
    
    def build_edges_from_entities(
        self,
        entities: List["Entity"],
        lines: List["UnifiedLine"] = None,
    ) -> List[GraphEdge]:
        """Build graph edges directly from entities."""
        if not BOOTSTRAP_AVAILABLE or not entities:
            return []
        
        # Build context graph
        graph = self.build_context_graph(entities, lines)
        
        return self.build_edges_from_context_graph(graph)
    
    def build_context_graph(
        self,
        entities: List["Entity"],
        lines: List["UnifiedLine"] = None,
        region_id: str = None,
        page_id: str = None,
    ) -> ContextGraph:
        """Build a context graph from entities."""
        graph = ContextGraph(region_id=region_id, page_id=page_id)
        
        if not BOOTSTRAP_AVAILABLE:
            return graph
        
        # Add nodes
        for i, entity in enumerate(entities):
            node = ContextNode.from_entity(entity, entity_id=f"E{i:04d}")
            graph.add_node(node)
        
        # Compute pairwise edges
        node_list = list(graph.nodes.values())
        for i, node_a in enumerate(node_list):
            for node_b in node_list[i + 1:]:
                score, features = self.scorer.score_relationship(node_a, node_b, lines)
                
                if score >= self.min_score:
                    rel_type = self.scorer.infer_relationship_type(node_a, node_b)
                    graph.add_edge(ContextEdge(
                        node_a=node_a.entity_id,
                        node_b=node_b.entity_id,
                        score=score,
                        features=features,
                        relationship_type=rel_type,
                    ))
        
        return graph
    
    def export_to_dict(
        self,
        edges: List[GraphEdge],
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """Export edges to dictionary format."""
        return {
            "edges": [e.to_dict() for e in edges],
            "count": len(edges),
            "metadata": {
                "min_score": self.min_score,
            } if include_metadata else {},
        }
    
    def export_to_json(
        self,
        edges: List[GraphEdge],
        path: str,
        include_metadata: bool = True,
    ) -> None:
        """Export edges to JSON file."""
        data = self.export_to_dict(edges, include_metadata)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# CONTEXT GRAPH BUILDER FUNCTION

def build_context_graph(
    entities: List["Entity"],
    lines: List["UnifiedLine"] = None,
    scorer: RelationshipScorer = None,
    min_score: float = 0.3,
    region_id: str = None,
    page_id: str = None,
) -> ContextGraph:
    """Build a context graph from entities. """
    exporter = GraphExporter(scorer=scorer, min_score=min_score)
    return exporter.build_context_graph(entities, lines, region_id, page_id)


# UNIFIED LINE TYPE 

@dataclass
class UnifiedLine:
    """Unified line representation that wraps both RegionLine and ClusterLine"""
    id: str
    text: str
    text_norm: str
    bbox: BBox
    tokens: List[str]
    source_type: str  # "region" or "cluster"
    source_obj: Union[RegionLine, ClusterLine]
    
    # Computed geometry
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    x_mid: float = 0.0
    y_mid: float = 0.0
    width: float = 0.0
    height: float = 0.0
    
    # Optional metadata
    block_id: Optional[str] = None
    poly_id: Optional[str] = None
    
    def __post_init__(self):
        self.x0, self.y0, self.x1, self.y1 = self.bbox
        self.width = self.x1 - self.x0
        self.height = self.y1 - self.y0
        self.x_mid = (self.x0 + self.x1) / 2
        self.y_mid = (self.y0 + self.y1) / 2
    
    @classmethod
    def from_region_line(cls, line: RegionLine) -> "UnifiedLine":
        """Create UnifiedLine from RegionLine."""
        return cls(
            id=line.id,
            text=line.text_raw,
            text_norm=line.text_norm,
            bbox=line.bbox,
            tokens=line.tokens or [],
            source_type="region",
            source_obj=line,
            block_id=line.block_id,
        )
    
    @classmethod
    def from_cluster_line(cls, line: ClusterLine) -> "UnifiedLine":
        """Create UnifiedLine from ClusterLine."""
        return cls(
            id=line.id,
            text=line.text_raw,
            text_norm=line.text_norm,
            bbox=line.bbox,
            tokens=line.tokens or [],
            source_type="cluster",
            source_obj=line,
            poly_id=line.poly_id,
        )



# ANCHOR WRAPPER 


@dataclass
class ResolverAnchor:
    """Wrapper that bridges CandidateAnchor to resolver expectations."""
    anchor: CandidateAnchor
    line: Optional[UnifiedLine] = None  # Reconstructed line if available
    
    @property
    def score(self) -> float:
        return self.anchor.score
    
    @property
    def query_name(self) -> str:
        return self.anchor.query_name
    
    @property
    def source_type(self) -> str:
        return self.anchor.source_type
    
    @property
    def source_id(self) -> str:
        return self.anchor.source_id
    
    @property
    def block_id(self) -> str:
        return self.anchor.block_id
    
    @property
    def bbox(self) -> BBox:
        return self.anchor.bbox
    
    @property
    def text(self) -> str:
        return self.anchor.anchor_text


# TELEMETRY

@dataclass
class Telemetry:
    """Telemetry for tracking resolver decisions."""
    events: List[Dict[str, Any]] = field(default_factory=list)
    band_widen_count: int = 0
    orientation_fallback_count: int = 0
    axis_ambiguity_count: int = 0
    join_count: int = 0
    
    def log(self, event_type: str, data: Any = None) -> None:
        self.events.append({"type": event_type, "data": data})
    
    def log_band_widen(self) -> None:
        self.band_widen_count += 1
        self.log("band_widen")
    
    def log_orientation_fallback(self, reason: str) -> None:
        self.orientation_fallback_count += 1
        self.log("orientation_fallback", reason)
    
    def log_axis_ambiguity(self) -> None:
        self.axis_ambiguity_count += 1
        self.log("axis_ambiguity")
    
    def record(self, key: str, value: Any) -> None:
        self.log(key, value)


# RESULT SCHEMA

@dataclass
class RelationshipResult:
    """Standardized result from relationship resolution."""
    field: str
    format: str  # table, kv, list, paragraph, string, cluster
    definition: str
    values: List[str]
    score: float
    meta: Dict[str, Any] = field(default_factory=dict)


def make_result(
    field: str,
    format_type: str,
    definition: str,
    values: List[str],
    score: float,
    telemetry: Optional[Telemetry] = None,
    **extra_meta
) -> Dict[str, Any]:
    """Create a standardized result dictionary."""
    meta = dict(extra_meta)
    if telemetry:
        meta["telemetry"] = {
            "band_widen_count": telemetry.band_widen_count,
            "orientation_fallback_count": telemetry.orientation_fallback_count,
            "axis_ambiguity_count": telemetry.axis_ambiguity_count,
            "join_count": telemetry.join_count,
            "events": telemetry.events[-10:],  # Last 10 events
        }
    return {
        "field": field,
        "format": format_type,
        "definition": definition,
        "values": values,
        "score": score,
        "meta": meta,
    }


# SPATIAL STATISTICS ENGINE - Supports mixed geometry

@dataclass
class SpatialStats:
    """Spatial statistics computed from lines."""
    median_height: float
    median_width: float
    p40_height: float
    p35_width: float
    median_vgap: float
    median_hgap: float
    line_count: int = 0


class SpatialStatisticsEngine:
    """Computes spatial statistics from UnifiedLine objects."""
    
    @staticmethod
    def compute(lines: List[UnifiedLine]) -> SpatialStats:
        """Compute spatial statistics from a list of UnifiedLine objects."""
        if not lines:
            return SpatialStats(0, 0, 0, 0, 0, 0, 0)
        
        heights = np.array([ln.height for ln in lines if ln.height > 0])
        widths = np.array([ln.width for ln in lines if ln.width > 0])
        
        # Vertical gaps (based on y_mid sorted top→bottom)
        ys = sorted([ln.y_mid for ln in lines])
        v_gaps = np.diff(ys) if len(ys) > 1 else np.array([0])
        
        # Horizontal gaps (based on x_mid sorted left→right)
        xs = sorted([ln.x_mid for ln in lines])
        h_gaps = np.diff(xs) if len(xs) > 1 else np.array([0])
        
        median_height = float(np.median(heights)) if len(heights) else 0.0
        median_width = float(np.median(widths)) if len(widths) else 0.0
        
        p40_height = float(np.percentile(heights, 40)) if len(heights) else median_height
        p35_width = float(np.percentile(widths, 35)) if len(widths) else median_width
        
        median_vgap = float(np.median(v_gaps)) if len(v_gaps) else 0.0
        median_hgap = float(np.median(h_gaps)) if len(h_gaps) else 0.0
        
        return SpatialStats(
            median_height=median_height,
            median_width=median_width,
            p40_height=p40_height,
            p35_width=p35_width,
            median_vgap=median_vgap,
            median_hgap=median_hgap,
            line_count=len(lines),
        )
    
    @staticmethod
    def compute_from_raw(
        region_lines: List[RegionLine],
        cluster_lines: List[ClusterLine],
    ) -> SpatialStats:
        """Compute stats from raw v2 line types."""
        unified = []
        for rl in region_lines:
            unified.append(UnifiedLine.from_region_line(rl))
        for cl in cluster_lines:
            unified.append(UnifiedLine.from_cluster_line(cl))
        return SpatialStatisticsEngine.compute(unified)

# TOLERANCE ENGINE

@dataclass
class Tolerances:
    """Tolerance values for spatial analysis."""
    tau_y: float
    tau_x: float
    close_gap: float
    max_gap: float
    coord_tol: float
    expanded: bool = False


class ToleranceEngine:
    """Computes tolerance values from spatial statistics."""
    
    @staticmethod
    def compute(stats: SpatialStats, config: ResolverConfigV2 = DEFAULT_CONFIG) -> Tolerances:
        """Compute tolerances from spatial statistics."""
        tau_y = stats.p40_height if stats.p40_height > 0 else 12.0
        tau_x = max(config.band_x_min_px, stats.p35_width)
        
        if stats.median_vgap > 0 and stats.median_hgap > 0:
            close_gap = min(stats.median_vgap, stats.median_hgap) * config.band_y_factor
            max_gap = max(stats.median_vgap, stats.median_hgap) * config.gap_stop_multiplier
        else:
            close_gap = 6.0
            max_gap = 40.0
        
        coord_tol = max(tau_x, tau_y) * 0.5
        
        return Tolerances(
            tau_y=tau_y,
            tau_x=tau_x,
            close_gap=close_gap,
            max_gap=max_gap,
            coord_tol=coord_tol,
        )
    
    @staticmethod
    def expand_band(tol: Tolerances, telemetry: Telemetry) -> None:
        """Expand tolerance bands when insufficient candidates found."""
        telemetry.log_band_widen()
        factor = 1.35
        tol.tau_y *= factor
        tol.tau_x *= factor
        tol.coord_tol *= factor
        tol.expanded = True


# NEIGHBOR ENGINE - Builds spatial neighbor graph

@dataclass
class NeighborInfo:
    """Neighbor information for a line in each direction."""
    up: List[Tuple[UnifiedLine, float]] = field(default_factory=list)
    down: List[Tuple[UnifiedLine, float]] = field(default_factory=list)
    left: List[Tuple[UnifiedLine, float]] = field(default_factory=list)
    right: List[Tuple[UnifiedLine, float]] = field(default_factory=list)


class NeighborEngine:
    """Builds and queries spatial neighbor graphs."""
    
    @staticmethod
    def build_neighbor_graph(
        lines: List[UnifiedLine],
        tol: Tolerances
    ) -> Dict[str, NeighborInfo]:
        """Build neighbor graph for all lines."""
        graph: Dict[str, NeighborInfo] = {ln.id: NeighborInfo() for ln in lines}
        
        for i, a in enumerate(lines):
            for b in lines[i + 1:]:
                dx = b.x_mid - a.x_mid
                dy = b.y_mid - a.y_mid
                dist = math.sqrt(dx * dx + dy * dy)
                
                # Vertical alignment (same column)
                if abs(dx) <= tol.tau_x:
                    if dy > 0:
                        graph[a.id].down.append((b, dy))
                        graph[b.id].up.append((a, dy))
                    else:
                        graph[a.id].up.append((b, -dy))
                        graph[b.id].down.append((a, -dy))
                
                # Horizontal alignment (same row)
                if abs(dy) <= tol.tau_y:
                    if dx > 0:
                        graph[a.id].right.append((b, dx))
                        graph[b.id].left.append((a, dx))
                    else:
                        graph[a.id].left.append((b, -dx))
                        graph[b.id].right.append((a, -dx))
        
        # Sort neighbors by distance
        for nid, info in graph.items():
            info.up.sort(key=lambda x: x[1])
            info.down.sort(key=lambda x: x[1])
            info.left.sort(key=lambda x: x[1])
            info.right.sort(key=lambda x: x[1])
        
        return graph
    
    @staticmethod
    def select_orientation(
        anchor: UnifiedLine,
        graph: Dict[str, NeighborInfo],
        stats: SpatialStats,
        tol: Tolerances,
        telemetry: Telemetry
    ) -> int:
        """Select orientation case for value extraction."""
        info = graph.get(anchor.id, NeighborInfo())
        
        n_up = len(info.up)
        n_down = len(info.down)
        n_left = len(info.left)
        n_right = len(info.right)
        
        # Must have at least one neighbor
        if (n_up + n_down + n_left + n_right) == 0:
            telemetry.log_orientation_fallback("no-neighbors")
            return 1
        
        # Direction dominance score
        vertical_score = n_down * 2 + n_up
        horizontal_score = n_right * 2 + n_left
        
        # Header-token lexical bias
        txt = anchor.text.lower()
        row_header_bias = int(any(w in txt for w in ["qty", "quantity", "units", "#"]))
        col_header_bias = int(any(w in txt for w in ["description", "item", "material"]))
        
        vertical_score += row_header_bias
        horizontal_score += col_header_bias
        
        # Direct comparison
        if vertical_score > horizontal_score:
            if n_down > 0:
                return 1
            elif n_up > 0:
                return 3
        
        if horizontal_score > vertical_score:
            if n_right > 0:
                return 2
            elif n_left > 0:
                return 4
        
        # Tie-breaking by closest neighbor
        nearest_dist = math.inf
        orientation = 1
        
        def update_orientation(neighbors, case_id):
            nonlocal nearest_dist, orientation
            if neighbors:
                d = neighbors[0][1]
                if d < nearest_dist:
                    nearest_dist = d
                    orientation = case_id
        
        update_orientation(info.down, 1)
        update_orientation(info.right, 2)
        update_orientation(info.up, 3)
        update_orientation(info.left, 4)
        
        return orientation
    
    @staticmethod
    def collect_axis_candidates(
        anchor: UnifiedLine,
        lines: List[UnifiedLine],
        tol: Tolerances,
        orientation_case: int,
        telemetry: Telemetry
    ) -> List[UnifiedLine]:
        """Collect lines aligned along the axis direction."""
        axis = []
        center_x = anchor.x_mid
        center_y = anchor.y_mid
        
        # Vertical axis (Cases 1 & 3)
        if orientation_case in (1, 3):
            for ln in lines:
                if ln.id == anchor.id:
                    continue
                if abs(ln.x_mid - center_x) <= tol.tau_x:
                    axis.append(ln)
            axis.sort(key=lambda z: z.y_mid)
        
        # Horizontal axis (Cases 2 & 4)
        else:
            for ln in lines:
                if ln.id == anchor.id:
                    continue
                if abs(ln.y_mid - center_y) <= tol.tau_y:
                    axis.append(ln)
            axis.sort(key=lambda z: z.x_mid)
        
        # Remove duplicates
        seen = set()
        unique = []
        for ln in axis:
            if ln.id not in seen:
                seen.add(ln.id)
                unique.append(ln)
        
        telemetry.record("axis_candidates", len(unique))
        return unique


# GEOMETRY RECONSTRUCTION

class GeometryReconstructor:
    """Reconstructs geometry context from anchor source information."""
    
    def __init__(self, inputs: DataCleaningInputsV2):
        self.inputs = inputs
        
        # Build lookup indices
        self._region_line_by_id: Dict[str, RegionLine] = {
            rl.id: rl for rl in inputs.region_lines
        }
        self._cluster_line_by_id: Dict[str, ClusterLine] = {
            cl.id: cl for cl in inputs.cluster_lines
        }
        self._cluster_lines_by_poly: Dict[str, List[ClusterLine]] = defaultdict(list)
        for cl in inputs.cluster_lines:
            if cl.poly_id:
                self._cluster_lines_by_poly[cl.poly_id].append(cl)
        
        self._rectangle_by_poly: Dict[str, Rectangle] = {
            r.poly_id: r for r in inputs.rectangles
        }
        self._region_block_by_id: Dict[str, RegionBlock] = {
            rb.id: rb for rb in inputs.region_blocks
        }
    
    def get_lines_for_anchor(
        self,
        anchor: CandidateAnchor,
        consolidation_results: Optional[Dict[str, Any]] = None
    ) -> List[UnifiedLine]:
        """Get relevant lines for an anchor based on its source type."""
        source_type = anchor.source_type
        source_id = anchor.source_id
        
        if source_type == "table_cell":
            return self._get_table_lines(source_id, anchor.bbox, consolidation_results)
        elif source_type == "cluster":
            return self._get_cluster_lines(source_id)
        elif source_type == "region_line":
            return self._get_region_line(source_id)
        elif source_type == "standalone":
            return self._get_region_line(source_id)
        elif source_type in ("list", "paragraph", "key_value", "string"):
            # These come from cluster blocks, get cluster lines
            return self._get_cluster_lines(source_id)
        else:
            # Fallback: try cluster then region
            lines = self._get_cluster_lines(source_id)
            if not lines:
                lines = self._get_region_line(source_id)
            return lines
    
    def _get_table_lines(
        self,
        source_id: str,
        bbox: BBox,
        consolidation_results: Optional[Dict[str, Any]] = None
    ) -> List[UnifiedLine]:
        """Get ALL lines from a table for proper neighbor detection."""
        parts = source_id.split("/")
        if not parts:
            return []
        
        table_id = parts[0]
        lines = []
        
        # Strategy 1: Use consolidated table cells if available
        if consolidation_results:
            for table in consolidation_results.get("tables", []):
                if table.get("table_id") == table_id:
                    for cell in table.get("cells", []):
                        cell_text = cell.get("text", "").strip()
                        if not cell_text:
                            continue
                        cell_bbox = tuple(cell.get("bbox", (0, 0, 0, 0)))
                        cell_id = cell.get("cell_id", "")
                        
                        lines.append(UnifiedLine(
                            id=f"{table_id}/{cell_id}",
                            text=cell_text,
                            text_norm=cell_text.lower(),
                            bbox=cell_bbox,
                            tokens=cell_text.split(),
                            source_type="table_cell",
                            source_obj=cell,
                            poly_id=table_id,
                        ))
                    break
        
        # Strategy 2: Fallback to cluster lines if no consolidated data
        if not lines:
            # Get all cluster lines that belong to rectangles in this table
            for cl in self.inputs.cluster_lines:
                if cl.poly_id and cl.poly_id.startswith(table_id):
                    lines.append(UnifiedLine.from_cluster_line(cl))
            
            # If still no lines, get cluster lines by bbox overlap with table area
            if not lines:
                # Expand bbox to cover potential table area
                x1, y1, x2, y2 = bbox
                # Look for lines in a reasonable table area (expand significantly)
                table_margin = 500  # pixels
                for cl in self.inputs.cluster_lines:
                    cx1, cy1, cx2, cy2 = cl.bbox
                    if (cx1 < x2 + table_margin and cx2 > x1 - table_margin and
                        cy1 < y2 + table_margin and cy2 > y1 - table_margin):
                        lines.append(UnifiedLine.from_cluster_line(cl))
        
        return lines
    
    def _get_cluster_lines(self, poly_id: str) -> List[UnifiedLine]:
        """Get all cluster lines for a polygon ID."""
        cls = self._cluster_lines_by_poly.get(poly_id, [])
        return [UnifiedLine.from_cluster_line(cl) for cl in cls]
    
    def _get_region_line(self, line_id: str) -> List[UnifiedLine]:
        """Get a single region line by ID."""
        rl = self._region_line_by_id.get(line_id)
        if rl:
            return [UnifiedLine.from_region_line(rl)]
        return []
    
    def get_block_lines(
        self,
        block_id: str,
        block_type: str
    ) -> List[UnifiedLine]:
        """Get all lines for a block based on its type."""
        if block_type == "TABLE":
            # Get all cluster lines for the table's rectangles
            lines = []
            for cl in self.inputs.cluster_lines:
                if cl.poly_id and cl.poly_id.startswith(block_id):
                    lines.append(UnifiedLine.from_cluster_line(cl))
            return lines
        
        elif block_type in ("INFO_CLUSTER", "CLUSTER"):
            return self._get_cluster_lines(block_id)
        
        else:
            # Region-based block
            block = self._region_block_by_id.get(block_id)
            if block:
                lines = []
                for line_id in block.line_ids:
                    rl = self._region_line_by_id.get(line_id)
                    if rl:
                        lines.append(UnifiedLine.from_region_line(rl))
                return lines
        
        return []
    
    def get_rectangle_for_poly(self, poly_id: str) -> Optional[Rectangle]:
        """Get rectangle for a polygon ID."""
        return self._rectangle_by_poly.get(poly_id)
    
    def create_anchor_line(self, anchor: CandidateAnchor) -> UnifiedLine:
        """Create a UnifiedLine representing the anchor itself."""
        return UnifiedLine(
            id=f"anchor_{anchor.source_id}",
            text=anchor.anchor_text,
            text_norm=anchor.text_norm,
            bbox=anchor.bbox,
            tokens=list(anchor.tokens) if anchor.tokens else [],
            source_type="anchor",
            source_obj=anchor,  # type: ignore
            block_id=anchor.block_id,
        )


# TEXT PATTERN UTILITIES

KV_SPLIT_REGEX = re.compile(r"\s*[:=\u2013\u2014\-–]\s*")
ENUM_REGEX = re.compile(r"^\s*(?:[-•∙▪‣*]|\(?[0-9]+[\).]|[a-zA-Z][\).])\s+")


class TypeEngine:
    """Type detection for extracted values."""
    NUMERIC = re.compile(r"^-?\d+(\.\d+)?$")
    CURRENCY = re.compile(r"^[£$€]\s*\d+(\.\d+)?$")
    PERCENT = re.compile(r"^\d+(\.\d+)?%$")
    ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    MDY = re.compile(r"^\d{1,2}-\d{1,2}-\d{2,4}$")
    DMY = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")
    
    def is_numeric(self, t: str) -> bool:
        return bool(self.NUMERIC.match(t))
    
    def is_currency(self, t: str) -> bool:
        return bool(self.CURRENCY.match(t))
    
    def is_percent(self, t: str) -> bool:
        return bool(self.PERCENT.match(t))
    
    def is_date(self, t: str) -> bool:
        return bool(self.ISO_DATE.match(t) or self.MDY.match(t) or self.DMY.match(t))
    
    def validate(self, t: str) -> Optional[str]:
        if not t.strip():
            return None
        if self.is_numeric(t):
            return "numeric"
        if self.is_currency(t):
            return "currency"
        if self.is_percent(t):
            return "percent"
        if self.is_date(t):
            return "date"
        return None


# BASE RESOLVER CLASS

class BaseResolverV2:
    """Base class for all v2 resolvers. """
    
    name: str = "base"
    
    def empty_result(self, query: str, telemetry: Telemetry) -> Dict[str, Any]:
        """Return empty result structure."""
        return make_result(
            field=query,
            format_type=self.name,
            definition="",
            values=[],
            score=0.0,
            telemetry=telemetry,
            strategy=self.name,
            fallback_used=True,
        )
    
    def resolve(
        self,
        lines: List[UnifiedLine],
        query: str,
        anchors: List[ResolverAnchor],
        stats: SpatialStats,
        tol: Tolerances,
        block_type: str,
        geometry_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve relationships. Override in subclasses."""
        raise NotImplementedError


# TABLE RESOLVER

class TableResolverV2(BaseResolverV2):
    """Resolves values from table structures. """
    
    name = "table"
    
    def resolve(
        self,
        lines: List[UnifiedLine],
        query: str,
        anchors: List[ResolverAnchor],
        stats: SpatialStats,
        tol: Tolerances,
        block_type: str,
        geometry_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        telemetry = Telemetry()
        
        if not anchors:
            return self.empty_result(query, telemetry)
        
        anchor = anchors[0]
        anchor_line = geometry_context.get("anchor_line")
        if not anchor_line:
            # Create from anchor
            anchor_line = UnifiedLine(
                id=f"anchor_{anchor.source_id}",
                text=anchor.text,
                text_norm=anchor.anchor.text_norm,
                bbox=anchor.bbox,
                tokens=[],
                source_type="anchor",
                source_obj=anchor.anchor,
            )
        
        # Build neighbor graph
        graph = NeighborEngine.build_neighbor_graph(lines, tol)
        
        # Select orientation
        orientation = NeighborEngine.select_orientation(
            anchor_line, graph, stats, tol, telemetry
        )
        
        # Collect axis candidates
        axis_candidates = NeighborEngine.collect_axis_candidates(
            anchor_line, lines, tol, orientation, telemetry
        )
        
        # Extract values based on orientation
        values = []
        if orientation in (1, 3):  # Vertical
            # Values are below (1) or above (3) the anchor
            for ln in axis_candidates:
                if orientation == 1 and ln.y_mid > anchor_line.y_mid:
                    values.append(ln.text.strip())
                elif orientation == 3 and ln.y_mid < anchor_line.y_mid:
                    values.append(ln.text.strip())
        else:  # Horizontal
            # Values are right (2) or left (4) of the anchor
            for ln in axis_candidates:
                if orientation == 2 and ln.x_mid > anchor_line.x_mid:
                    values.append(ln.text.strip())
                elif orientation == 4 and ln.x_mid < anchor_line.x_mid:
                    values.append(ln.text.strip())
        
        # Compute score based on type consistency
        score = self._compute_type_score(values)
        
        return make_result(
            field=query,
            format_type="table",
            definition=anchor.text,
            values=values,
            score=score,
            telemetry=telemetry,
            strategy="table",
            orientation_case=orientation,
            line_ids=[ln.id for ln in axis_candidates],
        )
    
    def _compute_type_score(self, values: List[str]) -> float:
        """Compute score based on type consistency."""
        if not values:
            return 0.2
        
        type_engine = TypeEngine()
        type_votes = {"numeric": 0, "date": 0, "other": 0}
        
        for v in values:
            if type_engine.is_numeric(v) or type_engine.is_currency(v):
                type_votes["numeric"] += 1
            elif type_engine.is_date(v):
                type_votes["date"] += 1
            else:
                type_votes["other"] += 1
        
        total = len(values)
        majority = max(type_votes, key=type_votes.get)
        consistency = type_votes[majority] / total
        
        return 0.5 + 0.5 * consistency


# TABLE RECORD ASSEMBLER

@dataclass
class TableCell:
    """A cell in a table with position and value."""
    text: str
    bbox: BBox
    row_idx: int = -1
    col_idx: int = -1
    field_name: Optional[str] = None
    
    @property
    def x_mid(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2
    
    @property
    def y_mid(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2


@dataclass
class TableRecord:
    """A single row/record from a table."""
    row_idx: int
    fields: Dict[str, Optional[str]]
    row_y: float  # Y-coordinate for vertical tables, X for horizontal
    confidence: float = 1.0


class TableRecordAssembler:
    """Assembles table records by aligning values along the detected reading axis. """
    
    def __init__(
        self,
        orientation: int,
        tolerance: float = 15.0,
        page_height: float = 1000.0,
    ):
        """Initialize the assembler."""
        self.orientation = orientation
        self.tolerance = tolerance
        self.page_height = page_height
        
        # Storage for columns
        self.columns: Dict[str, List[TableCell]] = {}
        self.header_positions: Dict[str, BBox] = {}
        
        # Computed row positions
        self._row_positions: List[float] = []
        self._assembled = False
    
    @property
    def is_vertical(self) -> bool:
        """True if table has vertical reading direction (values below/above headers)."""
        return self.orientation in (1, 3)
    
    @property
    def is_horizontal(self) -> bool:
        """True if table has horizontal reading direction (values right/left of headers)."""
        return self.orientation in (2, 4)
    
    def add_column(
        self,
        field_name: str,
        header_bbox: BBox,
        value_cells: List[Tuple[str, BBox]],
    ) -> None:
        """Add a column of values for a field. """
        self.header_positions[field_name] = header_bbox
        
        cells = []
        for text, bbox in value_cells:
            cells.append(TableCell(
                text=text.strip(),
                bbox=bbox,
                field_name=field_name,
            ))
        
        self.columns[field_name] = cells
        self._assembled = False
    
    def add_column_from_lines(
        self,
        field_name: str,
        anchor_line: UnifiedLine,
        value_lines: List[UnifiedLine],
    ) -> None:
        """Add a column from UnifiedLine objects."""
        self.header_positions[field_name] = anchor_line.bbox
        
        cells = []
        for ln in value_lines:
            cells.append(TableCell(
                text=ln.text.strip(),
                bbox=ln.bbox,
                field_name=field_name,
            ))
        
        self.columns[field_name] = cells
        self._assembled = False
    
    def _compute_row_positions(self) -> List[float]:
        """Compute distinct row positions from all cells. """
        # Collect all position values with their sources
        positions = []
        for cells in self.columns.values():
            for cell in cells:
                if self.is_vertical:
                    positions.append(cell.y_mid)
                else:
                    positions.append(cell.x_mid)
        
        if not positions:
            return []
        
        # Sort positions
        positions.sort()
        
        # Compute adaptive tolerance based on median gap between positions
        if len(positions) >= 2:
            gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            gaps = [g for g in gaps if g > 1.0]  # Filter out near-zero gaps (same row)
            if gaps:
                median_gap = sorted(gaps)[len(gaps) // 2]
                # Use half the median gap as tolerance for clustering
                adaptive_tolerance = min(self.tolerance, median_gap * 0.4)
            else:
                adaptive_tolerance = self.tolerance
        else:
            adaptive_tolerance = self.tolerance
        
        # Cluster positions - compare against cluster CENTER
        clusters: List[List[float]] = []
        
        for pos in positions:
            # Find if this position belongs to an existing cluster
            assigned = False
            for cluster in clusters:
                cluster_center = sum(cluster) / len(cluster)
                if abs(pos - cluster_center) <= adaptive_tolerance:
                    cluster.append(pos)
                    assigned = True
                    break
            
            if not assigned:
                # Start new cluster
                clusters.append([pos])
        
        # Compute cluster centers
        row_positions = [sum(c) / len(c) for c in clusters]
        
        # Sort by reading order
        if self.orientation == 1:  # Down: top to bottom
            row_positions.sort()
        elif self.orientation == 3:  # Up: bottom to top
            row_positions.sort(reverse=True)
        elif self.orientation == 2:  # Right: left to right
            row_positions.sort()
        elif self.orientation == 4:  # Left: right to left
            row_positions.sort(reverse=True)
        
        return row_positions
    
    def _assign_row_indices(self) -> None:
        """Assign row indices to all cells based on position clustering."""
        self._row_positions = self._compute_row_positions()
        
        for cells in self.columns.values():
            for cell in cells:
                pos = cell.y_mid if self.is_vertical else cell.x_mid
                
                # Find closest row position
                min_dist = float('inf')
                best_idx = -1
                
                for idx, row_pos in enumerate(self._row_positions):
                    dist = abs(pos - row_pos)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx
                
                if min_dist <= self.tolerance:
                    cell.row_idx = best_idx
                else:
                    cell.row_idx = -1  # Orphan cell
    
    def assemble(self) -> List[TableRecord]:
        """Assemble all columns into aligned records. """
        if not self.columns:
            return []
        
        # Assign row indices to all cells
        self._assign_row_indices()
        
        # Build records
        records = []
        field_names = list(self.columns.keys())
        
        for row_idx, row_pos in enumerate(self._row_positions):
            fields = {}
            
            for field_name in field_names:
                # Find cell for this field at this row
                cell_value = None
                for cell in self.columns[field_name]:
                    if cell.row_idx == row_idx:
                        cell_value = cell.text if cell.text else None
                        break
                
                fields[field_name] = cell_value
            
            records.append(TableRecord(
                row_idx=row_idx,
                fields=fields,
                row_y=row_pos,
            ))
        
        self._assembled = True
        return records
    
    def assemble_as_dicts(self) -> List[Dict[str, Optional[str]]]:
        """Assemble and return as simple dictionaries."""
        records = self.assemble()
        return [r.fields for r in records]
    
    def get_column_values(self, field_name: str) -> List[Optional[str]]:
        """Get values for a single column in row order."""
        if not self._assembled:
            self.assemble()
        
        if field_name not in self.columns:
            return []
        
        values = [None] * len(self._row_positions)
        for cell in self.columns[field_name]:
            if 0 <= cell.row_idx < len(values):
                values[cell.row_idx] = cell.text if cell.text else None
        
        return values
    
    @classmethod
    def from_resolver_results(
        cls,
        results: Dict[str, Any],
        orientation: int,
        tolerance: float = 15.0,
    ) -> "TableRecordAssembler":
        """Create assembler from multiple resolver results."""
        assembler = cls(orientation=orientation, tolerance=tolerance)
        
        for field_name, result in results.items():
            if not result:
                continue
            
            values = result.get("values", [])
            bboxes = result.get("line_bboxes", [])
            anchor_bbox = result.get("anchor_bbox", (0, 0, 0, 0))
            
            if values and bboxes and len(values) == len(bboxes):
                value_cells = list(zip(values, bboxes))
                assembler.add_column(field_name, anchor_bbox, value_cells)
        
        return assembler


def assemble_table_records(
    table_id: str,
    anchor_results: List[Dict[str, Any]],
    geometry: GeometryReconstructor,
    consolidation_results: Dict[str, Any],
    stats: SpatialStats,
    tol: Tolerances,
) -> List[Dict[str, Optional[str]]]:
    """Convenience function to assemble table records from anchor results."""
    if not anchor_results:
        return []
    
    # Get all lines for the table
    first_anchor = anchor_results[0]
    anchor = CandidateAnchor(
        query_name=first_anchor["query_name"],
        anchor_text=first_anchor["anchor_text"],
        text_norm=first_anchor.get("text_norm", ""),
        score=first_anchor["score"],
        alias_matched=first_anchor.get("alias_matched", False),
        source_type=first_anchor["source_type"],
        source_id=first_anchor["source_id"],
        block_id=first_anchor["block_id"],
        bbox=tuple(first_anchor["bbox"]),
    )
    
    lines = geometry.get_lines_for_anchor(anchor, consolidation_results)
    
    if not lines:
        return []
    
    # Detect orientation from first anchor
    anchor_line = geometry.create_anchor_line(anchor)
    if not any(ln.id == anchor_line.id for ln in lines):
        lines.append(anchor_line)
    
    graph = NeighborEngine.build_neighbor_graph(lines, tol)
    telemetry = Telemetry()
    orientation = NeighborEngine.select_orientation(anchor_line, graph, stats, tol, telemetry)
    
    # Create assembler
    assembler = TableRecordAssembler(
        orientation=orientation,
        tolerance=tol.tau_y if orientation in (1, 3) else tol.tau_x,
    )
    
    # Process each anchor
    for ar in anchor_results:
        field_name = ar["query_name"]
        ar_anchor = CandidateAnchor(
            query_name=ar["query_name"],
            anchor_text=ar["anchor_text"],
            text_norm=ar.get("text_norm", ""),
            score=ar["score"],
            alias_matched=ar.get("alias_matched", False),
            source_type=ar["source_type"],
            source_id=ar["source_id"],
            block_id=ar["block_id"],
            bbox=tuple(ar["bbox"]),
        )
        
        ar_anchor_line = geometry.create_anchor_line(ar_anchor)
        
        # Collect axis candidates
        axis_candidates = NeighborEngine.collect_axis_candidates(
            ar_anchor_line, lines, tol, orientation, telemetry
        )
        
        # Filter to values only (not the header itself)
        value_lines = []
        for ln in axis_candidates:
            if ln.id == ar_anchor_line.id:
                continue
            
            if orientation == 1 and ln.y_mid > ar_anchor_line.y_mid:
                value_lines.append(ln)
            elif orientation == 3 and ln.y_mid < ar_anchor_line.y_mid:
                value_lines.append(ln)
            elif orientation == 2 and ln.x_mid > ar_anchor_line.x_mid:
                value_lines.append(ln)
            elif orientation == 4 and ln.x_mid < ar_anchor_line.x_mid:
                value_lines.append(ln)
        
        assembler.add_column_from_lines(field_name, ar_anchor_line, value_lines)
    
    return assembler.assemble_as_dicts()

# KEY-VALUE RESOLVER

class KeyValueResolverV2(BaseResolverV2):
    """Resolves key-value pairs."""
    
    name = "kv"
    
    def resolve(
        self,
        lines: List[UnifiedLine],
        query: str,
        anchors: List[ResolverAnchor],
        stats: SpatialStats,
        tol: Tolerances,
        block_type: str,
        geometry_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        telemetry = Telemetry()
        
        if not anchors:
            return self.empty_result(query, telemetry)
        
        anchor = anchors[0]
        anchor_text = anchor.text
        
        # Strategy 1: Inline K:V
        inline_result = self._find_inline_pair(anchor_text, query)
        if inline_result:
            key, val, score = inline_result
            return make_result(
                field=query,
                format_type="kv",
                definition=key,
                values=[val],
                score=score,
                telemetry=telemetry,
                strategy="inline_kv",
            )
        
        # Strategy 2: Two-band rectangle search
        anchor_line = geometry_context.get("anchor_line")
        if anchor_line and lines:
            band_hits = self._search_two_band(anchor_line, lines, tol)
            val_line = self._pick_nearest_value(anchor_line, band_hits)
            
            if val_line:
                telemetry.record("line_ids", [val_line.id])
                return make_result(
                    field=query,
                    format_type="kv",
                    definition=anchor_text,
                    values=[val_line.text.strip()],
                    score=0.68,
                    telemetry=telemetry,
                    strategy="two_band",
                )
        
        # No value found
        return make_result(
            field=query,
            format_type="kv",
            definition=anchor_text,
            values=[],
            score=0.2,
            telemetry=telemetry,
            strategy="kv",
            fallback_used=True,
        )
    
    def _find_inline_pair(
        self, text: str, query: str
    ) -> Optional[Tuple[str, str, float]]:
        """Find inline key:value pair."""
        parts = KV_SPLIT_REGEX.split(text, maxsplit=1)
        if len(parts) < 2:
            return None
        
        key_raw = parts[0].strip()
        val_raw = parts[1].strip()
        
        if not key_raw or not val_raw:
            return None
        
        # Fuzzy match key to query
        score = fuzz.token_set_ratio(key_raw.lower(), query.lower()) / 100.0
        if score < 0.55:
            return None
        
        return key_raw, val_raw, score
    
    def _search_two_band(
        self,
        anchor: UnifiedLine,
        lines: List[UnifiedLine],
        tol: Tolerances
    ) -> List[UnifiedLine]:
        """Search within horizontal bands for values."""
        y0 = anchor.y0 - tol.tau_y
        y1 = anchor.y1 + tol.tau_y
        
        hits = []
        for ln in lines:
            if ln.id == anchor.id:
                continue
            if ln.y1 >= y0 and ln.y0 <= y1:
                hits.append(ln)
        
        return hits
    
    def _pick_nearest_value(
        self,
        anchor: UnifiedLine,
        candidates: List[UnifiedLine]
    ) -> Optional[UnifiedLine]:
        """Pick nearest value to the right or left."""
        right = []
        left = []
        
        for ln in candidates:
            if ln.x0 >= anchor.x1:
                dist = ln.x0 - anchor.x1
                right.append((dist, ln))
            elif ln.x1 <= anchor.x0:
                dist = anchor.x0 - ln.x1
                left.append((dist, ln))
        
        right.sort(key=lambda x: x[0])
        left.sort(key=lambda x: x[0])
        
        if right and left:
            return right[0][1] if right[0][0] <= left[0][0] else left[0][1]
        if right:
            return right[0][1]
        if left:
            return left[0][1]
        
        return None

# LIST RESOLVER

class ListResolverV2(BaseResolverV2):
    """Resolves list structures."""
    
    name = "list"
    
    def resolve(
        self,
        lines: List[UnifiedLine],
        query: str,
        anchors: List[ResolverAnchor],
        stats: SpatialStats,
        tol: Tolerances,
        block_type: str,
        geometry_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        telemetry = Telemetry()
        
        if not anchors:
            return self.empty_result(query, telemetry)
        
        anchor = anchors[0]
        anchor_line = geometry_context.get("anchor_line")
        
        # Check for inline K:V override
        inline = self._inline_kv(anchor.text, query)
        if inline:
            key, val, score = inline
            return make_result(
                field=query,
                format_type="list",
                definition=key,
                values=[val],
                score=score,
                telemetry=telemetry,
                strategy="inline_kv",
            )
        
        # Collect list block
        if anchor_line and lines:
            block = self._collect_list_block(anchor_line, lines, stats, tol)
            telemetry.record("line_ids", [ln.id for ln in block])
            
            if block:
                values = []
                for ln in block:
                    # Remove bullet/enumeration marker
                    clean = ENUM_REGEX.sub("", ln.text).strip()
                    if clean:
                        values.append(clean)
                
                return make_result(
                    field=query,
                    format_type="list",
                    definition=anchor.text,
                    values=values,
                    score=0.72,
                    telemetry=telemetry,
                    strategy="list_block",
                )
        
        # Fallback: use anchor text as value
        return make_result(
            field=query,
            format_type="list",
            definition=anchor.text,
            values=[anchor.text.strip()],
            score=0.55,
            telemetry=telemetry,
            strategy="list",
            fallback_used=True,
        )
    
    def _inline_kv(
        self, text: str, query: str
    ) -> Optional[Tuple[str, str, float]]:
        """Check for inline key:value."""
        if ":" not in text:
            return None
        
        parts = text.split(":", 1)
        key = parts[0].strip()
        val = parts[1].strip()
        
        score = fuzz.token_set_ratio(key.lower(), query.lower()) / 100.0
        if score < 0.55:
            return None
        
        return key, val, score
    
    def _collect_list_block(
        self,
        anchor: UnifiedLine,
        lines: List[UnifiedLine],
        stats: SpatialStats,
        tol: Tolerances
    ) -> List[UnifiedLine]:
        """Collect lines that are part of the list below anchor."""
        # Sort lines by vertical position
        sorted_lines = sorted(lines, key=lambda ln: ln.y0)
        
        # Find anchor index
        anchor_idx = None
        for idx, ln in enumerate(sorted_lines):
            if ln.id == anchor.id:
                anchor_idx = idx
                break
        
        if anchor_idx is None:
            return []
        
        anchor_indent = self._get_indent(anchor)
        collected = []
        last_y = anchor.y1
        
        for ln in sorted_lines[anchor_idx + 1:]:
            # Stop on large gap
            gap = ln.y0 - last_y
            if gap > tol.max_gap:
                break
            
            ln_indent = self._get_indent(ln)
            is_enum = bool(ENUM_REGEX.match(ln.text))
            
            # Stop at next main enumerated line at same or lower indent
            if is_enum and ln_indent <= anchor_indent:
                break
            
            # Include if enumerated subitem, indented, or close continuation
            if is_enum or ln_indent > anchor_indent or gap <= stats.median_vgap * 1.4:
                collected.append(ln)
                last_y = ln.y1
            else:
                break
        
        return collected
    
    def _get_indent(self, line: UnifiedLine) -> float:
        """Get indentation level of a line."""
        return line.x0


# PARAGRAPH RESOLVER

class ParagraphResolverV2(BaseResolverV2):
    """Resolves paragraph structures."""
    
    name = "paragraph"
    
    def resolve(
        self,
        lines: List[UnifiedLine],
        query: str,
        anchors: List[ResolverAnchor],
        stats: SpatialStats,
        tol: Tolerances,
        block_type: str,
        geometry_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        telemetry = Telemetry()
        
        if not anchors:
            return self.empty_result(query, telemetry)
        
        anchor = anchors[0]
        anchor_line = geometry_context.get("anchor_line")
        
        # Check for inline K:V override
        inline = self._find_inline_kv(anchor.text, query)
        if inline:
            key, val = inline
            return make_result(
                field=query,
                format_type="paragraph",
                definition=f"{key}: {val}",
                values=[val],
                score=anchor.score,
                telemetry=telemetry,
                strategy="inline_kv",
            )
        
        # Build paragraph block
        if anchor_line and lines:
            block = self._build_paragraph_block(anchor_line, lines, stats, tol)
            telemetry.record("line_ids", [ln.id for ln in block])
            
            # Join wrapped lines
            joined_text = " ".join(ln.text.strip() for ln in block)
            
            return make_result(
                field=query,
                format_type="paragraph",
                definition=joined_text,
                values=[joined_text],
                score=anchor.score,
                telemetry=telemetry,
                strategy="paragraph_block",
            )
        
        # Fallback
        return make_result(
            field=query,
            format_type="paragraph",
            definition=anchor.text,
            values=[anchor.text],
            score=anchor.score,
            telemetry=telemetry,
            strategy="paragraph",
            fallback_used=True,
        )
    
    def _find_inline_kv(
        self, text: str, query: str
    ) -> Optional[Tuple[str, str]]:
        """Find inline key:value."""
        parts = KV_SPLIT_REGEX.split(text, maxsplit=1)
        if len(parts) < 2:
            return None
        
        key = parts[0].strip()
        val = parts[1].strip()
        
        if not key or not val:
            return None
        
        score = fuzz.token_set_ratio(key.lower(), query.lower()) / 100.0
        if score < 0.55:
            return None
        
        return key, val
    
    def _build_paragraph_block(
        self,
        anchor: UnifiedLine,
        lines: List[UnifiedLine],
        stats: SpatialStats,
        tol: Tolerances
    ) -> List[UnifiedLine]:
        """Build paragraph block around anchor."""
        # Find anchor in lines
        try:
            anchor_idx = next(i for i, ln in enumerate(lines) if ln.id == anchor.id)
        except StopIteration:
            return [anchor]
        
        block = [anchor]
        
        # Scan upward
        i = anchor_idx - 1
        while i >= 0:
            if not self._is_continuation(lines[i], block[0], stats, tol):
                break
            block.insert(0, lines[i])
            i -= 1
        
        # Scan downward
        i = anchor_idx + 1
        while i < len(lines):
            if not self._is_continuation(lines[i], block[-1], stats, tol):
                break
            block.append(lines[i])
            i += 1
        
        return block
    
    def _is_continuation(
        self,
        candidate: UnifiedLine,
        reference: UnifiedLine,
        stats: SpatialStats,
        tol: Tolerances
    ) -> bool:
        """Check if candidate continues the paragraph."""
        # Large vertical gap
        dy = abs(candidate.y0 - reference.y1)
        if dy > 1.8 * stats.median_vgap:
            return False
        
        # Enumeration boundary
        if ENUM_REGEX.match(candidate.text):
            return False
        
        # Large indentation change
        indent_diff = abs(candidate.x0 - reference.x0)
        if indent_diff > 2.5 * stats.median_hgap:
            return False
        
        return True

# CLUSTER RESOLVER

class ClusterResolverV2(BaseResolverV2):
    """Resolves cluster structures (INFO_CLUSTER)."""
    
    name = "cluster"
    
    def resolve(
        self,
        lines: List[UnifiedLine],
        query: str,
        anchors: List[ResolverAnchor],
        stats: SpatialStats,
        tol: Tolerances,
        block_type: str,
        geometry_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        telemetry = Telemetry()
        
        if not anchors:
            return self.empty_result(query, telemetry)
        
        anchor = anchors[0]
        anchor_line = geometry_context.get("anchor_line")
        
        if not anchor_line:
            return self.empty_result(query, telemetry)
        
        # Use NeighborEngine to find spatial neighbors
        graph = NeighborEngine.build_neighbor_graph(lines, tol)
        info = graph.get(anchor_line.id, NeighborInfo())
        
        telemetry.record("neighbors_right", len(info.right))
        telemetry.record("neighbors_down", len(info.down))
        
        # Strategy 1: Look for value to the right (horizontal label-value pair)
        if info.right:
            value_line = info.right[0][0]  # Nearest right neighbor
            telemetry.record("strategy", "right_neighbor")
            return make_result(
                field=query,
                format_type="cluster",
                definition=anchor.text,
                values=[value_line.text.strip()],
                score=anchor.score,
                telemetry=telemetry,
                strategy="cluster_right",
            )
        
        # Strategy 2: Look for value below (vertical label-value pair)
        if info.down:
            value_line = info.down[0][0]  # Nearest down neighbor
            telemetry.record("strategy", "down_neighbor")
            return make_result(
                field=query,
                format_type="cluster",
                definition=anchor.text,
                values=[value_line.text.strip()],
                score=anchor.score,
                telemetry=telemetry,
                strategy="cluster_down",
            )
        
        # Fallback to KV-style resolution
        return self._fallback_kv(query, anchor, lines, tol, telemetry)
    
    def _fallback_kv(
        self,
        query: str,
        anchor: ResolverAnchor,
        lines: List[UnifiedLine],
        tol: Tolerances,
        telemetry: Telemetry
    ) -> Dict[str, Any]:
        """Fallback to KV-style resolution."""
        telemetry.record("fallback", "kv")
        
        # Try inline KV
        parts = KV_SPLIT_REGEX.split(anchor.text, maxsplit=1)
        if len(parts) >= 2:
            val = parts[1].strip()
            if val:
                return make_result(
                    field=query,
                    format_type="cluster",
                    definition=anchor.text,
                    values=[val],
                    score=0.6,
                    telemetry=telemetry,
                    strategy="cluster_fallback_kv",
                )
        
        return make_result(
            field=query,
            format_type="cluster",
            definition=anchor.text,
            values=[],
            score=0.2,
            telemetry=telemetry,
            strategy="cluster",
            fallback_used=True,
        )


# STRING RESOLVER

class StringResolverV2(BaseResolverV2):
    """Resolves single-line string fields."""
    
    name = "string"
    
    def resolve(
        self,
        lines: List[UnifiedLine],
        query: str,
        anchors: List[ResolverAnchor],
        stats: SpatialStats,
        tol: Tolerances,
        block_type: str,
        geometry_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        telemetry = Telemetry()
        
        if not anchors:
            return self.empty_result(query, telemetry)
        
        anchor = anchors[0]
        text = anchor.text.strip()
        
        # Strategy 1: Inline K:V
        parts = KV_SPLIT_REGEX.split(text, maxsplit=1)
        if len(parts) >= 2:
            key = parts[0].strip()
            val = parts[1].strip()
            if val:
                return make_result(
                    field=query,
                    format_type="string",
                    definition=key,
                    values=[val],
                    score=anchor.score,
                    telemetry=telemetry,
                    strategy="inline_kv",
                )
        
        # Strategy 2: Forward typed token search
        type_engine = TypeEngine()
        tokens = text.split()
        for tok in tokens:
            if type_engine.validate(tok):
                return make_result(
                    field=query,
                    format_type="string",
                    definition=text,
                    values=[tok],
                    score=anchor.score,
                    telemetry=telemetry,
                    strategy="typed_token",
                )
        
        # Strategy 3: Whole line as value
        return make_result(
            field=query,
            format_type="string",
            definition=text,
            values=[text],
            score=anchor.score,
            telemetry=telemetry,
            strategy="string",
            fallback_used=True,
        )


# RESOLVER REGISTRY

RESOLVER_REGISTRY: Dict[str, type] = {
    "TABLE": TableResolverV2,
    "KEY_VALUE": KeyValueResolverV2,
    "LIST": ListResolverV2,
    "PARAGRAPH": ParagraphResolverV2,
    "INFO_CLUSTER": ClusterResolverV2,
    "CLUSTER": ClusterResolverV2,
    "STRING": StringResolverV2,
}

# Label priority for tie-breaking (lower = higher priority)
LABEL_PRIORITY = {
    "KEY_VALUE": 0,
    "TABLE": 1,
    "LIST": 2,
    "PARAGRAPH": 3,
    "INFO_CLUSTER": 4,
    "CLUSTER": 5,
    "STRING": 6,
}


# RELATIONSHIP ENGINE V2 - Main Orchestrator

class RelationshipEngineV2:
    """V2 Relationship Engine - Consumes data_cleaningv2 output."""
    
    # Platt calibration parameters per format
    FORMAT_PLATT_PARAMS = {
        "table": (1.5, -0.5),
        "kv": (1.7, -0.6),
        "list": (1.4, -0.4),
        "paragraph": (1.3, -0.4),
        "string": (1.2, -0.3),
        "cluster": (1.5, -0.5),
    }
    
    def __init__(
        self,
        inputs: DataCleaningInputsV2,
        classification_results: Dict[str, Any],
        consolidation_results: Optional[Dict[str, Any]] = None,
        config: ResolverConfigV2 = DEFAULT_CONFIG,
        # Entity-aware components (R2-R6)
        entities: List["Entity"] = None,
        proto_entities: List["ResolvedEntity"] = None,
        rule_pack: "RulePack" = None,
        scoring_weights: ScoringWeights = None,
        stats_pack: Any = None,  # StatsPack for co-occurrence and OCR adjustment
        region_mapping: Dict[str, str] = None,  # Explicit region ID mapping
    ):
        """Initialize the engine. """
        self.inputs = inputs
        self.classification_results = classification_results
        self.consolidation_results = consolidation_results
        self.config = config
        self.stats_pack = stats_pack
        
        # Build geometry reconstructor
        self.geometry = GeometryReconstructor(inputs)
        
        # Pre-compute spatial stats
        self.stats = SpatialStatisticsEngine.compute_from_raw(
            inputs.region_lines, inputs.cluster_lines
        )
        self.tolerances = ToleranceEngine.compute(self.stats, config)
        
        # Classification maps
        self.cluster_classes = classification_results.get("clusters", {})
        self.block_classes = classification_results.get("blocks", {})
        self.standalone_classes = classification_results.get("standalone_lines", {})
        
        # Entity-aware components (R2-R6)
        self.entity_overlay = None
        self.query_interpreter = None
        self.relationship_scorer = None
        self.entity_templates = None
        self.graph_exporter = None
        
        if BOOTSTRAP_AVAILABLE:
            # R2 - Entity Overlay with region mapping support
            self.entity_overlay = EntityOverlay(
                entities=entities,
                proto_entities=proto_entities,
                rule_pack=rule_pack,
                region_mapping=region_mapping,
            )
            
            # R2.6 - Relationship Scorer with StatsPack integration
            self.relationship_scorer = RelationshipScorer(
                weights=scoring_weights or DEFAULT_SCORING_WEIGHTS,
                distance_decay_px=config.distance_decay_px,
                stats_pack=stats_pack,
            )
            
            # R3 - Query Interpreter
            self.query_interpreter = QueryInterpreter(rule_pack=rule_pack)
            
            # R4 - Entity Templates
            self.entity_templates = EntityTemplates(
                entity_overlay=self.entity_overlay,
                scorer=self.relationship_scorer,
            )
            
            # R6 - Graph Exporter
            self.graph_exporter = GraphExporter(
                scorer=self.relationship_scorer,
                min_score=0.3,
            )
    
    def resolve_anchor(
        self,
        anchor: CandidateAnchor,
        query: str,
    ) -> Optional[RelationshipResult]:
        """Resolve a single anchor to extract its associated value."""
        # Determine block type from classification
        block_type = self._get_block_type(anchor)
        
        # Get resolver for this block type
        resolver_cls = RESOLVER_REGISTRY.get(block_type)
        if not resolver_cls:
            # Default to string resolver
            resolver_cls = StringResolverV2
        
        resolver = resolver_cls()
        
        # Get lines for this anchor's context
        # Pass consolidation_results for table anchors to get all cells
        lines = self.geometry.get_lines_for_anchor(anchor, self.consolidation_results)
        
        # Create anchor line
        anchor_line = self.geometry.create_anchor_line(anchor)
        
        # Add anchor line to lines if not present
        if not any(ln.id == anchor_line.id for ln in lines):
            lines.append(anchor_line)
        
        # Build geometry context
        geometry_context = {
            "anchor_line": anchor_line,
            "block_id": anchor.block_id,
            "source_type": anchor.source_type,
        }
        
        # Wrap anchor
        resolver_anchor = ResolverAnchor(anchor=anchor, line=anchor_line)
        
        # Resolve
        result = resolver.resolve(
            lines=lines,
            query=query,
            anchors=[resolver_anchor],
            stats=self.stats,
            tol=self.tolerances,
            block_type=block_type,
            geometry_context=geometry_context,
        )
        
        if not result:
            return None
        
        # Calibrate score
        raw_score = result.get("score", 0.0)
        fmt = result.get("format", "string")
        calibrated_score = self._calibrate_score(fmt, raw_score)
        
        # Build meta
        meta = dict(result.get("meta", {}))
        meta["block_type"] = block_type
        meta["source_type"] = anchor.source_type
        meta["block_id"] = anchor.block_id
        meta["score_raw"] = raw_score
        meta["score_calibrated"] = calibrated_score
        meta["anchor_score"] = anchor.score
        
        return RelationshipResult(
            field=query,
            format=fmt,
            definition=result.get("definition", ""),
            values=result.get("values", []),
            score=calibrated_score,
            meta=meta,
        )
    
    # Max anchors to resolve per query so duplicate-resolution step stays finite
    MAX_ANCHORS_PER_QUERY = 20

    def _result_selection_key(self, result: Optional[RelationshipResult]) -> Tuple[bool, float, int]:
        """Key for choosing the best result among duplicates: prefer most high-confidence results.
        Sort descending: (has_values, score, num_values) so result with values and higher score wins.
        """
        if not result:
            return (False, 0.0, 0)
        has_values = len(result.values) > 0
        return (has_values, result.score, len(result.values))

    def resolve_query(
        self,
        query: str,
        anchors: List[CandidateAnchor],
    ) -> Optional[RelationshipResult]:
        """Resolve a query using the best anchor.
        
        When multiple anchors exist for the same query on one page (e.g. REV in two places),
        returns the single result that has the most relative high-confidence results:
        prefer result with values over none, then higher score, then more values.
        
        Args:
            query: Query field name
            anchors: List of CandidateAnchor objects for this query
        
        Returns:
            Best RelationshipResult or None
        """
        if not anchors:
            return None
        
        # Sort anchors by score (descending) so we try best matches first
        sorted_anchors = sorted(anchors, key=lambda a: -a.score)
        cap = min(len(sorted_anchors), self.MAX_ANCHORS_PER_QUERY)
        candidates: List[Optional[RelationshipResult]] = []
        for anchor in sorted_anchors[:cap]:
            result = self.resolve_anchor(anchor, query)
            candidates.append(result)
        # Choose the result with the best selection key (has values > score > num values)
        best = max(candidates, key=self._result_selection_key)
        return best
    
    def resolve_all(
        self,
        anchors_by_query: Dict[str, List[CandidateAnchor]],
    ) -> Dict[str, Optional[RelationshipResult]]:
        """Resolve all queries."""
        results = {}
        for query, anchors in anchors_by_query.items():
            results[query] = self.resolve_query(query, anchors)
        return results
    
    def _get_block_type(self, anchor: CandidateAnchor) -> str:
        """Determine block type from anchor and classification maps."""
        source_type = anchor.source_type
        block_id = anchor.block_id
        
        # Table cells are always TABLE
        if source_type == "table_cell":
            return "TABLE"
        
        # Check cluster classification
        if source_type == "cluster" or block_id in self.cluster_classes:
            cluster_info = self.cluster_classes.get(block_id, {})
            label = cluster_info.get("label", "INFO_CLUSTER")
            return label
        
        # Check standalone line classification
        if source_type in ("region_line", "standalone"):
            standalone_info = self.standalone_classes.get(block_id, {})
            if standalone_info:
                return standalone_info.get("label", "STRING")
        
        # Check block classification
        block_info = self.block_classes.get(block_id, {})
        if block_info:
            return block_info.get("label", "STRING")
        
        # Map source_type to block_type as fallback
        source_to_block = {
            "list": "LIST",
            "paragraph": "PARAGRAPH",
            "key_value": "KEY_VALUE",
            "string": "STRING",
        }
        return source_to_block.get(source_type, "STRING")
    
    def _calibrate_score(self, fmt: str, raw: float) -> float:
        """Apply Platt calibration to raw score."""
        params = self.FORMAT_PLATT_PARAMS.get(fmt.lower())
        if not params:
            return max(0.0, min(1.0, raw))
        
        a, b = params
        z = a * raw + b
        
        if z >= 0.0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)
    
    # ENTITY-AWARE METHODS (R2-R6 Integration)
    
    def resolve_with_entities(
        self,
        query: str,
        anchors: List[CandidateAnchor] = None,
        min_score: float = 0.5,
    ) -> Optional[ResolvedRegion]:
        """
        Resolve a query using entity-aware pipeline."""
        if not BOOTSTRAP_AVAILABLE or not self.query_interpreter:
            # Fallback to legacy resolution
            if anchors:
                result = self.resolve_query(query, anchors)
                if result:
                    return ResolvedRegion(
                        region_id=result.meta.get("block_id", ""),
                        page_id="",
                        format_type=result.format,
                        text=result.definition,
                        values=result.values,
                        entities=[],
                        score=result.score,
                        metadata=result.meta,
                    )
            return None
        
        # R3 - Interpret query to entity anchors
        entity_anchors, unmatched = self.query_interpreter.interpret_with_fallback(query)
        
        if entity_anchors:
            # Try entity-based resolution
            for anchor in entity_anchors:
                if anchor.entity_type == "TAG" and anchor.canonical_value:
                    # Use equipment spec template
                    result = self.entity_templates.get_equipment_spec(
                        anchor.canonical_value,
                        min_score=min_score,
                    )
                    if result:
                        return result
                
                elif anchor.entity_type == "MATERIAL" and anchor.canonical_value:
                    # Use parts by material template
                    results = self.entity_templates.get_parts_by_material(
                        anchor.canonical_value,
                        min_score=min_score,
                    )
                    if results:
                        return results[0]
                
                elif anchor.entity_type == "ACTION":
                    # Find related TAGs
                    tag_anchors = [a for a in entity_anchors if a.entity_type == "TAG"]
                    if tag_anchors:
                        results = self.entity_templates.get_actions_for_tag(
                            tag_anchors[0].canonical_value,
                            min_score=min_score,
                        )
                        if results:
                            return results[0]
        
        # Fallback to legacy resolution with unmatched terms
        if anchors:
            result = self.resolve_query(query, anchors)
            if result:
                # Attach entities from overlay if available
                entities = []
                if self.entity_overlay and result.meta.get("block_id"):
                    entities = self.entity_overlay.get_entities_for_region(
                        result.meta["block_id"]
                    )
                
                return ResolvedRegion(
                    region_id=result.meta.get("block_id", ""),
                    page_id="",
                    format_type=result.format,
                    text=result.definition,
                    values=result.values,
                    entities=entities,
                    score=result.score,
                    metadata=result.meta,
                )
        
        return None
    
    def get_entities_for_anchor(
        self,
        anchor: CandidateAnchor,
        entity_types: List[str] = None,
        min_confidence: float = 0.0,
    ) -> List["Entity"]:
        """ Get Module B entities for an anchor's region."""
        if not self.entity_overlay:
            return []
        
        return self.entity_overlay.get_entities_for_region(
            region_id=anchor.block_id,
            entity_types=entity_types,
            min_confidence=min_confidence,
        )
    
    def build_context_graph_for_anchor(
        self,
        anchor: CandidateAnchor,
        min_score: float = 0.3,
    ) -> Optional[ContextGraph]:
        """ Build a context graph for an anchor's region. """
        if not self.entity_overlay or not self.graph_exporter:
            return None
        
        entities = self.get_entities_for_anchor(anchor)
        if not entities:
            return None
        
        lines = self.geometry.get_lines_for_anchor(anchor, self.consolidation_results)
        
        return self.graph_exporter.build_context_graph(
            entities=entities,
            lines=lines,
            region_id=anchor.block_id,
        )
    
    def export_entity_graph(
        self,
        anchors: List[CandidateAnchor] = None,
        min_score: float = 0.3,
    ) -> Dict[str, Any]:
        """Export entity relationship graph for all anchors. """
        if not self.graph_exporter:
            return {"edges": [], "count": 0}
        
        all_edges = []
        
        if anchors:
            for anchor in anchors:
                graph = self.build_context_graph_for_anchor(anchor, min_score)
                if graph:
                    edges = self.graph_exporter.build_edges_from_context_graph(graph)
                    all_edges.extend(edges)
        elif self.entity_overlay:
            # Get all entities and build one big graph
            all_entities = []
            for entity_type in ["TAG", "PART", "MATERIAL", "SIZE", "ACTION"]:
                all_entities.extend(self.entity_overlay.get_entities_by_type(entity_type))
            
            if all_entities:
                edges = self.graph_exporter.build_edges_from_entities(all_entities)
                all_edges.extend(edges)
        
        return self.graph_exporter.export_to_dict(all_edges)
    
    def interpret_query(self, query: str) -> Tuple[List[EntityAnchor], List[str]]:
        """Interpret a query string into entity anchors."""
        if not self.query_interpreter:
            return [], query.split()
        
        return self.query_interpreter.interpret_with_fallback(query)


# CONVENIENCE FUNCTIONS

def resolve_relationships_v2(
    inputs: DataCleaningInputsV2,
    classification_results: Dict[str, Any],
    anchor_results: Dict[str, Any],
    consolidation_results: Optional[Dict[str, Any]] = None,
    config: Optional[ResolverConfigV2] = None,
    proto_entities: Optional[List["ResolvedEntity"]] = None,
    entities: Optional[List["Entity"]] = None,
    rule_pack: Optional["RulePack"] = None,
    stats_pack: Any = None,
) -> Dict[str, Optional[RelationshipResult]]:
    """Convenience function to resolve all relationships."""
    engine = RelationshipEngineV2(
        inputs=inputs,
        classification_results=classification_results,
        consolidation_results=consolidation_results,
        config=config or DEFAULT_CONFIG,
        proto_entities=proto_entities,
        entities=entities,
        rule_pack=rule_pack,
        stats_pack=stats_pack,
    )
    
    # Convert anchor dicts to CandidateAnchor objects
    anchors_by_query: Dict[str, List[CandidateAnchor]] = {}
    
    for query_name, anchor_dicts in anchor_results.get("by_query", {}).items():
        anchors = []
        for ad in anchor_dicts:
            anchors.append(CandidateAnchor(
                query_name=ad["query_name"],
                anchor_text=ad["anchor_text"],
                text_norm=ad["text_norm"],
                score=ad["score"],
                alias_matched=ad["alias_matched"],
                source_type=ad["source_type"],
                source_id=ad["source_id"],
                block_id=ad["block_id"],
                bbox=tuple(ad["bbox"]),
                fuzz_score=ad.get("fuzz_score", 0.0),
                cosine_score=ad.get("cosine_score", 0.0),
            ))
        anchors_by_query[query_name] = anchors
    
    return engine.resolve_all(anchors_by_query)


def create_resolver_for_block_type(block_type: str) -> BaseResolverV2:
    """Create a resolver instance for a given block type."""
    resolver_cls = RESOLVER_REGISTRY.get(block_type, StringResolverV2)
    return resolver_cls()

