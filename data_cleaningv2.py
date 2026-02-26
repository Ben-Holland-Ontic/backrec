from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any, Set
from collections import defaultdict
from pathlib import Path

import logging
import math
import os
import re
import unicodedata

import numpy as np
import cv2
import yaml
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Shared type aliases

BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]

# Aggregate container returned by process_input -> data_cleaning v2

@dataclass
class DataCleaningInputsV2:
    """Structured payload handed to the rewritten data-cleaning pipeline."""

    region_lines: List[RegionLine]
    region_blocks: List[RegionBlock]
    cluster_lines: List[ClusterLine]
    rectangles: List[Rectangle]
    contours: List[Contour]
    page: Optional[PageMetadata]
    cluster_blocks: List[Dict[str, Any]] = field(default_factory=list)
    cluster_features: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    region_block_features: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    region_line_features: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def by_block(self) -> Dict[str, List[RegionLine]]:
        """Utility helper: group RegionLines by block for quick lookup."""

        lookup: Dict[str, List[RegionLine]] = {}
        for line in self.region_lines:
            if not line.block_id:
                continue
            bucket = lookup.setdefault(line.block_id, [])
            bucket.append(line)
        return lookup

# OCR ingestion helpers

def _coerce_bbox(raw_bbox: Any) -> BBox:
    if not raw_bbox or len(raw_bbox) != 4:
        return (0.0, 0.0, 0.0, 0.0)
    x1, y1, x2, y2 = [float(v) for v in raw_bbox]
    if x2 <= x1 or y2 <= y1:
        # Treat as (x, y, w, h)
        x2 = x1 + abs(x2)
        y2 = y1 + abs(y2)
    return (x1, y1, x2, y2)


def _tokenize(text: str) -> List[str]:
    return [tok for tok in re.split(r"\s+", text.strip()) if tok]


def _bbox_area(bbox: BBox) -> float:
    """Compute area of a bounding box."""
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _text_density(text: str, bbox: BBox) -> float:
    area = max(1.0, _bbox_area(bbox))
    return float(len(text)) / area


def _ingest_region_payload(region_results: Optional[Dict[str, Any]]) -> Tuple[List[RegionLine], List[RegionBlock]]:
    if not isinstance(region_results, dict):
        return [], []

    raw_lines = region_results.get("Region_Line") or region_results.get("lines") or []
    raw_blocks = region_results.get("Region_Block") or []

    region_lines: List[RegionLine] = []
    region_blocks: List[RegionBlock] = []

    for entry in raw_lines:
        line_id = entry.get("line_id") or entry.get("id")
        if not line_id:
            continue
        bbox = _coerce_bbox(entry.get("bbox") or (0, 0, 0, 0))
        text = entry.get("text_raw") or entry.get("text") or ""
        norm_payload = _TEXT_NORMALIZER.normalize(text)
        tokens = _tokenize(norm_payload["text_norm"])
        region_lines.append(
            RegionLine(
                id=line_id,
                text_raw=norm_payload["text_raw"],
                text_norm=norm_payload["text_norm"],
                bbox=bbox,
                block_id=entry.get("block_id"),
                tokens=tokens,
                font_size=bbox[3] - bbox[1],
                text_density=_text_density(norm_payload["text_norm"], bbox),
            )
        )

    line_ids_per_block: Dict[str, List[str]] = defaultdict(list)
    for line in region_lines:
        if line.block_id:
            line_ids_per_block[line.block_id].append(line.id)

    for entry in raw_blocks:
        block_id = entry.get("block_id") or entry.get("id")
        if not block_id:
            continue
        bbox = _coerce_bbox(entry.get("bbox") or (0, 0, 0, 0))
        text = entry.get("text_raw") or entry.get("text") or ""
        norm_payload = _TEXT_NORMALIZER.normalize(text)
        density = _text_density(norm_payload["text_norm"], bbox)
        region_blocks.append(
            RegionBlock(
                id=block_id,
                bbox=bbox,
                line_ids=line_ids_per_block.get(block_id, []),
                density=density,
                structure_type=entry.get("structure_type"),
                alignment_type=entry.get("alignment_type"),
            )
        )

    return region_lines, region_blocks


def _coerce_contour_points(raw_pts: Any) -> List[Point]:
    if raw_pts is None:
        return []
    pts: List[Point] = []
    for pt in raw_pts:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            pts.append((float(pt[0]), float(pt[1])))
        elif hasattr(pt, "tolist"):
            coords = pt.tolist()
            if isinstance(coords, (list, tuple)):
                if isinstance(coords[0], (list, tuple)):
                    coords = coords[0]
                if len(coords) >= 2:
                    pts.append((float(coords[0]), float(coords[1])))
    return pts


def _ingest_cluster_payload(
    cluster_results: Optional[Dict[str, Any]],
    rect_data: Optional[Dict[str, Any]],
) -> Tuple[List[ClusterLine], List[Rectangle], List[Contour], List[Dict[str, Any]]]:
    cluster_lines: List[ClusterLine] = []
    rectangles: List[Rectangle] = []
    contours: List[Contour] = []
    cluster_blocks: List[Dict[str, Any]] = []

    if isinstance(cluster_results, dict):
        for entry in cluster_results.get("Cluster_Line", []):
            line_id = entry.get("line_id") or entry.get("id")
            if not line_id:
                continue
            bbox = _coerce_bbox(entry.get("bbox") or (0, 0, 0, 0))
            text = entry.get("text_raw") or entry.get("text") or ""
            norm_payload = _TEXT_NORMALIZER.normalize(text)
            cluster_lines.append(
                ClusterLine(
                    id=line_id,
                    text_raw=norm_payload["text_raw"],
                    text_norm=norm_payload["text_norm"],
                    bbox=bbox,
                    poly_id=entry.get("poly_id"),
                    tokens=_tokenize(norm_payload["text_norm"]),
                    text_density=_text_density(norm_payload["text_norm"], bbox),
                    col_index=entry.get("col_index"),
                )
            )

        cluster_blocks = list(cluster_results.get("Cluster_Block", []))

        for idx, entry in enumerate(cluster_results.get("Rectangles", [])):
            poly_id = entry.get("poly_id") or f"POLY_{idx}"
            bbox = _coerce_bbox(entry.get("bbox") or (0, 0, 0, 0))
            rectangles.append(
                Rectangle(
                    id=entry.get("id") or poly_id,
                    poly_id=poly_id,
                        bbox=bbox,
                    rotation=float(entry.get("rotation", 0.0) or 0.0),
                    confidence=entry.get("confidence"),
                    source_module=entry.get("source") or "cluster_ocr",
                    type=entry.get("type"),
                    line_ids=[],
                )
            )

        for idx, entry in enumerate(cluster_results.get("Contours", [])):
            poly_id = entry.get("poly_id") or f"POLY_{idx}"
            points = _coerce_contour_points(entry.get("points"))
            bbox = _coerce_bbox(entry.get("bbox") or (0, 0, 0, 0))
            contours.append(
                Contour(
                    id=entry.get("id") or poly_id,
                    poly_id=poly_id,
                    points=points,
                    bbox=bbox,
                    area=float(entry.get("area", _bbox_area(bbox)) or _bbox_area(bbox)),
                )
            )

    existing_poly_ids = {rect.poly_id for rect in rectangles}
    existing_bboxes = {rect.bbox for rect in rectangles}  # Also track bboxes to avoid duplicates
    if isinstance(rect_data, dict):
        for idx, entry in enumerate(rect_data.get("rectangles", [])):
            bbox_local = entry.get("bbox_global") or entry.get("bbox") or (0, 0, 0, 0)
            bbox = _coerce_bbox(bbox_local)
            poly_id = entry.get("poly_id")
            if not poly_id:
                poly_id = f"RECTDET_{idx}"
            # Skip if poly_id already exists OR if bbox already exists (duplicate geometry)
            if poly_id in existing_poly_ids or bbox in existing_bboxes:
                continue
            rectangles.append(
                Rectangle(
                    id=entry.get("id") or poly_id,
                    poly_id=poly_id,
                    bbox=bbox,
                    rotation=float(entry.get("rotation", 0.0) or 0.0),
                    confidence=None,
                    source_module="rectangle_detection",
                    type=None,
                )
            )
            existing_poly_ids.add(poly_id)
            contour_pts = entry.get("contour")
            if contour_pts is not None:
                contours.append(
                    Contour(
                        id=f"CONTOUR_{poly_id}",
                        poly_id=poly_id,
                        points=_coerce_contour_points(contour_pts),
                        bbox=bbox,
                        area=float(entry.get("area", _bbox_area(bbox)) or _bbox_area(bbox)),
                    )
                )

    lines_by_poly: Dict[str, List[str]] = defaultdict(list)
    for line in cluster_lines:
        if line.poly_id:
            lines_by_poly[line.poly_id].append(line.id)

    for rect in rectangles:
        rect.line_ids = lines_by_poly.get(rect.poly_id, [])

    return cluster_lines, rectangles, contours, cluster_blocks


def _coerce_page_meta(page_meta: Optional[Any]) -> Optional[PageMetadata]:
    if page_meta is None:
        return None
    if isinstance(page_meta, PageMetadata):
        return page_meta
    if isinstance(page_meta, dict):
        return PageMetadata(
            page_id=str(page_meta.get("page_id") or page_meta.get("PAGE_ID") or ""),
            width=float(page_meta.get("width") or page_meta.get("WIDTH") or 0.0),
            height=float(page_meta.get("height") or page_meta.get("HEIGHT") or 0.0),
            batch_id=page_meta.get("batch_id") or page_meta.get("BATCH_ID"),
            document_id=page_meta.get("document_id") or page_meta.get("DOCUMENT_ID"),
            rotation_deg=float(page_meta.get("rotation_deg") or 0.0),
        )
    return None


def build_data_cleaning_inputs_v2(
    rect_data: Optional[Dict[str, Any]],
    cluster_results: Optional[Dict[str, Any]],
    region_results: Optional[Dict[str, Any]],
    page_meta: Optional[Any] = None,
) -> DataCleaningInputsV2:
    region_lines, region_blocks = _ingest_region_payload(region_results)
    cluster_lines, rectangles, contours, cluster_blocks = _ingest_cluster_payload(
        cluster_results, rect_data
    )
    page_norm = _coerce_page_meta(page_meta)
    if page_norm and abs(page_norm.rotation_deg) > 1e-3:
        _apply_rotation_normalization(page_norm.rotation_deg, region_lines, region_blocks)
        _apply_cluster_rotation(page_norm.rotation_deg, cluster_lines, rectangles)
        if contours:
            _apply_contour_rotation(page_norm.rotation_deg, contours)
    inputs = DataCleaningInputsV2(
        region_lines=region_lines,
        region_blocks=region_blocks,
        cluster_lines=cluster_lines,
        rectangles=rectangles,
        contours=contours,
        page=page_norm,
        cluster_blocks=cluster_blocks,
    )
    enricher = FeatureEnrichmentEngine(inputs)
    (
        inputs.cluster_features,
        inputs.region_block_features,
        inputs.region_line_features,
    ) = enricher.compute()
    return inputs


# Region-driven text structures (lists, paragraphs, KV pairs, strings)

@dataclass
class RegionLine:
    """Single text string produced by Region OCR.

    These lines carry rich geometric metadata so we can reason about lists,
    paragraphs, strings, and key-value structures without referencing cluster
    contours.
    """

    id: str
    text_raw: str
    text_norm: str
    bbox: BBox
    tokens: List[str] = field(default_factory=list)
    block_id: Optional[str] = None
    x_start: Optional[float] = None
    x_mid: Optional[float] = None
    y_mid: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    font_size: Optional[float] = None
    text_density: Optional[float] = None

    def __post_init__(self) -> None:
        self.refresh_geometry()

    def refresh_geometry(self) -> None:
        x1, y1, x2, y2 = self.bbox
        self.x_start = x1
        self.width = x2 - x1
        self.height = y2 - y1
        self.x_mid = (x1 + x2) / 2
        self.y_mid = (y1 + y2) / 2


@dataclass
class RegionBlock:
    """Grouping of RegionLines describing higher-level structures."""

    id: str
    bbox: BBox
    line_ids: List[str]
    density: Optional[float] = None
    structure_type: Optional[str] = None  # paragraph|list|string|kv
    alignment_type: Optional[str] = None  # left|center|right|justified

# Cluster-driven text structures (tables, cell clusters, contours)

@dataclass
class ClusterLine:
    """Minimal recall unit for text extracted inside rectangle/contour."""

    id: str
    text_raw: str
    text_norm: str
    bbox: BBox
    poly_id: Optional[str]
    tokens: List[str] = field(default_factory=list)
    text_density: Optional[float] = None
    col_index: Optional[int] = None


@dataclass
class Rectangle:
    """Geometric enclosure sourced from rectangle detection / contours."""

    id: str
    poly_id: str
    bbox: BBox
    rotation: float = 0.0
    confidence: Optional[float] = None
    source_module: str = "rectangle_detection"
    type: Optional[str] = None
    line_ids: List[str] = field(default_factory=list)


@dataclass
class Contour:
    """Polygonal enclosure (full set of points for IOU/containment tests)."""

    id: str
    poly_id: str
    points: List[Point]
    bbox: BBox
    area: float
    is_closed: bool = True

# Page metadata

@dataclass
class PageMetadata:
    page_id: str
    width: float
    height: float
    batch_id: Optional[str] = None
    document_id: Optional[str] = None
    rotation_deg: float = 0.0


# Heuristic constants

BULLET_ENUM_PATTERN = re.compile(
    r"^\s*(?:[-*•‣∙▪]|(?:\(?[0-9]+\)?|[a-zA-Z])[\.)])\s+"
)
DELIMITER_PATTERN = re.compile(r"[:=]")
ID_LIKE_PATTERN = re.compile(r"^[A-Za-z]{1,6}[A-Za-z0-9\-_/]*\d[A-Za-z0-9\-_/]*$")
NUMERIC_VALUE_PATTERN = re.compile(r"^\s*[\-+]?\d[\d,\.]*\s*(?:[%$]|[x×]\s*\d+)?\s*$")
INDENT_BUCKET_PX = 4.0
DELIMITER_BAND_PX = 12.0


# Text normalization utilities (ported from legacy data_cleaning)

class TextNormalizer:
    """Engineering document normalization with safeguards."""

    def __init__(self, locale: str = "en_GB"):
        self.locale = locale
        self.acronyms: Set[str] = {
            "PO",
            "VAT",
            "PN",
            "ID",
            "REV",
            "IN",
            "DN",
            "OP",
            "ITM",
            "QTY",
            "PT",
            "BOM",
            "WI",
        }
        self.no_fold_pattern = re.compile(
            r"\b(?:\d+-\d+[A-Za-z]+(?:PH|SS)?|M\d+[-x]\d+(?:\.\d+)?|REV-[A-Z])\b"
        )

    def normalize(self, text_raw: str) -> Dict[str, str]:
        """Returns dict with both raw and normalized text."""

        protected = {m.group() for m in self.no_fold_pattern.finditer(text_raw)}

        text = unicodedata.normalize("NFC", text_raw)
        text = self._collapse_whitespace(text)
        text = text.lower()
        text = re.sub(r"[\u2012-\u2015\u2053]", "-", text)
        text = self._normalize_bullets(text)
        text = self._normalize_hyphens(text, protected)
        text = re.sub(r"(\d),(\d{3}\b)", r"\1\2", text)

        return {"text_raw": text_raw, "text_norm": text}

    def _collapse_whitespace(self, text: str) -> str:
        lines = text.splitlines()
        return " ".join(
            (" " * (len(line) - len(line.lstrip())) if i > 0 else "") + line.strip()
            for i, line in enumerate(lines)
        )

    def _normalize_bullets(self, text: str) -> str:
        return re.sub(
            r"(^[\s]*)[•‣∙▪]\s+(\S{1,4}\b)",
            r"\1- \2",
            text,
            flags=re.MULTILINE,
        )

    def _normalize_hyphens(self, text: str, protected: Set[str]) -> str:
        parts: List[str] = []
        last_end = 0
        for match in self.no_fold_pattern.finditer(text):
            parts.append(text[last_end : match.start()])
            parts.append(match.group())
            last_end = match.end()
        parts.append(text[last_end:])
        base_text = "".join(parts)
        return re.sub(r"(\w)-(\w)", r"\1\2", base_text)


_TEXT_NORMALIZER = TextNormalizer()


# Coordinate rotation normalization

def _rotate_bbox(center: Tuple[float, float], bbox: BBox, angle_rad: float) -> BBox:
    x1, y1, x2, y2 = bbox
    points = np.array(
        [
            [x1, y1],
            [x1, y2],
            [x2, y1],
            [x2, y2],
        ],
        dtype=np.float32,
    )
    rot_matrix = np.array(
        [
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)],
        ]
    )
    translated = points - center
    rotated = translated @ rot_matrix.T + center
    min_xy = rotated.min(axis=0)
    max_xy = rotated.max(axis=0)
    return (float(min_xy[0]), float(min_xy[1]), float(max_xy[0]), float(max_xy[1]))


def _apply_rotation_normalization(
    rotation_deg: float,
    region_lines: List[RegionLine],
    region_blocks: List[RegionBlock],
) -> None:
    angle_rad = math.radians(-rotation_deg)
    for line in region_lines:
        line.bbox = _rotate_bbox((0.0, 0.0), line.bbox, angle_rad)
        line.refresh_geometry()
    for block in region_blocks:
        block.bbox = _rotate_bbox((0.0, 0.0), block.bbox, angle_rad)


def _apply_cluster_rotation(
    rotation_deg: float,
    cluster_lines: List[ClusterLine],
    rectangles: List[Rectangle],
) -> None:
    angle_rad = math.radians(-rotation_deg)
    for rect in rectangles:
        rect.bbox = _rotate_bbox((0.0, 0.0), rect.bbox, angle_rad)
    for line in cluster_lines:
        line.bbox = _rotate_bbox((0.0, 0.0), line.bbox, angle_rad)


def _apply_contour_rotation(rotation_deg: float, contours: List[Contour]) -> None:
    angle_rad = math.radians(-rotation_deg)
    rot_matrix = np.array(
        [
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)],
        ]
    )
    for contour in contours:
        if not contour.points:
            continue
        pts = np.array(contour.points, dtype=np.float32)
        rotated = (pts @ rot_matrix.T)
        contour.points = [(float(x), float(y)) for x, y in rotated]
        xs = rotated[:, 0]
        ys = rotated[:, 1]
        contour.bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


# Feature enrichment helpers

def _mad(values: List[float]) -> float:
    if not values:
        return 0.0
    median = float(np.median(values))
    deviations = [abs(v - median) for v in values]
    return float(np.median(deviations))


def _cv_from_mad(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = float(np.mean(values))
    if math.isclose(mean, 0.0):
        return 0.0
    return _mad(values) / max(mean, 1e-6)


def _cv_from_values(values: List[float]) -> float:
    """Coefficient of variation (std/mean). Used by format classifier grid/survival helpers."""
    if not values:
        return 0.0
    m = _safe_mean(values)
    if m == 0.0:
        return 0.0
    var = _safe_mean([(v - m) ** 2 for v in values])
    return float(math.sqrt(var)) / m


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _line_char_length(line: RegionLine) -> int:
    text = line.text_norm or line.text_raw or ""
    return len(text.strip())


def _line_has_bullet(line: RegionLine) -> bool:
    text = (line.text_norm or line.text_raw or "").lstrip()
    return bool(BULLET_ENUM_PATTERN.match(text))


def _line_has_delimiter(line: RegionLine) -> bool:
    text = line.text_norm or line.text_raw or ""
    return bool(DELIMITER_PATTERN.search(text))


def _line_numeric_fraction(line: RegionLine) -> float:
    tokens = line.tokens or _tokenize(line.text_norm or line.text_raw or "")
    if not tokens:
        return 0.0
    numeric = sum(1 for tok in tokens if any(ch.isdigit() for ch in tok))
    return float(numeric) / float(len(tokens))


def _delimiter_position_norm(line: RegionLine) -> Optional[float]:
    text = line.text_norm or line.text_raw or ""
    match = DELIMITER_PATTERN.search(text)
    if not match:
        return None
    idx = match.start()
    length = max(1, len(text))
    return float(idx) / float(length)


def _indent_bucket(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    bucket = INDENT_BUCKET_PX * round(value / max(INDENT_BUCKET_PX, 1.0))
    return bucket


def _polygon_contains_point(point: Point, polygon: List[Point]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-6) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _bbox_fully_inside(bbox: BBox, polygon: List[Point]) -> bool:
    x1, y1, x2, y2 = bbox
    corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
    return all(_polygon_contains_point(pt, polygon) for pt in corners)


def _cluster_positions(values: List[Tuple[float, Any]], tolerance: float) -> List[Dict[str, Any]]:
    if not values:
        return []
    clusters: List[Dict[str, Any]] = []
    for value, payload in sorted(values, key=lambda v: v[0]):
        if not clusters or value - clusters[-1]["max_val"] > tolerance:
            clusters.append({
                "values": [value],
                "payloads": [payload],
                "min_val": value,
                "max_val": value,
            })
        else:
            clusters[-1]["values"].append(value)
            clusters[-1]["payloads"].append(payload)
            clusters[-1]["max_val"] = value
    for cluster in clusters:
        vals = cluster["values"]
        cluster["center"] = sum(vals) / len(vals)
        if len(vals) > 1:
            cluster["std"] = float(np.std(vals))
        else:
            cluster["std"] = 0.0
    return clusters


def _line_area(line: ClusterLine) -> float:
    x1, y1, x2, y2 = line.bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _poly_points_from(rect: Rectangle, contour: Optional[Contour]) -> List[Point]:
    if contour and contour.points:
        return contour.points
    x1, y1, x2, y2 = rect.bbox
    return [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]


def _polygon_area(polygon: List[Point]) -> float:
    """Compute polygon area using shoelace formula."""
    if len(polygon) < 3:
        return 0.0
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def _detect_ruling_lines(poly_points: List[Point], poly_width: float, poly_height: float) -> bool:
    """Detect if polygon geometry indicates ruling lines (table borders)"""
    if len(poly_points) < 4:
        return False

    # Count horizontal and vertical edges
    h_edges = 0
    v_edges = 0
    total_edges = len(poly_points)
    angle_tolerance = 5.0  

    for i in range(total_edges):
        p1 = poly_points[i]
        p2 = poly_points[(i + 1) % total_edges]
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        edge_len = math.sqrt(dx * dx + dy * dy)
        if edge_len < 1e-6:
            continue

        # Check if edge is horizontal (dy ~ 0)
        angle_from_horizontal = math.degrees(math.atan2(dy, dx))
        if angle_from_horizontal < angle_tolerance or angle_from_horizontal > (180 - angle_tolerance):
            h_edges += 1
        # Check if edge is vertical (dx ~ 0)
        angle_from_vertical = abs(90 - angle_from_horizontal)
        if angle_from_vertical < angle_tolerance:
            v_edges += 1

    # Ruling lines indicated if most edges are axis-aligned
    axis_aligned_ratio = (h_edges + v_edges) / max(total_edges, 1)

    # Check for rectangular shape (4 corners, axis-aligned)
    is_rectangular = (
        len(poly_points) == 4
        and h_edges >= 2
        and v_edges >= 2
    )

    # Check corner angles for sharpness (~90 degrees)
    sharp_corners = 0
    for i in range(len(poly_points)):
        p0 = poly_points[(i - 1) % len(poly_points)]
        p1 = poly_points[i]
        p2 = poly_points[(i + 1) % len(poly_points)]
        
        v1 = (p0[0] - p1[0], p0[1] - p1[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        
        len1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        len2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if len1 < 1e-6 or len2 < 1e-6:
            continue
        
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cos_angle = dot / (len1 * len2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_deg = math.degrees(math.acos(cos_angle))
        
        # Sharp corner if angle is close to 90 degrees
        if 80 <= angle_deg <= 100:
            sharp_corners += 1

    has_ruling = (
        is_rectangular
        or (axis_aligned_ratio >= 0.75 and sharp_corners >= 2)
        or (h_edges >= 2 and v_edges >= 2 and sharp_corners >= 4)
    )

    return has_ruling


def _poly_iou(poly_a: List[Point], poly_b: List[Point]) -> float:
    try:
        hull_a = cv2.convexHull(np.array(poly_a, dtype=np.float32))
        hull_b = cv2.convexHull(np.array(poly_b, dtype=np.float32))
        area_a = float(cv2.contourArea(hull_a))
        area_b = float(cv2.contourArea(hull_b))
        if area_a <= 0 or area_b <= 0:
            return 0.0
        retval, inter_area = cv2.intersectConvexConvex(hull_a, hull_b)
        if not retval:
            return 0.0
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return float(inter_area) / union
    except Exception:
        return 0.0


# Feature Enrichment Engine

class FeatureEnrichmentEngine:

    def __init__(self, inputs: DataCleaningInputsV2) -> None:
        self.inputs = inputs
        self.page_width = (
            float(inputs.page.width)
            if inputs.page and inputs.page.width
            else 0.0
        )
        self.page_height = (
            float(inputs.page.height)
            if inputs.page and inputs.page.height
            else 0.0
        )

    def compute(
        self,
    ) -> Tuple[
        Dict[str, Dict[str, Any]],
        Dict[str, Dict[str, Any]],
        Dict[str, Dict[str, Any]],
    ]:
        """Compute all enriched features."""
        cluster_features = self._enrich_cluster_geometry()
        region_block_features, region_line_features = self._enrich_region_features()
        return cluster_features, region_block_features, region_line_features

    # Region feature enrichment

    def _enrich_region_features(
        self,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        line_lookup = {line.id: line for line in self.inputs.region_lines}
        block_features: Dict[str, Dict[str, Any]] = {}
        line_features: Dict[str, Dict[str, Any]] = {}

        for block in self.inputs.region_blocks or []:
            block_lines = [line_lookup[lid] for lid in block.line_ids if lid in line_lookup]
            if not block_lines:
                continue
            y_mids = sorted(
                line.y_mid or (line.bbox[1] + line.bbox[3]) / 2 for line in block_lines
            )
            gaps = [y_mids[i + 1] - y_mids[i] for i in range(len(y_mids) - 1)]
            gap_cv = _cv_from_mad(gaps)

            char_lengths = [_line_char_length(line) for line in block_lines]
            mean_len = _safe_mean(char_lengths)
            line_len_cv = _cv_from_mad(char_lengths)

            numeric_fraction = _safe_mean(
                [_line_numeric_fraction(line) for line in block_lines]
            )
            bullet_hits = [_line_has_bullet(line) for line in block_lines]
            bullet_frac = float(sum(1 for hit in bullet_hits if hit)) / len(block_lines)
            bullet_line_count = int(sum(bullet_hits))
            colon_hits = [_line_has_delimiter(line) for line in block_lines]
            colon_frac = float(sum(1 for hit in colon_hits if hit)) / len(block_lines)

            widths = [line.width or (line.bbox[2] - line.bbox[0]) for line in block_lines]
            width_cv = _cv_from_mad(widths)
            mean_line_width = _safe_mean(widths)

            x_start_rounded = [
                _indent_bucket(line.x_start)
                for line in block_lines
                if line.x_start is not None
            ]
            indent_levels = sorted({val for val in x_start_rounded if val is not None})
            indent_levels_count = len(indent_levels)
            indent_ladder = 0.0
            if indent_levels_count >= 2:
                indent_ladder = (indent_levels[-1] - indent_levels[0]) / max(
                    mean_line_width, 1e-3
                )

            tokens_total = sum(len(line.tokens) for line in block_lines)

            x1, y1, x2, y2 = block.bbox
            block_width = max(1e-3, x2 - x1)
            block_height = max(1e-3, y2 - y1)
            aspect_ratio = block_width / block_height if block_height else 0.0
            width_rel_page = (
                (block_width / self.page_width) if self.page_width else 0.0
            )
            wrap_ratio = mean_line_width / block_width if block_width else 0.0

            density = block.density
            if density is None:
                all_text = " ".join(
                    line.text_norm or line.text_raw or "" for line in block_lines
                )
                density = _text_density(all_text, block.bbox)

            font_sizes = [
                line.font_size or line.height or (line.bbox[3] - line.bbox[1])
                for line in block_lines
            ]
            font_size_mean = _safe_mean(font_sizes)
            font_size_max = max(font_sizes) if font_sizes else 0.0
            font_size_rel = (
                (font_size_mean / self.page_height) if self.page_height else 0.0
            )

            delimiter_bands = set()
            for line, has_delim in zip(block_lines, colon_hits):
                if not has_delim:
                    continue
                band = int(
                    (line.y_mid or (line.bbox[1] + line.bbox[3]) / 2)
                    / max(DELIMITER_BAND_PX, 1.0)
                )
                delimiter_bands.add(band)

            block_features[block.id] = {
                "n_lines": len(block_lines),
                "mean_len": mean_len,
                "num_frac": numeric_fraction,
                "bullet_frac": bullet_frac,
                "bullet_line_count": bullet_line_count,
                "colon_frac": colon_frac,
                "aspect_ratio": aspect_ratio,
                "width_cv": width_cv,
                "line_len_cv": line_len_cv,
                "gap_cv": gap_cv,
                "width_rel_page": width_rel_page,
                "wrap_ratio": wrap_ratio,
                "indent_levels_count": indent_levels_count,
                "indent_ladder": indent_ladder,
                "token_count": tokens_total,
                "density": density,
                "font_size_mean": font_size_mean,
                "font_size_max": font_size_max,
                "font_size_rel": font_size_rel,
                "delimiter_band_count": len(delimiter_bands),
            }

        for line in self.inputs.region_lines:
            text = line.text_norm or line.text_raw or ""
            num_frac = _line_numeric_fraction(line)
            has_delim = _line_has_delimiter(line)
            is_id_like = bool(ID_LIKE_PATTERN.match(text.strip()))
            is_numeric_like = bool(NUMERIC_VALUE_PATTERN.match(text))

            is_label_candidate = bool(
                has_delim
                or text.strip().endswith(":")
                or (len(line.tokens) <= 6 and not is_numeric_like and not num_frac > 0.6)
            )
            is_value_candidate = bool(
                (not is_label_candidate and (is_numeric_like or num_frac > 0.5))
                or (has_delim and len(line.tokens) >= 2)
            )

            line_features[line.id] = {
                "is_single_line_label_candidate": is_label_candidate,
                "is_single_line_value_candidate": is_value_candidate,
                "delimiter_position_norm": _delimiter_position_norm(line),
                "num_frac": num_frac,
                "is_id_like": is_id_like,
                "is_numeric_like": is_numeric_like,
            }

        return block_features, line_features

    # Cluster feature enrichment

    def _enrich_cluster_geometry(self) -> Dict[str, Dict[str, Any]]:
        rectangles = self.inputs.rectangles or []
        contours = {c.poly_id: c for c in self.inputs.contours or []}
        lines_by_poly: Dict[str, List[ClusterLine]] = defaultdict(list)
        for line in self.inputs.cluster_lines:
            if line.poly_id:
                lines_by_poly[line.poly_id].append(line)

        widths: List[float] = []
        heights: List[float] = []
        centers: Dict[str, Tuple[float, float]] = {}
        polygons: Dict[str, List[Point]] = {}
        features: Dict[str, Dict[str, Any]] = {}

        for rect in rectangles:
            poly_points = _poly_points_from(rect, contours.get(rect.poly_id))
            polygons[rect.poly_id] = poly_points

            # Derive geometry from polygon points, not bbox
            poly_xs = [p[0] for p in poly_points]
            poly_ys = [p[1] for p in poly_points]
            poly_x_min = min(poly_xs)
            poly_x_max = max(poly_xs)
            poly_y_min = min(poly_ys)
            poly_y_max = max(poly_ys)
            poly_width = max(1e-3, poly_x_max - poly_x_min)
            poly_height = max(1e-3, poly_y_max - poly_y_min)

            # Derive left/right edges from polygon (for alignment)
            left_edge_x = poly_x_min
            right_edge_x = poly_x_max

            widths.append(poly_width)
            heights.append(poly_height)
            centers[rect.poly_id] = (
                (poly_x_min + poly_x_max) / 2.0,
                (poly_y_min + poly_y_max) / 2.0,
            )

            # Compute polygon area using shoelace formula
            poly_area = _polygon_area(poly_points)
            if poly_area < 1e-6:
                poly_area = poly_width * poly_height

            lines = lines_by_poly.get(rect.poly_id, [])
            line_area_sum = sum(_line_area(line) for line in lines)
            density = (line_area_sum / poly_area) if poly_area > 0 else 0.0
            aspect_ratio = poly_width / poly_height if poly_height else 0.0

            # Enclosure: check if line bbox is fully inside polygon
            enclosed_counts = sum(
                1 for line in lines if _bbox_fully_inside(line.bbox, poly_points)
            )
            enclosed_fraction = (enclosed_counts / len(lines)) if lines else 0.0
            enclosed_in_poly = enclosed_fraction >= 0.95

            # x_start clustering per contour
            x_positions = [(line.bbox[0], line) for line in lines]
            tol_x = max(poly_width * 0.05, 6.0)
            x_clusters = _cluster_positions(x_positions, tol_x)
            cols_hint = min(len(x_clusters), 6)
            for idx, cluster in enumerate(x_clusters):
                for line in cluster["payloads"]:
                    line.col_index = idx

            tab_stops_stability = 0.0
            if x_clusters and poly_width > 0:
                avg_std = sum(cluster["std"] for cluster in x_clusters) / len(x_clusters)
                tab_stops_stability = max(0.0, 1.0 - (avg_std / poly_width))

            # y-band row consistency per contour
            y_positions = [((line.bbox[1] + line.bbox[3]) / 2.0, line) for line in lines]
            tol_y = max(poly_height * 0.07, 6.0)
            y_clusters = _cluster_positions(y_positions, tol_y)
            row_count = len(y_clusters)
            row_centers = [cluster["center"] for cluster in y_clusters]
            row_centers.sort()
            row_gaps = [
                row_centers[i + 1] - row_centers[i] for i in range(len(row_centers) - 1)
            ]
            row_gap_mean = float(np.mean(row_gaps)) if row_gaps else 0.0
            row_gap_cv = (_mad(row_gaps) / row_gap_mean) if row_gap_mean else 0.0

            # Vertical projection peaks from contour geometry
            # Project polygon points onto vertical axis (count density per y-bin)
            bins = 16
            vert_profile = [0.0] * bins
            for px, py in poly_points:
                bin_idx = max(0, min(bins - 1, int(((py - poly_y_min) / poly_height) * bins)))
                vert_profile[bin_idx] += 1.0
            # Also add line coverage to vertical profile
            for line in lines:
                ly1 = line.bbox[1]
                ly2 = line.bbox[3]
                start = max(0, min(bins - 1, int(((ly1 - poly_y_min) / poly_height) * bins)))
                end = max(start, min(bins - 1, int(((ly2 - poly_y_min) / poly_height) * bins)))
                line_width = max(1.0, line.bbox[2] - line.bbox[0])
                for b in range(start, end + 1):
                    vert_profile[b] += line_width
            max_bin = max(vert_profile) if vert_profile else 0.0
            vert_projection_peaks = 0
            if max_bin > 0:
                threshold = max_bin * 0.6
                vert_projection_peaks = sum(1 for val in vert_profile if val >= threshold)

            # rect_alignment: compare line x_start/x_end with poly edge-derived left/right
            rect_alignment_left = rect_alignment_right = 0.0
            if lines and poly_width > 0:
                rect_alignment_left = (
                    sum(abs(line.bbox[0] - left_edge_x) / poly_width for line in lines)
                    / len(lines)
                )
                rect_alignment_right = (
                    sum(abs(line.bbox[2] - right_edge_x) / poly_width for line in lines)
                    / len(lines)
                )
            # Combined rectangle_alignment score (0 = perfect alignment)
            rectangle_alignment = min(rect_alignment_left, rect_alignment_right)

            has_ruling_lines = _detect_ruling_lines(poly_points, poly_width, poly_height)

            features[rect.poly_id] = {
                "aspect_ratio": aspect_ratio,
                "density": density,
                "n_lines_in_rect": len(lines),
                "cols_hint": cols_hint,
                "tab_stops_stability": tab_stops_stability,
                "row_count": row_count,
                "row_gap_mean": row_gap_mean,
                "row_gap_cv": row_gap_cv,
                "vert_projection_peaks": vert_projection_peaks,
                "enclosed_fraction": enclosed_fraction,
                "enclosed_in_poly": enclosed_in_poly,
                "rect_alignment_left": rect_alignment_left,
                "rect_alignment_right": rect_alignment_right,
                "rectangle_alignment": rectangle_alignment,
                "has_ruling_lines": has_ruling_lines,
                "poly_points": poly_points,
                "poly_width": poly_width,
                "poly_height": poly_height,
                "poly_area": poly_area,
            }

        if not features:
            return {}

        median_width = float(np.median(widths)) if widths else 0.0
        median_height = float(np.median(heights)) if heights else 0.0

        sorted_x = sorted(rectangles, key=lambda r: centers.get(r.poly_id, (0, 0))[0])
        sorted_y = sorted(rectangles, key=lambda r: centers.get(r.poly_id, (0, 0))[1])

        # rect_spacing: compute spacing between rectangles
        def _assign_min_gap(sorted_rects, axis: int, key_name: str) -> None:
            for idx, rect in enumerate(sorted_rects):
                cx, cy = centers.get(rect.poly_id, (0.0, 0.0))
                neighbor_vals = []
                if idx > 0:
                    px = centers.get(sorted_rects[idx - 1].poly_id, (0.0, 0.0))
                    neighbor_vals.append(abs((cx, cy)[axis] - (px[axis])))
                if idx + 1 < len(sorted_rects):
                    nx = centers.get(sorted_rects[idx + 1].poly_id, (0.0, 0.0))
                    neighbor_vals.append(abs((cx, cy)[axis] - (nx[axis])))
                gap = min(neighbor_vals) if neighbor_vals else 0.0
                features[rect.poly_id][key_name] = gap

        _assign_min_gap(sorted_x, 0, "inter_rect_gap_x")
        _assign_min_gap(sorted_y, 1, "inter_rect_gap_y")

        # rect_size_similarity: compare to median dimensions
        for rect in rectangles:
            feat = features.get(rect.poly_id)
            if not feat:
                continue
            pw = feat.get("poly_width", 1e-3)
            ph = feat.get("poly_height", 1e-3)
            sim_w = 1.0 - abs(pw - median_width) / max(median_width, 1e-3)
            sim_h = 1.0 - abs(ph - median_height) / max(median_height, 1e-3)
            rect_size_similarity = max(0.0, (sim_w + sim_h) / 2.0)
            feat["rect_size_similarity"] = rect_size_similarity

        # iou_overlap: use polygon for overlap measurement
        for rect in rectangles:
            poly_a = polygons.get(rect.poly_id)
            if not poly_a:
                continue
            max_iou = 0.0
            for other in rectangles:
                if other.poly_id == rect.poly_id:
                    continue
                poly_b = polygons.get(other.poly_id)
                if not poly_b:
                    continue
                iou = _poly_iou(poly_a, poly_b)
                if iou > max_iou:
                    max_iou = iou
            features[rect.poly_id]["iou_overlap_max"] = max_iou

        # GRID DETECTION: Detect LOCAL grid formations (isolated table structures)
        
        # Initialize all rectangles as not part of grid
        for rect in rectangles:
            feat = features.get(rect.poly_id)
            if feat:
                feat["is_part_of_grid"] = False
                feat["is_strong_grid"] = False
                feat["grid_cell_count"] = 0
                feat["grid_rows"] = 0
                feat["grid_cols"] = 0
                feat["grid_regularity"] = 0.0
                feat["grid_row_idx"] = -1
                feat["grid_col_idx"] = -1

        if len(rectangles) < 4:
            return features

        # Collect rectangle data with spatial info
        rect_data_list = []
        for rect in rectangles:
            feat = features.get(rect.poly_id)
            if not feat:
                continue
            cx, cy = centers.get(rect.poly_id, (0, 0))
            pw = feat.get("poly_width", 0)
            ph = feat.get("poly_height", 0)
            rect_data_list.append({
                "poly_id": rect.poly_id,
                "cx": cx,
                "cy": cy,
                "width": pw,
                "height": ph,
                "x1": cx - pw / 2,
                "y1": cy - ph / 2,
                "x2": cx + pw / 2,
                "y2": cy + ph / 2,
                "area": pw * ph,
            })

        # Build spatial connectivity graph
        
        def are_spatially_adjacent(r1: Dict, r2: Dict, gap_tolerance: float = 0.5) -> bool:
            # Use the smaller height as reference for gap tolerance
            ref_size = min(r1["height"], r2["height"])
            max_gap = ref_size * gap_tolerance  # Tighter: 0.5 instead of 0.8
            
            # Check horizontal adjacency (same row, side by side)
            # Y ranges must overlap significantly
            y_overlap_amount = min(r1["y2"], r2["y2"]) - max(r1["y1"], r2["y1"])
            y_overlap = y_overlap_amount > ref_size * 0.5  # Stricter: 50% height overlap
            x_gap = min(abs(r1["x2"] - r2["x1"]), abs(r2["x2"] - r1["x1"]))
            x_adjacent = x_gap < max_gap
            
            # Check vertical adjacency (same column, stacked)
            # X ranges must overlap significantly
            x_overlap_amount = min(r1["x2"], r2["x2"]) - max(r1["x1"], r2["x1"])
            x_overlap = x_overlap_amount > min(r1["width"], r2["width"]) * 0.5  # Stricter: 50% width overlap
            y_gap = min(abs(r1["y2"] - r2["y1"]), abs(r2["y2"] - r1["y1"]))
            y_adjacent = y_gap < max_gap
            
            return (y_overlap and x_adjacent) or (x_overlap and y_adjacent)
        
        # Build adjacency list
        n = len(rect_data_list)
        adjacency = {r["poly_id"]: set() for r in rect_data_list}
        
        for i in range(n):
            for j in range(i + 1, n):
                r1, r2 = rect_data_list[i], rect_data_list[j]
                
                # Exclude very wide cells (aspect ratio > 15) from being bridges
                # These are typically spanning headers that connect unrelated regions
                r1_aspect = r1["width"] / r1["height"] if r1["height"] > 0 else 0
                r2_aspect = r2["width"] / r2["height"] if r2["height"] > 0 else 0
                either_is_spanning = r1_aspect > 15 or r2_aspect > 15
                
                # Check column alignment - cells sharing column boundaries are likely same table
                # This handles BOM tables where header row has different height than data rows
                x_overlap = min(r1["x2"], r2["x2"]) - max(r1["x1"], r2["x1"])
                min_width = min(r1["width"], r2["width"])
                columns_aligned = x_overlap > min_width * 0.5  # At least 50% column overlap
                
                # For vertically adjacent cells, use column alignment instead of height similarity
                # For horizontally adjacent cells, still check height similarity
                y_gap = min(abs(r1["y2"] - r2["y1"]), abs(r2["y2"] - r1["y1"]))
                is_vertical_neighbor = y_gap < max(r1["height"], r2["height"]) * 0.8
                
                h_similar = abs(r1["height"] - r2["height"]) < max(r1["height"], r2["height"]) * 0.5
                
                if are_spatially_adjacent(r1, r2) and not either_is_spanning:
                    # Accept if: (1) columns aligned for vertical neighbors, OR (2) heights similar
                    if (is_vertical_neighbor and columns_aligned) or h_similar:
                        adjacency[r1["poly_id"]].add(r2["poly_id"])
                        adjacency[r2["poly_id"]].add(r1["poly_id"])
        
        # Find connected components using BFS
        poly_id_to_rect = {r["poly_id"]: r for r in rect_data_list}
        visited = set()
        connected_components: List[List[Dict]] = []
        
        for r in rect_data_list:
            if r["poly_id"] in visited:
                continue
            # BFS to find all connected rectangles
            component = []
            queue = [r["poly_id"]]
            while queue:
                pid = queue.pop(0)
                if pid in visited:
                    continue
                visited.add(pid)
                component.append(poly_id_to_rect[pid])
                for neighbor in adjacency[pid]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            if len(component) >= 4:  # Need at least 4 for a 2x2 grid
                connected_components.append(component)
        
        # Keep connected components together
        size_groups: List[List[Dict]] = []
        
        for component in connected_components:
            if len(component) >= 4:
                size_groups.append(component)

        # For each size group, check if they form a grid pattern
        for group in size_groups:
            if len(group) < 4:
                continue
            
            # Compute median dimensions for this group
            group_widths = [r["width"] for r in group]
            group_heights = [r["height"] for r in group]
            group_median_w = float(np.median(group_widths))
            group_median_h = float(np.median(group_heights))
            
            # Use tight tolerance based on this group's dimensions
            y_tol = group_median_h * 0.25
            x_tol = group_median_w * 0.25
            
            # Cluster by Y (rows) within this size group
            y_positions = [(r["cy"], r) for r in group]
            row_clusters = _cluster_positions(y_positions, y_tol)
            
            # Cluster by X (columns) within this size group
            x_positions = [(r["cx"], r) for r in group]
            col_clusters = _cluster_positions(x_positions, x_tol)
            
            n_rows = len(row_clusters)
            n_cols = len(col_clusters)
            
            # Must have at least 2 rows AND 2 columns
            if n_rows < 2 or n_cols < 2:
                continue
            
            # Build row/column membership
            row_membership = {}
            col_membership = {}
            for row_idx, cluster in enumerate(row_clusters):
                for r in cluster["payloads"]:
                    row_membership[r["poly_id"]] = row_idx
            for col_idx, cluster in enumerate(col_clusters):
                for r in cluster["payloads"]:
                    col_membership[r["poly_id"]] = col_idx
            
            # Count rectangles per row and column
            row_counts = {}
            col_counts = {}
            for r in group:
                rid = row_membership.get(r["poly_id"], -1)
                cid = col_membership.get(r["poly_id"], -1)
                if rid >= 0:
                    row_counts[rid] = row_counts.get(rid, 0) + 1
                if cid >= 0:
                    col_counts[cid] = col_counts.get(cid, 0) + 1
            
            # Find rectangles that are truly part of a grid:
            # - Their row has >= 2 rectangles
            # - Their column has >= 2 rectangles
            grid_members = []
            for r in group:
                rid = row_membership.get(r["poly_id"], -1)
                cid = col_membership.get(r["poly_id"], -1)
                if row_counts.get(rid, 0) >= 2 and col_counts.get(cid, 0) >= 2:
                    grid_members.append(r)
            
            if len(grid_members) < 4:
                continue
            
            # Compute grid quality metrics
            # Row spacing consistency
            row_centers = sorted([c["center"] for c in row_clusters if len(c["payloads"]) >= 2])
            row_gaps = [row_centers[i+1] - row_centers[i] for i in range(len(row_centers)-1)]
            row_gap_cv = (_mad(row_gaps) / np.mean(row_gaps)) if row_gaps and np.mean(row_gaps) > 0 else 1.0
            
            # Column spacing consistency
            col_centers = sorted([c["center"] for c in col_clusters if len(c["payloads"]) >= 2])
            col_gaps = [col_centers[i+1] - col_centers[i] for i in range(len(col_centers)-1)]
            col_gap_cv = (_mad(col_gaps) / np.mean(col_gaps)) if col_gaps and np.mean(col_gaps) > 0 else 1.0
            
            # Size consistency
            width_cv = (_mad(group_widths) / group_median_w) if group_median_w > 0 else 1.0
            height_cv = (_mad(group_heights) / group_median_h) if group_median_h > 0 else 1.0
            
            # Grid regularity score
            row_regularity = max(0.0, 1.0 - row_gap_cv)
            col_regularity = max(0.0, 1.0 - col_gap_cv)
            size_regularity = max(0.0, 1.0 - (width_cv + height_cv) / 2.0)
            grid_regularity = (row_regularity + col_regularity + size_regularity) / 3.0
            
            # Is this a strong grid?
            is_strong = grid_regularity >= 0.5 and len(grid_members) >= 4
            
            # Count actual rows/cols with multiple members
            actual_rows = sum(1 for c in row_counts.values() if c >= 2)
            actual_cols = sum(1 for c in col_counts.values() if c >= 2)
            
            # Assign grid features to members
            for r in grid_members:
                feat = features.get(r["poly_id"])
                if feat:
                    feat["is_part_of_grid"] = True
                    feat["is_strong_grid"] = is_strong
                    feat["grid_cell_count"] = len(grid_members)
                    feat["grid_rows"] = actual_rows
                    feat["grid_cols"] = actual_cols
                    feat["grid_regularity"] = grid_regularity
                    feat["grid_row_idx"] = row_membership.get(r["poly_id"], -1)
                    feat["grid_col_idx"] = col_membership.get(r["poly_id"], -1)

        return features


# Standalone enrichment functions 

def _enrich_cluster_geometry(inputs: DataCleaningInputsV2) -> Dict[str, Dict[str, Any]]:
    """Compute cluster/rectangle geometry features."""
    engine = FeatureEnrichmentEngine(inputs)
    return engine._enrich_cluster_geometry()


def _enrich_region_features(
    inputs: DataCleaningInputsV2,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Compute region block and line features."""
    engine = FeatureEnrichmentEngine(inputs)
    return engine._enrich_region_features()

# Format Classifier

# Reference line thickness (thin clean line) for scaling
REFERENCE_LINE_THICKNESS_PX = 2.0
# Base gap tolerance (pixels) used in scaling formula
BASE_GAP_TOLERANCE = 5.0
# Base row/col tolerance multiplier (e.g. 0.25 * median_dim)
BASE_ROW_COL_TOL = 0.25
# Line-like filter: max/min dimension ratio > this => ruling line
LINE_LIKE_AR_THRESHOLD = 8.0
# Max grid size (cell count) to treat as "small grid" for no-numeric → INFO (tune with aspect/style)
SMALL_GRID_MAX_CELLS = 16
# Selection: if TABLE and INFO_CLUSTER scores are both below this, choose ABSTAIN (notes/labels → neither)
ABSTAIN_BOTH_NEGATIVE_THRESHOLD = -8.0
# ---------------------------------------------------------------------------
# Grid formation: union-find by centre distance (<= max_dist), then row-structure split.
# - First pass uses geometry_only=True so content-compatibility is not applied.
# - max_dist = GRID_SPATIAL_CLUSTER_FRAC * page_diag, then scaled by aspect ratio and line thickness.
# - Adaptive scaling: wider/taller pages and thicker lines get larger max_dist so different
#   drawing styles merge grids (e.g. two-column title blocks) more accurately.
# - Optional title-block zone relaxation (GRID_TITLE_BLOCK_DISTANCE_RELAX) can add extra scale in bottom zone.
# ---------------------------------------------------------------------------
GRID_SPATIAL_CLUSTER_FRAC = 0.07
# Adaptive max_dist scaling: aspect ratio (wider page -> larger horizontal gaps -> allow more merge distance)
GRID_MAX_DIST_AR_SCALE = 0.25   # max_dist *= (1 + this * min(1, log1p(ar-1))) for ar >= 1
# Adaptive max_dist scaling: line thickness (thicker lines -> cells further apart in drawings)
GRID_MAX_DIST_THICKNESS_SCALE = 0.15  # max_dist *= (1 + this * (thickness_scale - 1)), cap 1.35
GRID_TITLE_BLOCK_ZONE_Y_MIN = 0.80  # cy/page_h >= this => title-block zone (when RELAX > 1)
GRID_TITLE_BLOCK_DISTANCE_RELAX = 1.0  # 1.0 = off; 1.5 allows ~0.105*diag in zone (for 175985)
# Grid cell count above this: gated penalty only (clarify borderline; don't touch normal 48–54 cell tables)
LARGE_GRID_SIZE_THRESHOLD = 65

# Real table: minimum cell count for TABLE (Google-style: reserve TABLE for data tables, not 2×2/2×4 label grids)
REAL_TABLE_MIN_CELLS = 8
# Title block: bottom of page (structural_position >= this). Adaptive: combined with dyn label_heavy_bar.
# Slightly more conservative so bottom-ish grid rows of real tables are not misclassified as title block.
TITLE_BLOCK_POSITION_MIN = 0.62
# Dimension-like: extreme aspect ratio (narrow strip or wide bar) → INFO/ABSTAIN, never TABLE
DIMENSION_AR_MIN = 5.0  # cell_ar_normalized > this or < 1/this => dimension-like
# Title block rows: treat bottom N rows of a grid as title block. N = max(2, grid_rows // 4) so it scales.
TITLE_BLOCK_ROWS_FRAC = 0.25  # bottom fraction of grid rows; at least 2 rows

# Per-cell disqualify: survival is meant to let a cell's own characteristics override the grid label.
# When in title block and cell looks metadata-like, use per-cell content in survival vector so it can score INFO.
SURVIVAL_TITLE_BLOCK_USE_PER_CELL_NUMERIC_MAX = 0.12   # use per-cell numeric when <= this (no data)
SURVIVAL_TITLE_BLOCK_USE_PER_CELL_LABEL_MIN = 0.5      # and label_value_ratio >= this → use per-cell in vector
# Hard override: cell clearly metadata in title block → INFO regardless of grid.
PER_CELL_OVERRIDE_TITLE_BLOCK_NUMERIC_MAX = 0.0        # numeric_fraction == 0
PER_CELL_OVERRIDE_TITLE_BLOCK_LABEL_MIN = 0.7          # label_value_ratio > this

# Survival per-cell: when grid regularity/alignment are below these, use per-cell content in title block.
TITLE_BLOCK_GRID_REGULARITY_TABLE_MIN = 0.65          # grid_regularity_mean below this → not table-like
# Alignment bar for title-block override; adaptive scale can only lower it, never exceed 0.72 (so 2d_assembly align=0.736 never triggers).
TITLE_BLOCK_GRID_ALIGNMENT_TABLE_MIN_BASE = 0.72
ALIGN_AR_SCALE = 0.06                                 # scale from aspect ratio (same style as grid max_dist)
ALIGN_THICKNESS_SCALE = 0.04                          # scale from thickness

# Position + content override: title block + no numeric + (low regularity OR low alignment).
# Guard: grid_numeric_fraction < this so we only flip when the grid is not a data table (title blocks
# have near-zero grid numeric; BOMs have part numbers, qty → higher). 152021 has high alignment so
# "both low" never fired; OR catches it. Guard avoids regressing BOM-at-bottom docs.
TITLE_BLOCK_POSITION_CONTENT_NUMERIC_MAX = 0.0        # per-cell numeric_fraction <= this
TITLE_BLOCK_GRID_NUMERIC_MAX = 0.10                   # grid_numeric_fraction < this to apply override

# Borderline band for weak structural/content rules: only act when base classifier is undecided.
# All structural and "large grid" rules must be gated by weak_scale so they clarify only—never override
# the base + content rules that got us close to targets.
BORDERLINE_SCORE_DIFF = 6.0   # how close TABLE vs INFO must be to count as borderline
BORDERLINE_MAX_SCORE = 20.0   # do not touch very strong base decisions

# ---------------------------------------------------------------------------
# ADAPTIVE vs FIXED (tune for generalisation across drawing styles, not per-image).
# Adaptive (derived per page):
#   - Page geometry: aspect ratio, line thickness → dyn thresholds (numeric_min, header_delta_min,
#     label_heavy_bar, cell_to_line_min, completeness/regularity). See _dynamic_table_thresholds.
#   - Page content stats: median/p75 numeric_fraction and label_value_ratio → title_block_numeric_max,
#     title_block_header_delta_max so title-block bars follow document style.
#   - Title block rows: bottom max(2, grid_rows*TITLE_BLOCK_ROWS_FRAC) rows; label-heavy bar from dyn.
#   - weak_scale, info_scale: borderline band and table-heavy page context.
# Fixed (tune these for cross-style behaviour): SCORING weights, REAL_TABLE_MIN_CELLS,
#   TITLE_BLOCK_POSITION_MIN, DIMENSION_AR_MIN, BORDERLINE_*, ABSTAIN_BOTH_NEGATIVE_THRESHOLD.
# ---------------------------------------------------------------------------
# Simplified, targeted scoring (Google-style: tables = data tables only; title block/dimensions = explicit INFO).
SCORING = {
    # Real table: grid + content bar only
    "TABLE_boost_numeric_grid": 4.5,
    "TABLE_boost_dictionary_header_extra": 4.0,
    "TABLE_boost_header_only_grid": 4.0,
    "TABLE_boost_cell_to_line": 3.0,
    "TABLE_boost_numeric_regular_rows": 5.0,
    "TABLE_penalty_sub_2x2": 12.0,
    "TABLE_penalty_anomalously_wide": 4.0,
    "TABLE_penalty_non_dominant_grid": 10.0,
    "TABLE_penalty_small_grid_not_data_table": 12.0,  # grid < REAL_TABLE_MIN_CELLS without strong data → no TABLE
    # Title block / dimensions: explicit INFO, never TABLE
    "INFO_boost_title_block": 26.0,
    "TABLE_penalty_title_block": 26.0,
    "INFO_boost_title_block_rows": 20.0,
    "TABLE_penalty_title_block_rows": 22.0,
    "INFO_boost_dimension_like": 18.0,
    "TABLE_penalty_dimension_like": 20.0,
    # Label-heavy / no-dict-header (title blocks, metadata)
    # Label-heavy / no-dict-header (title blocks, metadata) – softened so strong grids can still be TABLE
    "INFO_boost_label_heavy_grid": 9.0,
    "TABLE_penalty_label_heavy_grid": 10.0,
    "INFO_boost_no_dict_header": 14.0,
    "TABLE_penalty_no_dict_header": 16.0,
    "INFO_boost_small_grid_no_numeric": 10.0,
    "TABLE_penalty_small_grid_no_numeric": 10.0,
    "INFO_boost_no_grid_or_single_cell": 12.0,
    "TABLE_penalty_no_grid_or_single_cell": 14.0,
    "TABLE_penalty_small_grid_weak_numeric": 5.0,
    # Structural: irregular/incomplete grid → INFO (push harder so they compete with base TABLE)
    "TABLE_penalty_jagged_grid": 10.0,
    "TABLE_penalty_incomplete_grid": 9.0,
    # Cell size variation per row/col above threshold → INFO bias (explicit rule, not borderline-only)
    "TABLE_penalty_cell_size_variation": 10.0,
    "INFO_boost_cell_size_variation": 9.0,
    "TABLE_boost_neighbour_table": 2.0,
    "INFO_boost_neighbour_info": 2.0,
    "TABLE_penalty_border_sparse": 3.0,
    "TABLE_boost_interior_dense": 2.0,
    "TABLE_penalty_large_grid_border": 2.5,
    "TABLE_penalty_oversized_grid": 0.08,
    "TABLE_penalty_oversized_grid_cap": 2.5,
    # Override base classifier when grid has no/missing content and cell is in title-block zone;
    # gated by weak_scale so it only acts when base TABLE/INFO is borderline.
    "TABLE_penalty_title_block_zone_weak_content": 12.0,
    "INFO_boost_title_block_zone_weak_content": 10.0,
    # No equal size/continuity: incomplete or irregular grid → not a real table
    "TABLE_penalty_incomplete_irregular_grid": 14.0,
}
# Grid label as one term in accumulation (grid-first still runs; its vote is added to the score).
GRID_LABEL_BOOST = 8.0
GRID_LABEL_PENALTY = 6.0

# ---------------------------------------------------------------------------
# Survival mode: prototype distance + hard rules + confidence margin (no weighted sum).
# ---------------------------------------------------------------------------
# Fixed prototypes (feature name -> target value). All keys used in SURVIVAL_*_KEYS must exist.
PROTOTYPE_TABLE = {
    "completeness": 0.9,
    "regularity": 0.85,
    "numeric": 0.6,
    "header_delta": 0.3,
    "grid_size_norm": 0.6,
    "ar_entropy": 0.2,
    "cell_clustering_coef": 0.8,
    "column_alignment_score": 0.9,
    "token_repetition": 0.4,
    "vertical_text_sim": 0.5,
    "label_value_ratio": 0.1,
}
PROTOTYPE_INFO = {
    "completeness": 0.3,
    "regularity": 0.3,
    "numeric": 0.1,
    "label_value_ratio": 0.7,
    "grid_size_norm": 0.0,
    "header_delta": 0.0,
    "ar_entropy": 0.6,
    "cell_clustering_coef": 0.3,
    "column_alignment_score": 0.2,
    "token_repetition": 0.1,
    "vertical_text_sim": 0.1,
}
SURVIVAL_GEOMETRIC_WEIGHT = 0.6
SURVIVAL_CONTENT_WEIGHT = 0.4
SURVIVAL_BASE_THRESHOLD = 0.5
SURVIVAL_CONFIDENCE_MARGIN = 0.15
# Feature keys for distance: geometric first, then content (order for weighting).
SURVIVAL_GEOMETRIC_KEYS = [
    "completeness",
    "regularity",
    "grid_size_norm",
    "ar_entropy",
    "cell_clustering_coef",
    "column_alignment_score",
]
SURVIVAL_CONTENT_KEYS = ["numeric", "header_delta", "label_value_ratio", "token_repetition", "vertical_text_sim"]


@dataclass
class PageGeometryContext:
    """
    Page-level geometry computed once from inputs.page and inputs.rectangles.
    Used to scale tolerances and interpret per-cluster features in document context.
    """
    page_aspect_ratio: float
    page_line_thickness_px: float
    thickness_scale_factor: float
    bridge_ar_threshold: float
    gap_tolerance_scaled: float
    row_col_tol_scale: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_page_geometry_context(inputs: DataCleaningInputsV2) -> PageGeometryContext:
    """
    Compute page-level geometry from inputs.page and inputs.rectangles.
    Single computation per page; pass the result through enrichment and scoring.
    """
    page_w = 1.0
    page_h = 1.0
    if inputs.page and getattr(inputs.page, "width", None) and getattr(inputs.page, "height", None):
        page_w = float(inputs.page.width)
        page_h = float(inputs.page.height)

    page_aspect_ratio = page_w / page_h if page_h > 0 else 1.0

    # Line thickness proxy: median of min(poly_width, poly_height) for line-like rects
    line_like_min_dims: List[float] = []
    for rect in inputs.rectangles or []:
        x1, y1, x2, y2 = rect.bbox
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 0 or h <= 0:
            continue
        min_dim = min(w, h)
        max_dim = max(w, h)
        ar_rect = max_dim / min_dim
        if ar_rect >= LINE_LIKE_AR_THRESHOLD:
            line_like_min_dims.append(min_dim)

    if line_like_min_dims:
        line_like_min_dims.sort()
        page_line_thickness_px = float(line_like_min_dims[len(line_like_min_dims) // 2])
    else:
        # Fallback: median min-dim of all rects, or reference
        all_min_dims = []
        for rect in inputs.rectangles or []:
            x1, y1, x2, y2 = rect.bbox
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if w > 0 and h > 0:
                all_min_dims.append(min(w, h))
        if all_min_dims:
            all_min_dims.sort()
            page_line_thickness_px = float(all_min_dims[len(all_min_dims) // 2])
        else:
            page_line_thickness_px = REFERENCE_LINE_THICKNESS_PX

    thickness_scale_factor = max(1.0, page_line_thickness_px / REFERENCE_LINE_THICKNESS_PX)
    # gap_tolerance_scaled = base * (1 + 0.3 * log(thickness / reference))
    log_ratio = math.log(max(0.1, page_line_thickness_px / REFERENCE_LINE_THICKNESS_PX))
    gap_tolerance_scaled = BASE_GAP_TOLERANCE * (1.0 + 0.3 * log_ratio)
    bridge_ar_threshold = 15.0 * page_aspect_ratio * 0.6
    row_col_tol_scale = BASE_ROW_COL_TOL * thickness_scale_factor

    return PageGeometryContext(
        page_aspect_ratio=page_aspect_ratio,
        page_line_thickness_px=page_line_thickness_px,
        thickness_scale_factor=thickness_scale_factor,
        bridge_ar_threshold=bridge_ar_threshold,
        gap_tolerance_scaled=gap_tolerance_scaled,
        row_col_tol_scale=row_col_tol_scale,
    )


# Style normalisation: impact of aspect ratio and line thickness on thresholds (tune for different drawing styles)
AR_FACTOR_NUMERIC_OFFSET = 0.08   # extra numeric_min on wide/tall pages (0 = disable)
AR_FACTOR_LABEL_BAR_OFFSET = 0.06  # extra label_heavy_bar on wide/tall (title blocks → INFO)
THICKNESS_HEADER_DELTA_OFFSET = 0.05  # extra header_delta_min per thickness scale above 0.5
THICKNESS_CELL_TO_LINE_BASE = 6.0   # cell_to_line_min base; divided by t_scale for thick lines
THICKNESS_CELL_TO_LINE_EXTRA = 4.0   # added divisor term for thick-line drawings

# Geometric score scaling: aspect ratio and line thickness scale all geometric/structural score contributions.
# Wide/tall or thick-line pages get geometric rules weighted more (or less) so structure adapts to drawing style.
GEO_AR_WEIGHT = 0.15        # geometric_scale += this * ar_factor (ar_factor from log1p(|ar-1|))
GEO_THICKNESS_WEIGHT = 0.10  # geometric_scale += this * (thickness_scale - 1)
GEO_SCALE_MIN = 0.75
GEO_SCALE_MAX = 1.35


def _geometric_scale_from_page_ctx(page_ctx: "PageGeometryContext") -> float:
    """Scale factor for geometric/structural score contributions; aspect ratio and line thickness adapt impact."""
    ar = page_ctx.page_aspect_ratio
    t_scale = page_ctx.thickness_scale_factor
    ar_factor = min(1.0, max(0.0, math.log1p(abs(ar - 1.0))))
    scale = 1.0 + GEO_AR_WEIGHT * ar_factor + GEO_THICKNESS_WEIGHT * (t_scale - 1.0)
    return max(GEO_SCALE_MIN, min(GEO_SCALE_MAX, scale))
# Structural: grid regularity/completeness (computed from drawing)
# Minimum completeness/regularity for grid to be eligible for TABLE (below → INFO).
# 0.6 allows real tables with some merged cells or header-row size variation to pass.
TABLE_GRID_MIN_COMPLETENESS = 0.6
TABLE_GRID_MIN_REGULARITY = 0.6
COMPLETENESS_MIN_BASE = 0.85   # used for dynamic thresholds elsewhere
REGULARITY_MIN_BASE = 0.40     # used for dynamic thresholds elsewhere
COMPLETENESS_THICKNESS_RELAX = 0.04   # thick-line: relax completeness threshold slightly
# Cell size variation: CV of mean row height / mean col width above this → irregular grid, INFO bias
CELL_SIZE_CV_THRESHOLD = 0.35  # was 0.28; relax so moderate variation does not force INFO

# Content-aware grid clustering: do not merge rects that are clearly table-like vs info-like
# (so tables touching title blocks stay separate and get correct completeness/regularity).
GRID_MERGE_TABLE_LIKE_NUMERIC_MIN = 0.25   # numeric_fraction >= this → table-like
GRID_MERGE_TABLE_LIKE_HEADER_DELTA_MIN = 0.15  # header_body_numeric_delta >= this → table-like
GRID_MERGE_INFO_LIKE_NUMERIC_MAX = 0.18   # numeric_fraction < this and label-heavy → info-like
GRID_MERGE_INFO_LIKE_LABEL_MIN = 0.12   # label_value_ratio >= this (with low numeric) → info-like

# Row-structure split: if a cluster has a row where cell count drops by this many vs previous row, split
# so table (many cols) and title block (few cols) can be separate grids.
GRID_SPLIT_COL_DROP = 3
# Dense grid: cells per (grid_area / page_area); tune empirically (tables >> sparse clusters).
DENSE_GRID_THRESHOLD = 50.0

# Regions: gap between rect bboxes larger than this (fraction of page height) splits into separate regions.
# Engineering drawings / work instructions can have many regions (table, title block, revision block, etc.).
REGION_MAX_GAP_FRAC = 0.05


def _bbox_min_distance(bbox_a: Tuple[float, float, float, float], bbox_b: Tuple[float, float, float, float]) -> float:
    """Minimum distance between two axis-aligned boxes. 0 if they overlap."""
    x1a, y1a, x2a, y2a = bbox_a
    x1b, y1b, x2b, y2b = bbox_b
    dx = 0.0 if (x2a >= x1b and x2b >= x1a) else min(abs(x2a - x1b), abs(x2b - x1a))
    dy = 0.0 if (y2a >= y1b and y2b >= y1a) else min(abs(y2a - y1b), abs(y2b - y1a))
    return math.sqrt(dx * dx + dy * dy)


def _compute_regions(
    rectangles: List[Rectangle],
    page_w: float,
    page_h: float,
) -> Tuple[Dict[str, str], Dict[str, Tuple[float, float, float, float]]]:
    """
    Cluster rectangles into regions by spatial gap. Two rects are in the same region if
    the min distance between their bboxes is < REGION_MAX_GAP_FRAC * page_height.
    Returns (poly_id -> region_id, region_id -> (x1,y1,x2,y2) bbox of region).
    No limit on number of regions (engineering drawings / work instructions can have many).
    """
    if not rectangles or page_h <= 0:
        return {}, {}
    max_gap = REGION_MAX_GAP_FRAC * page_h
    poly_ids = [r.poly_id for r in rectangles if r.poly_id and r.bbox and len(r.bbox) == 4]
    if not poly_ids:
        return {}, {}
    bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    for r in rectangles:
        if not r.poly_id or not r.bbox or len(r.bbox) != 4:
            continue
        x1, y1, x2, y2 = float(r.bbox[0]), float(r.bbox[1]), float(r.bbox[2]), float(r.bbox[3])
        if x2 <= x1 or y2 <= y1:
            x2, y2 = x1 + abs(x2 - x1 or 1), y1 + abs(y2 - y1 or 1)
        bboxes[r.poly_id] = (x1, y1, x2, y2)

    parent: Dict[str, str] = {}

    def find(p: str) -> str:
        if p not in parent:
            parent[p] = p
        if parent[p] != p:
            parent[p] = find(parent[p])
        return parent[p]

    def union(a: str, b: str) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    for i, a in enumerate(poly_ids):
        if a not in bboxes:
            continue
        for b in poly_ids[i + 1 :]:
            if b not in bboxes:
                continue
            if _bbox_min_distance(bboxes[a], bboxes[b]) > max_gap:
                continue
            union(a, b)

    comps: Dict[str, List[str]] = defaultdict(list)
    for p in poly_ids:
        comps[find(p)].append(p)
    region_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    poly_to_region: Dict[str, str] = {}
    for idx, (root, pids) in enumerate(comps.items()):
        rid = f"R{idx}"
        x1_min = y1_min = float("inf")
        x2_max = y2_max = float("-inf")
        for pid in pids:
            poly_to_region[pid] = rid
            x1, y1, x2, y2 = bboxes.get(pid, (0, 0, 0, 0))
            x1_min = min(x1_min, x1)
            y1_min = min(y1_min, y1)
            x2_max = max(x2_max, x2)
            y2_max = max(y2_max, y2)
        region_bboxes[rid] = (x1_min, y1_min, x2_max, y2_max)
    return poly_to_region, region_bboxes


def _compute_column_alignment_proxy(
    cell_ids: List[str],
    pid_to_rc: Dict[str, Tuple[int, int]],
    centres: Dict[str, Tuple[float, float]],
) -> float:
    """
    Quick alignment score from x-position consistency per column.
    Returns 0.0-1.0 (higher = more aligned). Used before split to keep dense tables unified.
    """
    cols: Dict[int, List[float]] = defaultdict(list)
    for pid in cell_ids:
        if pid in pid_to_rc and pid in centres:
            _, col_idx = pid_to_rc[pid]
            x, _ = centres[pid]
            cols[col_idx].append(x)
    cvs: List[float] = []
    for col_x_vals in cols.values():
        if len(col_x_vals) >= 2:
            mean_x = sum(col_x_vals) / len(col_x_vals)
            var = sum((x - mean_x) ** 2 for x in col_x_vals) / len(col_x_vals)
            std_x = math.sqrt(var)
            cv = std_x / mean_x if mean_x != 0 else 1.0
            cvs.append(cv)
    if not cvs:
        return 0.0
    mean_cv = sum(cvs) / len(cvs)
    return max(0.0, 1.0 - min(1.0, mean_cv * 2))


def _split_cluster_by_row_structure(
    grid_cell_ids: List[str],
    centres: Dict[str, Tuple[float, float]],
    page_h: float,
) -> List[List[str]]:
    """
    If the cluster has a sharp drop in cells-per-row (e.g. table rows then title block rows),
    split into two sub-clusters so each gets its own grid and correct completeness.
    Before splitting: if cluster is dense and aligned (e.g. 5x7 BOM), keep as one grid.
    Returns [cluster] or [sub1, sub2] when both parts form at least 2x2.
    """
    if len(grid_cell_ids) < 4:
        return [grid_cell_ids]
    gr, gc, pid_to_rc = _infer_row_col_for_cluster(grid_cell_ids, centres, page_h)
    if gr < 3:
        return [grid_cell_ids]
    row_counts: Dict[int, int] = defaultdict(int)
    for pid in grid_cell_ids:
        ri, _ = pid_to_rc.get(pid, (-1, -1))
        if 0 <= ri < gr:
            row_counts[ri] += 1
    row_lengths = [row_counts[i] for i in range(gr)]
    split_at: Optional[int] = None
    for i in range(1, gr):
        if row_lengths[i] <= row_lengths[i - 1] - GRID_SPLIT_COL_DROP:
            split_at = i
            break
    if split_at is not None and len(grid_cell_ids) >= 8:
        col_alignment = _compute_column_alignment_proxy(grid_cell_ids, pid_to_rc, centres)
        if col_alignment >= 0.7 and len(grid_cell_ids) >= 12:
            logger.debug(
                "Grid split prevented: cells=%d, row_lengths=%s, alignment=%.2f, kept_unified=True",
                len(grid_cell_ids), row_lengths, col_alignment,
            )
            return [grid_cell_ids]
    if split_at is None:
        return [grid_cell_ids]
    sub1 = [pid for pid in grid_cell_ids if pid_to_rc.get(pid, (-1, -1))[0] < split_at]
    sub2 = [pid for pid in grid_cell_ids if pid_to_rc.get(pid, (-1, -1))[0] >= split_at]
    gr1, gc1, _ = _infer_row_col_for_cluster(sub1, centres, page_h)
    gr2, gc2, _ = _infer_row_col_for_cluster(sub2, centres, page_h)
    if gr1 >= 2 and gc1 >= 2 and gr2 >= 2 and gc2 >= 2:
        logger.debug(
            "Grid split: cells=%d -> %d+%d, row_lengths=%s, split_at=%d",
            len(grid_cell_ids), len(sub1), len(sub2), row_lengths, split_at,
        )
        return [sub1, sub2]
    return [grid_cell_ids]


def _content_compatible_for_grid_merge(
    fa: Dict[str, Any], fb: Dict[str, Any]
) -> bool:
    """
    Return True if two rects are content-compatible for grouping into the same grid.
    Return False when one is clearly table-like (numeric/header signal) and the other
    is clearly info-like (label-heavy, low numeric), so we don't merge tables with
    touching title blocks or notes.
    """
    def _table_like(f: Dict[str, Any]) -> bool:
        nf = f.get("numeric_fraction")
        hd = f.get("header_body_numeric_delta")
        if nf is None and hd is None:
            return False
        nf_val = float(nf) if nf is not None else 0.0
        hd_val = float(hd) if hd is not None else 0.0
        return (
            nf_val >= GRID_MERGE_TABLE_LIKE_NUMERIC_MIN
            or hd_val >= GRID_MERGE_TABLE_LIKE_HEADER_DELTA_MIN
        )

    def _info_like(f: Dict[str, Any]) -> bool:
        nf = f.get("numeric_fraction")
        lv = f.get("label_value_ratio")
        if nf is None and lv is None:
            return False
        nf_val = float(nf) if nf is not None else 1.0
        lv_val = float(lv) if lv is not None else 0.0
        return (
            nf_val < GRID_MERGE_INFO_LIKE_NUMERIC_MAX
            and lv_val >= GRID_MERGE_INFO_LIKE_LABEL_MIN
        )

    ta, tb = _table_like(fa), _table_like(fb)
    ia, ib = _info_like(fa), _info_like(fb)
    # Don't merge when one is table-like and the other is info-like
    if (ta and ib) or (tb and ia):
        return False
    return True


def _compute_page_content_stats(
    cluster_features: Dict[str, Dict[str, Any]],
    poly_ids_subset: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute content statistics so bars can adapt to document/region style.
    If poly_ids_subset is given, stats are for that subset only (e.g. one region).
    """
    if poly_ids_subset is not None:
        feats_iter = (cluster_features.get(pid, {}) for pid in poly_ids_subset)
    else:
        feats_iter = cluster_features.values()
    numerics: List[float] = []
    label_vals: List[float] = []
    for feat in feats_iter:
        n = feat.get("grid_numeric_fraction")
        if n is not None:
            numerics.append(float(n))
        else:
            v = feat.get("numeric_fraction")
            if v is not None:
                numerics.append(float(v))
        lv = feat.get("grid_label_value_ratio")
        if lv is not None:
            label_vals.append(float(lv))
        else:
            v = feat.get("label_value_ratio")
            if v is not None:
                label_vals.append(float(v))
    numerics.sort()
    label_vals.sort()
    n_n, n_l = len(numerics), len(label_vals)
    return {
        "page_median_numeric": float(numerics[n_n // 2]) if n_n else 0.5,
        "page_median_label_value": float(label_vals[n_l // 2]) if n_l else 0.2,
        "page_p75_numeric": float(numerics[int(0.75 * n_n)]) if n_n else 0.6,
        "page_p75_label_value": float(label_vals[int(0.75 * n_l)]) if n_l else 0.3,
    }


def _dynamic_table_thresholds(page_ctx: PageGeometryContext) -> Dict[str, float]:
    """
    Compute TABLE-vs-INFO_CLUSTER thresholds that adapt to page aspect ratio
    and line thickness so scoring is robust across drawing styles.
    All values are derived from the actual drawing (page_ctx), not hardcoded styles.
    """
    ar = page_ctx.page_aspect_ratio
    t_scale = page_ctx.thickness_scale_factor
    # On wide or tall pages, require slightly stronger content to accept TABLE
    ar_factor = min(1.0, max(0.0, math.log1p(abs(ar - 1.0))))
    numeric_min = 0.52 + AR_FACTOR_NUMERIC_OFFSET * ar_factor  # base: only clear table content → TABLE
    header_delta_min = 0.18 + THICKNESS_HEADER_DELTA_OFFSET * min(1.0, t_scale - 0.5)
    # Thick-line drawings: cell_to_line_ratio can be lower (cells still look like cells)
    cell_to_line_min = THICKNESS_CELL_TO_LINE_BASE + THICKNESS_CELL_TO_LINE_EXTRA / max(0.5, t_scale)
    # Label-heavy bar: above this label_value_ratio with low header = info cluster
    label_heavy_bar = 0.16 + AR_FACTOR_LABEL_BAR_OFFSET * ar_factor
    # Structural: completeness/regularity thresholds (thick lines tolerate slightly more jagged/incomplete)
    completeness_min = max(0.75, COMPLETENESS_MIN_BASE - COMPLETENESS_THICKNESS_RELAX * (t_scale - 1.0))
    regularity_min = REGULARITY_MIN_BASE
    return {
        "table_numeric_min": numeric_min,
        "table_header_delta_min": header_delta_min,
        "cell_to_line_min": cell_to_line_min,
        "label_heavy_bar": label_heavy_bar,
        "completeness_min": completeness_min,
        "regularity_min": regularity_min,
    }


def _infer_row_col_for_cluster(
    poly_ids: List[str],
    centres: Dict[str, Tuple[float, float]],
    page_h: float,
) -> Tuple[int, int, Dict[str, Tuple[int, int]]]:
    """
    Assign row/col indices to rects in a spatial cluster by alignment (y → row, x → col).
    Returns (grid_rows, grid_cols, pid -> (ri, ci)).
    """
    if not poly_ids or not centres:
        return 0, 0, {}
    tol = max(2.0, page_h * 0.015)
    sorted_by_y = sorted(poly_ids, key=lambda p: centres[p][1])
    rows: List[List[str]] = []
    for pid in sorted_by_y:
        cy = centres[pid][1]
        if not rows:
            rows.append([pid])
            continue
        last_cy = centres[rows[-1][0]][1]
        if abs(cy - last_cy) <= tol:
            rows[-1].append(pid)
        else:
            rows.append([pid])
    grid_rows = len(rows)
    grid_cols = max(len(r) for r in rows) if rows else 0
    pid_to_rc: Dict[str, Tuple[int, int]] = {}
    for ri, row_pids in enumerate(rows):
        sorted_by_x = sorted(row_pids, key=lambda p: centres[p][0])
        for ci, pid in enumerate(sorted_by_x):
            pid_to_rc[pid] = (ri, ci)
    return grid_rows, grid_cols, pid_to_rc


def _compute_grid_bbox(
    grid_cell_ids: List[str],
    rects_by_poly: Dict[str, Rectangle],
) -> Tuple[float, float, float, float]:
    """Bounding box (x1, y1, x2, y2) of all cell rects. Returns (0,0,0,0) if empty."""
    x1_min, y1_min, x2_max, y2_max = float("inf"), float("inf"), float("-inf"), float("-inf")
    for pid in grid_cell_ids:
        rect = rects_by_poly.get(pid)
        if not rect or not rect.bbox or len(rect.bbox) != 4:
            continue
        x1, y1, x2, y2 = rect.bbox[0], rect.bbox[1], rect.bbox[2], rect.bbox[3]
        x1_min = min(x1_min, float(x1))
        y1_min = min(y1_min, float(y1))
        x2_max = max(x2_max, float(x2))
        y2_max = max(y2_max, float(y2))
    if x1_min == float("inf"):
        return (0.0, 0.0, 0.0, 0.0)
    return (x1_min, y1_min, x2_max, y2_max)


def _compute_grid_connectivity(
    cell_ids: List[str],
    pid_to_rc: Dict[str, Tuple[int, int]],
) -> float:
    """Fraction of cells with all 4 neighbors present (indicates dense table)."""
    rc_set = set(pid_to_rc[pid] for pid in cell_ids if pid in pid_to_rc)
    full_neighbors = 0
    for pid in cell_ids:
        if pid not in pid_to_rc:
            continue
        r, c = pid_to_rc[pid]
        neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        if all(nb in rc_set for nb in neighbors):
            full_neighbors += 1
    return full_neighbors / len(cell_ids) if cell_ids else 0.0


def _adaptive_grid_max_dist(
    page_diag: float,
    page_ctx: Optional["PageGeometryContext"] = None,
) -> float:
    """
    Max merge distance for grid formation, adaptive to aspect ratio and line thickness
    so different drawing styles (wide title blocks, thick lines) merge grids more accurately.
    """
    max_dist = GRID_SPATIAL_CLUSTER_FRAC * page_diag
    if page_ctx is None:
        return max_dist
    ar = page_ctx.page_aspect_ratio
    t_scale = page_ctx.thickness_scale_factor
    # Wider/taller pages: allow larger merge distance (e.g. two-column title blocks with big gap)
    if ar >= 1.0:
        ar_factor = min(1.0, math.log1p(ar - 1.0))
        max_dist *= 1.0 + GRID_MAX_DIST_AR_SCALE * ar_factor
    else:
        ar_factor = min(1.0, math.log1p(1.0 / ar - 1.0)) if ar > 0 else 0.0
        max_dist *= 1.0 + GRID_MAX_DIST_AR_SCALE * ar_factor
    # Thicker lines: cells are often further apart in drawings
    if t_scale > 1.0:
        thickness_factor = min(1.35, 1.0 + GRID_MAX_DIST_THICKNESS_SCALE * (t_scale - 1.0))
        max_dist *= thickness_factor
    return max_dist


def _compute_grid_level_content(
    cluster_features: Dict[str, Dict[str, Any]],
    rectangles: List[Rectangle],
    page_w: float,
    page_h: float,
    geometry_only: bool = False,
    page_ctx: Optional["PageGeometryContext"] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Infer grid groupings from geometry for ALL rects (no v2 grid tags).
    When geometry_only=True (early run): merge by distance only; no content-compatibility.
    Otherwise: only merge rects that are spatially close and not opposite (table vs info).
    max_dist is adaptive to page aspect ratio and line thickness when page_ctx is provided.
    Returns per-poly_id grid geometry + content.
    """
    page_diag = math.sqrt(page_w * page_w + page_h * page_h) if (page_w > 0 and page_h > 0) else 1.0
    max_dist = _adaptive_grid_max_dist(page_diag, page_ctx)

    rects_by_poly = {r.poly_id: r for r in rectangles if r.poly_id}
    all_poly_ids = list(rects_by_poly.keys())
    centres: Dict[str, Tuple[float, float]] = {}
    for poly_id in all_poly_ids:
        rect = rects_by_poly.get(poly_id)
        if not rect or not rect.bbox:
            continue
        x1, y1, x2, y2 = rect.bbox
        centres[poly_id] = (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def distance(pid1: str, pid2: str) -> float:
        c1 = centres.get(pid1)
        c2 = centres.get(pid2)
        if not c1 or not c2:
            return float("inf")
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    parent: Dict[str, str] = {}

    def find(p: str) -> str:
        if p not in parent:
            parent[p] = p
        if parent[p] != p:
            parent[p] = find(parent[p])
        return parent[p]

    def union(a: str, b: str) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    # Optional: in title-block zone (bottom of page), allow larger merge distance so label|value
    # columns form one grid. Disabled (RELAX=1.0) to avoid regressions on 152021/2d_assembly;
    # enable by setting GRID_TITLE_BLOCK_DISTANCE_RELAX > 1.0 and tuning GRID_TITLE_BLOCK_ZONE_Y_MIN.
    for i, a in enumerate(all_poly_ids):
        if a not in centres:
            continue
        ca = centres[a]
        for b in all_poly_ids[i + 1 :]:
            if b not in centres:
                continue
            cb = centres[b]
            dist_ab = distance(a, b)
            max_d = max_dist
            if (
                page_h > 0
                and GRID_TITLE_BLOCK_DISTANCE_RELAX > 1.0
                and ca[1] >= page_h * GRID_TITLE_BLOCK_ZONE_Y_MIN
                and cb[1] >= page_h * GRID_TITLE_BLOCK_ZONE_Y_MIN
            ):
                max_d = max_dist * GRID_TITLE_BLOCK_DISTANCE_RELAX
            if dist_ab > max_d:
                continue
            if not geometry_only and not _content_compatible_for_grid_merge(
                cluster_features.get(a, {}), cluster_features.get(b, {})
            ):
                continue
            union(a, b)

    comps: Dict[str, List[str]] = defaultdict(list)
    for p in all_poly_ids:
        comps[find(p)].append(p)
    raw_clusters = [c for c in comps.values() if len(c) >= 2]

    # Split clusters by row structure so table (many cols) and title block (few cols) stay separate
    clusters = []
    for c in raw_clusters:
        for sub in _split_cluster_by_row_structure(c, centres, page_h):
            if len(sub) >= 2:
                clusters.append(sub)

    def _content_table_score(pid: str) -> float:
        f = cluster_features.get(pid, {})
        nf = float(f.get("numeric_fraction", 0.0))
        hbd = float(f.get("header_body_numeric_delta", 0.0))
        return min(1.0, nf + 0.4 * min(1.0, hbd / 0.25))

    out: Dict[str, Dict[str, Any]] = {}
    assigned: set[str] = set()

    for grid_cell_ids in clusters:
        gr, gc, pid_to_rc = _infer_row_col_for_cluster(grid_cell_ids, centres, page_h)
        if gr < 2 or gc < 2:
            continue
        rc_to_pid: Dict[Tuple[int, int], str] = {v: k for k, v in pid_to_rc.items()}
        expected_cells = gr * gc
        grid_completeness = min(1.0, len(grid_cell_ids) / expected_cells) if expected_cells > 0 else 1.0

        nf_vals = []
        hbd_vals = []
        lvr_vals = []
        for pid in grid_cell_ids:
            f = cluster_features.get(pid, {})
            if f.get("numeric_fraction") is not None:
                nf_vals.append(float(f["numeric_fraction"]))
            if f.get("header_body_numeric_delta") is not None:
                hbd_vals.append(float(f["header_body_numeric_delta"]))
            if f.get("label_value_ratio") is not None:
                lvr_vals.append(float(f["label_value_ratio"]))
        grid_numeric = _safe_mean(nf_vals) if nf_vals else 0.0
        grid_header_delta = max(hbd_vals) if hbd_vals else 0.0
        grid_label_value = _safe_mean(lvr_vals) if lvr_vals else 0.0

        # Cell size variation per row/col: high variation → irregular grid (not equal-size table)
        row_heights_by_idx: List[float] = [0.0] * gr
        row_counts: List[int] = [0] * gr
        col_widths_by_idx: List[float] = [0.0] * gc
        col_counts: List[int] = [0] * gc
        for pid in grid_cell_ids:
            rect = rects_by_poly.get(pid)
            if not rect or not rect.bbox or len(rect.bbox) != 4:
                continue
            x1, y1, x2, y2 = rect.bbox
            w = max(0.0, float(x2) - float(x1))
            h = max(0.0, float(y2) - float(y1))
            ri, ci = pid_to_rc.get(pid, (-1, -1))
            if 0 <= ri < gr:
                row_heights_by_idx[ri] += h
                row_counts[ri] += 1
            if 0 <= ci < gc:
                col_widths_by_idx[ci] += w
                col_counts[ci] += 1
        mean_heights = [
            row_heights_by_idx[i] / row_counts[i] if row_counts[i] > 0 else 0.0
            for i in range(gr)
        ]
        mean_widths = [
            col_widths_by_idx[j] / col_counts[j] if col_counts[j] > 0 else 0.0
            for j in range(gc)
        ]
        mean_heights = [x for x in mean_heights if x > 0]
        mean_widths = [x for x in mean_widths if x > 0]
        row_height_cv = _cv_from_values(mean_heights) if len(mean_heights) >= 2 else 0.0
        col_width_cv = _cv_from_values(mean_widths) if len(mean_widths) >= 2 else 0.0
        # Regularity: 1 = uniform sizes; 0 = very irregular
        combined_cv = (row_height_cv + col_width_cv) * 0.5 if (mean_heights and mean_widths) else 0.0
        grid_regularity_mean = max(0.0, min(1.0, 1.0 - min(1.0, combined_cv)))
        grid_cell_size_variation_high = (
            row_height_cv > CELL_SIZE_CV_THRESHOLD or col_width_cv > CELL_SIZE_CV_THRESHOLD
        )

        # Column alignment: tables align (low cv of left edges per column), info ragged
        col_left_edges: List[List[float]] = [[] for _ in range(gc)]
        for pid in grid_cell_ids:
            rect = rects_by_poly.get(pid)
            if not rect or not rect.bbox or len(rect.bbox) != 4:
                continue
            x1 = float(rect.bbox[0])
            ci = pid_to_rc.get(pid, (-1, -1))[1]
            if 0 <= ci < gc:
                col_left_edges[ci].append(x1)
        alignment_per_col = []
        for edges in col_left_edges:
            if len(edges) < 2:
                alignment_per_col.append(1.0)
            else:
                cv_left = _cv_from_values(edges)
                alignment_per_col.append(max(0.0, min(1.0, 1.0 - min(1.0, cv_left))))
        grid_column_alignment_score = _safe_mean(alignment_per_col) if alignment_per_col else 0.5

        border_nf = []
        interior_nf = []
        for pid in grid_cell_ids:
            ri, ci = pid_to_rc.get(pid, (-1, -1))
            nf = float(cluster_features.get(pid, {}).get("numeric_fraction", 0.0))
            if ri < 0:
                continue
            is_border = ri == 0 or ri == gr - 1 or ci == 0 or ci == gc - 1
            if is_border:
                border_nf.append(nf)
            else:
                interior_nf.append(nf)
        grid_border_numeric_mean = _safe_mean(border_nf) if border_nf else grid_numeric
        grid_interior_numeric_mean = _safe_mean(interior_nf) if interior_nf else grid_numeric

        neighbour_support: Dict[str, float] = {}
        for (ri, ci), pid in rc_to_pid.items():
            neighbours = [
                rc_to_pid.get((ri - 1, ci)),
                rc_to_pid.get((ri + 1, ci)),
                rc_to_pid.get((ri, ci - 1)),
                rc_to_pid.get((ri, ci + 1)),
            ]
            scores_n = [_content_table_score(n) for n in neighbours if n is not None]
            neighbour_support[pid] = _safe_mean(scores_n) if scores_n else 0.5

        # Stable id for this grid (same for all cells) for grid-first classification
        grid_id = min(grid_cell_ids)

        # Density and connectivity for dense vs sparse classification
        grid_bbox = _compute_grid_bbox(grid_cell_ids, rects_by_poly)
        grid_area = (grid_bbox[2] - grid_bbox[0]) * (grid_bbox[3] - grid_bbox[1])
        page_area = page_w * page_h if (page_w > 0 and page_h > 0) else 1.0
        grid_density = (
            len(grid_cell_ids) * page_area / grid_area
            if grid_area > 0
            else 0.0
        )
        grid_connectivity = _compute_grid_connectivity(grid_cell_ids, pid_to_rc)

        for pid in grid_cell_ids:
            ri, ci = pid_to_rc.get(pid, (-1, -1))
            is_border_cell = (
                (ri == 0 or ri == gr - 1 or ci == 0 or ci == gc - 1)
                if (ri >= 0 and ci >= 0) else False
            )
            assigned.add(pid)
            out[pid] = {
                "is_part_of_grid": True,
                "grid_cell_count": len(grid_cell_ids),
                "grid_rows": gr,
                "grid_cols": gc,
                "grid_row_idx": ri,
                "grid_col_idx": ci,
                "grid_numeric_fraction": grid_numeric,
                "grid_header_body_delta": grid_header_delta,
                "grid_label_value_ratio": grid_label_value,
                "grid_completeness": grid_completeness,
                "grid_regularity_mean": grid_regularity_mean,
                "grid_cell_size_variation_high": grid_cell_size_variation_high,
                "grid_column_alignment_score": grid_column_alignment_score,
                "grid_row_height_cv": row_height_cv,
                "grid_col_width_cv": col_width_cv,
                "grid_id": grid_id,
                "is_border_cell": is_border_cell,
                "grid_border_numeric_mean": grid_border_numeric_mean,
                "grid_interior_numeric_mean": grid_interior_numeric_mean,
                "neighbour_table_support": neighbour_support.get(pid, 0.5),
                "grid_density": grid_density,
                "grid_connectivity": grid_connectivity,
            }

    non_grid = {
        "is_part_of_grid": False,
        "grid_cell_count": 0,
        "grid_rows": 0,
        "grid_cols": 0,
        "grid_row_idx": -1,
        "grid_col_idx": -1,
        "grid_id": None,
        "grid_column_alignment_score": 0.0,
        "grid_density": 0.0,
        "grid_connectivity": 0.0,
    }
    for poly_id in all_poly_ids:
        if poly_id not in assigned:
            out[poly_id] = dict(non_grid)
    return out


def _classify_grid(
    feat: Dict[str, Any],
    table_numeric_min: float,
    table_header_delta_min: float,
    label_heavy_bar: float,
    title_block_numeric_max: float,
    title_block_header_delta_max: float,
) -> str:
    """
    Grid-first: classify the whole grid as TABLE or INFO_CLUSTER.
    Deterministic rules; no per-cell scoring. Used to propagate one label to all cells.
    """
    grid_rows = int(feat.get("grid_rows", 0))
    grid_cols = int(feat.get("grid_cols", 0))
    grid_completeness = float(feat.get("grid_completeness", 1.0))
    grid_regularity_mean = float(feat.get("grid_regularity_mean", 0.5))
    grid_cell_size_variation_high = bool(feat.get("grid_cell_size_variation_high", False))
    grid_numeric = float(feat.get("grid_numeric_fraction", 0.0))
    grid_header_delta = float(feat.get("grid_header_body_delta", 0.0))
    grid_label_value = float(feat.get("grid_label_value_ratio", 0.0))
    structural_position = float(feat.get("structural_position", 0.0))
    grid_column_alignment = float(feat.get("grid_column_alignment_score", 0.0))
    grid_density = float(feat.get("grid_density", 0.0))
    grid_connectivity = float(feat.get("grid_connectivity", 0.0))

    # Not a proper grid → INFO
    if grid_rows < 2 or grid_cols < 2:
        return "INFO_CLUSTER"

    grid_cell_count = int(feat.get("grid_cell_count", 0))
    # Content-first TABLE path (v2-style): alignment + content prove table in title-block only; skip comp/reg bar so BOM/title-block tables are easier to spot (avoid flipping drawing callouts)
    if (
        structural_position >= 0.6
        and grid_cell_count >= 6
        and grid_column_alignment >= 0.7
        and (grid_numeric >= 0.2 or grid_header_delta >= 0.15)
    ):
        return "TABLE"
    # Completeness/regularity bar: relax only for title-block aligned grids (8+ cells) so drawing-area grids keep strict bar
    if (
        structural_position >= 0.6
        and grid_column_alignment >= 0.7
        and grid_cell_count >= 8
    ):
        min_comp = 0.45
        min_reg = 0.45
    elif grid_cell_count >= 20:
        min_comp = 0.45
        min_reg = 0.5
    else:
        min_comp = TABLE_GRID_MIN_COMPLETENESS
        min_reg = TABLE_GRID_MIN_REGULARITY
    if grid_completeness < min_comp or grid_regularity_mean < min_reg:
        return "INFO_CLUSTER"
    # Cell size variation: force INFO only when content is weak AND grid is small or irregular (use same min_comp/min_reg as above)
    # Skip for very aligned title-block grids (e.g. 175985 bottom 5-cell block) so they can reach TABLE paths
    if grid_cell_size_variation_high and grid_numeric < table_numeric_min and grid_header_delta < table_header_delta_min:
        if (structural_position >= 0.6 and grid_column_alignment >= 0.95):
            pass  # allow very aligned title-block grids to continue to TABLE paths
        elif grid_cell_count < 6 or grid_completeness < min_comp or grid_regularity_mean < min_reg:
            return "INFO_CLUSTER"

    # Table: sufficient numeric and/or header/body structure
    if grid_numeric >= table_numeric_min or grid_header_delta >= table_header_delta_min:
        return "TABLE"
    # Large grid with moderate numeric (e.g. full-page BOM): treat as TABLE so one dominant table is not forced to INFO
    if grid_cell_count >= 40 and grid_numeric >= 0.35:
        return "TABLE"
    # Large structured grid with no OCR content: use same min_comp/min_reg so 152021 BOM (35 cells, comp 0.49) can be TABLE
    if grid_cell_count >= 20 and grid_completeness >= min_comp and grid_regularity_mean >= min_reg and structural_position < 0.75:
        return "TABLE"
    # Medium structured grid (e.g. 2×4): TABLE when not title-block and geometry is regular
    if grid_cell_count >= 6 and grid_rows >= 2 and grid_cols >= 2 and grid_completeness >= 0.6 and grid_regularity_mean >= 0.6 and structural_position < 0.7:
        return "TABLE"

    ar_entropy = float(feat.get("ar_entropy", 0.5))

    # Bottom-page big BOMs: allow TABLE when aligned and not label-heavy (content-aware position)
    if (
        grid_cell_count >= 20
        and structural_position >= 0.75
        and grid_completeness >= 0.45
        and grid_regularity_mean >= 0.35
        and grid_column_alignment >= 0.7
        and ar_entropy <= 0.6
        and (
            grid_label_value < label_heavy_bar
            or grid_numeric >= 0.2
            or grid_header_delta >= 0.15
        )
    ):
        return "TABLE"

    # Medium aligned grids (8–19 cells): catch aligned grids that fail strict regularity
    if (
        8 <= grid_cell_count < 20
        and grid_completeness >= 0.5
        and grid_regularity_mean >= 0.35
        and grid_column_alignment >= 0.7
        and ar_entropy <= 0.6
    ):
        if structural_position < 0.7:
            return "TABLE"
        if grid_numeric >= 0.2 or grid_header_delta >= 0.15:
            return "TABLE"

    # Title-block small tables: content-aware; 2×2 excluded (grid_cell_count >= 5)
    if (
        structural_position >= 0.6
        and grid_cell_count >= 5
        and grid_rows >= 2
        and grid_cols >= 2
        and grid_completeness >= 0.5
        and grid_regularity_mean >= 0.5
        and grid_column_alignment >= 0.7
        and ar_entropy <= 0.6
    ):
        # 5–8 cell mini tables: require some table-like content, or very high alignment (e.g. 175985 bottom 5-cell block)
        if grid_cell_count <= 8:
            if grid_numeric >= 0.2 or grid_header_delta >= 0.15:
                return "TABLE"
            # Very high alignment + regular: treat as table even with no OCR content (title-block metadata grid)
            if grid_column_alignment >= 0.95:
                return "TABLE"
        else:
            return "TABLE"

    # Title-block aligned table without density requirement (content or completeness proves table; v2-style table boost)
    if (
        structural_position >= 0.6
        and grid_cell_count >= 8
        and grid_column_alignment >= 0.7
        and (
            grid_completeness >= 0.45
            or grid_numeric >= 0.2
            or grid_header_delta >= 0.15
        )
    ):
        return "TABLE"
    # Dense title-block table: aligned BOMs/parts lists in title block (dense grid network vs sparse clusters)
    if (
        structural_position >= 0.6
        and grid_cell_count >= 8
        and grid_column_alignment >= 0.7
        and grid_density >= DENSE_GRID_THRESHOLD
        and (grid_connectivity >= 0.5 or grid_completeness >= 0.55)
    ):
        return "TABLE"

    # Title-block pattern: bottom, label-heavy, ragged or incomplete → INFO (strengthened so aligned tables are not blocked)
    if (
        structural_position >= TITLE_BLOCK_POSITION_MIN
        and grid_label_value >= label_heavy_bar
        and grid_header_delta < title_block_header_delta_max
        and grid_numeric < title_block_numeric_max
        and (grid_column_alignment <= 0.6 or grid_completeness < 0.4)
        and ar_entropy >= 0.5
    ):
        return "INFO_CLUSTER"

    return "INFO_CLUSTER"


def _update_grid_content_aggregates(cluster_features: Dict[str, Dict[str, Any]]) -> None:
    """Refresh grid_numeric_fraction, grid_header_body_delta, grid_label_value_ratio per grid after content enrichment."""
    from collections import defaultdict
    by_grid: Dict[str, List[str]] = defaultdict(list)
    for poly_id, f in cluster_features.items():
        if f.get("is_part_of_grid") and f.get("grid_id") is not None:
            by_grid[f["grid_id"]].append(poly_id)
    for gid, pids in by_grid.items():
        nf_vals = []
        hbd_vals = []
        lvr_vals = []
        for pid in pids:
            f = cluster_features.get(pid, {})
            if f.get("numeric_fraction") is not None:
                nf_vals.append(float(f["numeric_fraction"]))
            if f.get("header_body_numeric_delta") is not None:
                hbd_vals.append(float(f["header_body_numeric_delta"]))
            if f.get("label_value_ratio") is not None:
                lvr_vals.append(float(f["label_value_ratio"]))
        grid_numeric = _safe_mean(nf_vals) if nf_vals else 0.0
        grid_header_delta = max(hbd_vals) if hbd_vals else 0.0
        grid_label_value = _safe_mean(lvr_vals) if lvr_vals else 0.0
        for pid in pids:
            cluster_features[pid]["grid_numeric_fraction"] = grid_numeric
            cluster_features[pid]["grid_header_body_delta"] = grid_header_delta
            cluster_features[pid]["grid_label_value_ratio"] = grid_label_value


def _has_query_presence(feat: Dict[str, Any], inputs: Any) -> bool:
    """Query presence: from data cleaning at the end. Stub until pipeline provides it."""
    return bool(feat.get("query_presence", False))


def _has_structural_marker(
    feat: Dict[str, Any],
    page_ctx: "PageGeometryContext",
    page_h: float,
) -> bool:
    """Unique structural markers (not part of grid) -> INFO. Dimension-like, title-block zone, line-like, no grid."""
    is_part_of_grid = bool(feat.get("is_part_of_grid", False))
    if is_part_of_grid:
        return False
    cell_ar_n = float(feat.get("cell_ar_normalized", 1.0))
    structural_position = float(feat.get("structural_position", 0.0))
    grid_cell_count = int(feat.get("grid_cell_count", 0))
    label_value_ratio = float(feat.get("label_value_ratio", 0.0))
    if cell_ar_n >= DIMENSION_AR_MIN or (cell_ar_n > 0 and cell_ar_n <= 1.0 / DIMENSION_AR_MIN):
        return True
    if structural_position >= TITLE_BLOCK_POSITION_MIN and label_value_ratio >= 0.15:
        return True
    if grid_cell_count <= 1:
        return True
    return False


def _has_two_sets_of_text_in_cell(
    poly_id: str,
    feat: Dict[str, Any],
    inputs: Any,
) -> bool:
    """
    Hard rule: a cell that has 2+ lines and label+value character (e.g. "DRAWN" / "P.S.0.0" or
    "TITLE: ROLLER ASSEMBLY") must not be TABLE — it is a compound info cell. Applied only in
    title-block zone so BOM/data cells are not flipped. Returns True when the cell should be
    forced to INFO_CLUSTER.
    """
    structural_position = float(feat.get("structural_position", 0.0))
    if structural_position < 0.6:
        return False  # only apply in title-block area; avoid flipping data-table cells
    n_lines = feat.get("n_lines_in_rect")
    if n_lines is None and getattr(inputs, "cluster_lines", None):
        n_lines = sum(
            1 for ln in inputs.cluster_lines
            if getattr(ln, "poly_id", None) == poly_id
        )
    if not n_lines or n_lines < 2:
        return False
    label_value_ratio = float(feat.get("label_value_ratio", 0.0))
    if label_value_ratio >= 0.25:
        return True
    for ln in getattr(inputs, "cluster_lines", None) or []:
        if getattr(ln, "poly_id", None) != poly_id:
            continue
        text = (getattr(ln, "text_norm", None) or getattr(ln, "text", None) or "").strip()
        if text and LABEL_VALUE_RE.match(text):
            return True
    return False


def _survival_feature_vector(feat: Dict[str, Any]) -> Dict[str, float]:
    """
    Build feature vector for prototype distance. Normalised 0-1 where possible.
    When the cell is in the title-block zone and has metadata-like per-cell content (no numeric,
    label-heavy), we use per-cell numeric/label_value_ratio/header_delta so the cell's own
    characteristics can disqualify it from the grid's label (e.g. grid=TABLE but cell scores INFO).
    """
    completeness = float(feat.get("grid_completeness", 0.0))
    regularity = float(feat.get("grid_regularity_mean", 0.0))
    grid_cell_count = int(feat.get("grid_cell_count", 0))
    grid_size_norm = min(1.0, grid_cell_count / 8.0) if grid_cell_count > 1 else 0.0
    per_cell_numeric = float(feat.get("numeric_fraction", 0.0))
    per_cell_header_delta = float(feat.get("header_body_numeric_delta", 0.0))
    per_cell_label_value = float(feat.get("label_value_ratio", 0.0))
    structural_position = float(feat.get("structural_position", 0.0))
    grid_regularity = float(feat.get("grid_regularity_mean", 1.0))
    grid_alignment = float(feat.get("grid_column_alignment_score", 1.0))
    # Use per-cell content when cell can disqualify from grid: title block + no numeric, and either
    # label_value_ratio high (label-like) or grid is not table-like (low regularity/alignment).
    use_per_cell_content = (
        structural_position >= TITLE_BLOCK_POSITION_MIN
        and per_cell_numeric <= SURVIVAL_TITLE_BLOCK_USE_PER_CELL_NUMERIC_MAX
        and (
            per_cell_label_value >= SURVIVAL_TITLE_BLOCK_USE_PER_CELL_LABEL_MIN
            or grid_regularity < TITLE_BLOCK_GRID_REGULARITY_TABLE_MIN
            or grid_alignment < TITLE_BLOCK_GRID_ALIGNMENT_TABLE_MIN_BASE
        )
    )
    if use_per_cell_content:
        numeric = per_cell_numeric
        header_delta = per_cell_header_delta
        label_value_ratio = per_cell_label_value
    else:
        numeric = float(feat.get("grid_numeric_fraction", per_cell_numeric))
        header_delta = float(feat.get("grid_header_body_delta", per_cell_header_delta))
        label_value_ratio = float(feat.get("grid_label_value_ratio", per_cell_label_value))
    ar_entropy = float(feat.get("ar_entropy", 0.5))
    cell_clustering_coef = float(feat.get("cell_clustering_coef", 0.5))
    column_alignment_score = float(feat.get("grid_column_alignment_score", 0.5))
    token_repetition = float(feat.get("token_repetition_across_rows", 0.0))
    vertical_text_sim = float(feat.get("vertical_text_similarity", 0.0))
    return {
        "completeness": max(0.0, min(1.0, completeness)),
        "regularity": max(0.0, min(1.0, regularity)),
        "grid_size_norm": grid_size_norm,
        "numeric": max(0.0, min(1.0, numeric)),
        "header_delta": max(0.0, min(1.0, header_delta)),
        "label_value_ratio": max(0.0, min(1.0, label_value_ratio)),
        "ar_entropy": max(0.0, min(1.0, ar_entropy)),
        "cell_clustering_coef": max(0.0, min(1.0, cell_clustering_coef)),
        "column_alignment_score": max(0.0, min(1.0, column_alignment_score)),
        "token_repetition": max(0.0, min(1.0, token_repetition)),
        "vertical_text_sim": max(0.0, min(1.0, vertical_text_sim)),
    }


def _weighted_euclidean_distance(
    vec: Dict[str, float],
    prototype: Dict[str, float],
    geom_weight: float,
    content_weight: float,
    geom_keys: List[str],
    content_keys: List[str],
) -> float:
    """Weighted Euclidean: geometric features * geom_weight, content * content_weight."""
    geom_sum = 0.0
    for k in geom_keys:
        if k in prototype and k in vec:
            geom_sum += (vec[k] - prototype[k]) ** 2
    content_sum = 0.0
    for k in content_keys:
        if k in prototype and k in vec:
            content_sum += (vec[k] - prototype[k]) ** 2
    return math.sqrt(geom_weight * geom_sum + content_weight * content_sum)


def _adjusted_threshold(
    page_ctx: "PageGeometryContext",
    base: float,
    for_table: bool,
) -> float:
    """Rescale threshold: base * (1 + line_thickness_norm) * (1 + aspect_ratio_penalty). Thicker -> tighter; extreme AR -> looser for INFO, tighter for TABLE."""
    t_norm = max(0.0, page_ctx.thickness_scale_factor - 1.0)
    ar = page_ctx.page_aspect_ratio
    ar_penalty = min(1.0, max(0.0, math.log1p(abs(ar - 1.0))))
    if for_table:
        return base * (1.0 + 0.15 * t_norm) * (1.0 + 0.1 * ar_penalty)
    return base * (1.0 + 0.1 * t_norm) * (1.0 + 0.15 * ar_penalty)


def _effective_title_block_alignment_min(page_ctx: "PageGeometryContext") -> float:
    """
    Alignment bar for title-block override, adaptive to aspect ratio and line thickness.
    Never larger than 0.72 so 2d_assembly (align=0.736) never triggers on the alignment arm.
    """
    ar = page_ctx.page_aspect_ratio
    t_scale = page_ctx.thickness_scale_factor
    ar_factor = min(1.0, math.log1p(abs(ar - 1.0))) if ar > 0 else 0.0
    scale = (1.0 + ALIGN_AR_SCALE * ar_factor) * (1.0 + ALIGN_THICKNESS_SCALE * max(0.0, t_scale - 1.0))
    raw = TITLE_BLOCK_GRID_ALIGNMENT_TABLE_MIN_BASE * scale
    return min(TITLE_BLOCK_GRID_ALIGNMENT_TABLE_MIN_BASE, max(0.0, raw))


def _classify_survival(
    feat: Dict[str, Any],
    page_ctx: "PageGeometryContext",
    grid_label: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    """
    Survival mode: hard rules first (handled by caller), then prototype distance + confidence margin.
    Returns (label, debug_dict). debug_dict has d_table, d_info, margin, thresh_table, thresh_info, grid_label, winner.
    """
    vec = _survival_feature_vector(feat)
    proto_t = {k: v for k, v in PROTOTYPE_TABLE.items() if k in vec}
    proto_i = {k: v for k, v in PROTOTYPE_INFO.items() if k in vec}
    geom_keys = [k for k in SURVIVAL_GEOMETRIC_KEYS if k in vec and (k in proto_t or k in proto_i)]
    content_keys = [k for k in SURVIVAL_CONTENT_KEYS if k in vec and (k in proto_t or k in proto_i)]
    d_table = _weighted_euclidean_distance(
        vec, proto_t, SURVIVAL_GEOMETRIC_WEIGHT, SURVIVAL_CONTENT_WEIGHT, geom_keys, content_keys
    )
    d_info = _weighted_euclidean_distance(
        vec, proto_i, SURVIVAL_GEOMETRIC_WEIGHT, SURVIVAL_CONTENT_WEIGHT, geom_keys, content_keys
    )
    thresh_table = _adjusted_threshold(page_ctx, SURVIVAL_BASE_THRESHOLD, for_table=True)
    thresh_info = _adjusted_threshold(page_ctx, SURVIVAL_BASE_THRESHOLD, for_table=False)
    margin = d_info - d_table
    debug = {
        "d_table": round(d_table, 4),
        "d_info": round(d_info, 4),
        "margin": round(margin, 4),
        "thresh_table": round(thresh_table, 4),
        "thresh_info": round(thresh_info, 4),
        "grid_label": grid_label,
    }
    # TABLE from prototype distance only when in a proper grid (≥4 cells); prevents irregular 2×1 / single callouts from becoming TABLE
    is_part_of_grid = bool(feat.get("is_part_of_grid", False))
    grid_cell_count = int(feat.get("grid_cell_count", 0))
    structural_position = float(feat.get("structural_position", 0.0))
    table_ok_from_distance = (
        is_part_of_grid
        and grid_cell_count >= 4
        and not (
            grid_label == "INFO_CLUSTER"
            and grid_cell_count <= 6
            and structural_position > 0.6
        )
    )
    # Bonus for uniform structure (low ar_entropy): slightly relax TABLE threshold so aligned tables get adequate boost
    ar_entropy = vec.get("ar_entropy", 0.5)
    thresh_table_eff = thresh_table * (1.1 if ar_entropy < 0.35 else 1.0)
    if margin > SURVIVAL_CONFIDENCE_MARGIN and d_table <= thresh_table_eff and table_ok_from_distance:
        debug["winner"] = "TABLE"
        return "TABLE", debug
    if -margin > SURVIVAL_CONFIDENCE_MARGIN and d_info <= thresh_info:
        debug["winner"] = "INFO"
        return "INFO_CLUSTER", debug
    if grid_label in ("TABLE", "INFO_CLUSTER"):
        debug["winner"] = f"grid_label={grid_label}"
        return grid_label, debug
    debug["winner"] = "ABSTAIN"
    return "ABSTAIN", debug


def _add_survival_new_features(
    cluster_features: Dict[str, Dict[str, Any]],
    rectangles: List[Rectangle],
    inputs: DataCleaningInputsV2,
    page_ctx: PageGeometryContext,
) -> None:
    """Add: ar_entropy, cell_clustering_coef, font_weight_distribution (stub), token_repetition_across_rows, vertical_text_similarity."""
    rects_by_poly = {r.poly_id: r for r in (rectangles or []) if r.poly_id}
    page_h = float(inputs.page.height) if inputs.page and inputs.page.height else 1.0
    page_w = float(inputs.page.width) if inputs.page and getattr(inputs.page, "width", None) else 1.0

    for poly_id, feat in cluster_features.items():
        feat.setdefault("font_weight_distribution", 0.5)  # TODO: implement when OCR pipeline provides bold/regular

        rect = rects_by_poly.get(poly_id)
        if not rect or not rect.bbox:
            feat.setdefault("ar_entropy", 0.5)
            feat.setdefault("cell_clustering_coef", 0.5)
            feat.setdefault("token_repetition_across_rows", 0.0)
            feat.setdefault("vertical_text_similarity", 0.0)
            continue

        gid = feat.get("grid_id")
        is_grid = bool(feat.get("is_part_of_grid", False)) and gid is not None
        grid_pids = [p for p, f in cluster_features.items() if f.get("grid_id") == gid] if is_grid else [poly_id]

        ars = []
        centres_list = []
        for pid in grid_pids:
            r = rects_by_poly.get(pid)
            if not r or not r.bbox:
                continue
            x1, y1, x2, y2 = r.bbox
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if w > 0 and h > 0:
                ars.append((max(w, h) / min(w, h)))
            centres_list.append((0.5 * (x1 + x2), 0.5 * (y1 + y2)))

        if len(ars) >= 2:
            bins = 10
            hist = [0.0] * bins
            ar_min, ar_max = min(ars), max(ars)
            for a in ars:
                idx = min(bins - 1, int((a - ar_min) / (ar_max - ar_min + 1e-9) * bins))
                hist[idx] += 1.0
            total = sum(hist)
            if total > 0:
                probs = [h / total for h in hist]
                entropy = -sum(p * math.log(p + 1e-12) for p in probs)
                feat["ar_entropy"] = min(1.0, entropy / math.log(bins + 1))
            else:
                feat["ar_entropy"] = 0.5
            if len(centres_list) >= 2:
                page_diag = math.sqrt(page_w * page_w + page_h * page_h) or 1.0
                max_dist = GRID_SPATIAL_CLUSTER_FRAC * page_diag
                close = sum(1 for i in range(len(centres_list)) for j in range(len(centres_list)) if i != j and math.hypot(centres_list[i][0] - centres_list[j][0], centres_list[i][1] - centres_list[j][1]) <= max_dist)
                n_pairs = len(centres_list) * (len(centres_list) - 1)
                feat["cell_clustering_coef"] = (close / n_pairs) if n_pairs > 0 else 0.5
            else:
                feat["cell_clustering_coef"] = 0.5
        else:
            feat["ar_entropy"] = 0.5
            feat["cell_clustering_coef"] = 0.5

        lines = []
        for line in inputs.cluster_lines or []:
            if line.poly_id == poly_id:
                lines.append(line.text_norm or line.text or "")
        if len(lines) >= 2:
            tokens_per_line = [_tokenize(t) for t in lines]
            all_tokens = [t for row in tokens_per_line for t in row]
            if all_tokens:
                from collections import Counter
                cnt = Counter(all_tokens)
                repeats = sum(1 for c in cnt.values() if c > 1)
                feat["token_repetition_across_rows"] = min(1.0, repeats / max(1, len(cnt)))
            else:
                feat["token_repetition_across_rows"] = 0.0
        else:
            feat["token_repetition_across_rows"] = 0.0

        if is_grid and gid and "grid_col_idx" in feat:
            col_idx = feat.get("grid_col_idx", -1)
            same_col_texts = []
            for p in grid_pids:
                f = cluster_features.get(p, {})
                if int(f.get("grid_col_idx", -1)) == col_idx:
                    same_col_texts.extend([(ln.text_norm or ln.text or "") for ln in (inputs.cluster_lines or []) if getattr(ln, "poly_id", None) == p])
            if len(same_col_texts) >= 2:
                try:
                    from collections import Counter
                    c1 = Counter(_tokenize(same_col_texts[0]))
                    c2 = Counter(_tokenize(same_col_texts[1]))
                    dot = sum(c1[t] * c2[t] for t in c1 if t in c2)
                    norm1 = math.sqrt(sum(c * c for c in c1.values()))
                    norm2 = math.sqrt(sum(c * c for c in c2.values()))
                    feat["vertical_text_similarity"] = (dot / (norm1 * norm2 + 1e-12)) if (norm1 and norm2) else 0.0
                except Exception:
                    feat["vertical_text_similarity"] = 0.0
            else:
                feat["vertical_text_similarity"] = 0.0
        else:
            feat["vertical_text_similarity"] = 0.0


NUMERIC_TOKEN_RE = re.compile(r"^[0-9][0-9.,/±°xX\-]*$")
ALPHA_TOKEN_RE = re.compile(r"^[A-Za-z]+$")
# Label-value ratio: fraction of lines in the cell that look like "Label: value" (colon or equals).
# This regex is strict: only lines with punctuation (:=) count. Many info/title-block cells have
# label and value in separate cells or no punctuation, so they get ratio 0. Do not rely on it alone.
LABEL_VALUE_RE = re.compile(r".+[:=]\s*.+")




def _cluster_center(rect: Rectangle) -> Tuple[float, float]:
    x1, y1, x2, y2 = rect.bbox
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def _bbox_iou(bbox_a: Tuple[float, float, float, float], bbox_b: Tuple[float, float, float, float]) -> float:
    """IoU of two bboxes in (x1, y1, x2, y2) form."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class ClusterContentFeatures:
    numeric_fraction: float = 0.0
    alpha_fraction: float = 0.0
    token_count_mean: float = 0.0
    token_count_cv: float = 0.0
    row_token_cv: float = 0.0
    header_body_numeric_delta: float = 0.0
    label_value_ratio: float = 0.0
    structural_position: float = 0.0
    frequency_score: float = 0.0
    semantic_neighbourhood: float = 0.0
    isolation_penalty: float = 0.0


def enrich_cluster_content_features(
    inputs: DataCleaningInputsV2,
    page_ctx: Optional[PageGeometryContext] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute content-driven and additional geometry features per cluster (poly_id).
    If page_ctx is provided, row-gap tolerance is scaled by thickness for this document.
    """
    cluster_lines_by_poly: Dict[str, List[ClusterLine]] = {}
    for line in inputs.cluster_lines or []:
        if not line.poly_id:
            continue
        cluster_lines_by_poly.setdefault(line.poly_id, []).append(line)

    rectangles_by_poly: Dict[str, Rectangle] = {r.poly_id: r for r in inputs.rectangles or [] if r.poly_id}

    page_h = float(inputs.page.height) if inputs.page and inputs.page.height else 1.0
    page_w = float(inputs.page.width) if inputs.page and getattr(inputs.page, "width", None) else 1.0
    # Scale for size buckets and row gap: adaptive to page size (aspect ratio and resolution)
    size_scale = max(10.0, min(page_w, page_h) * 0.025)
    row_gap_page_tol = max(2.0, page_h * 0.004)
    if page_ctx is not None:
        row_gap_page_tol *= page_ctx.thickness_scale_factor

    # Pre-compute centres for neighbourhood metrics
    centres: Dict[str, Tuple[float, float]] = {}
    for poly_id, rect in rectangles_by_poly.items():
        centres[poly_id] = _cluster_center(rect)

    # Size histogram for frequency_score (page-relative bucket size)
    size_buckets: Dict[str, int] = {}
    for poly_id, rect in rectangles_by_poly.items():
        x1, y1, x2, y2 = rect.bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        key = f"{int(w // size_scale)}x{int(h // size_scale)}"
        size_buckets[key] = size_buckets.get(key, 0) + 1

    enriched: Dict[str, Dict[str, Any]] = {}

    for poly_id, rect in rectangles_by_poly.items():
        lines = cluster_lines_by_poly.get(poly_id, [])
        all_tokens: List[str] = []
        numeric_tokens = 0
        alpha_tokens = 0
        label_like_lines = 0

        # simple y-based row clustering: sort by y-mid, then group with small gaps
        rows: List[List[ClusterLine]] = []
        sorted_lines = sorted(lines, key=lambda ln: (ln.bbox[1] + ln.bbox[3]) * 0.5)
        for ln in sorted_lines:
            y_mid = 0.5 * (ln.bbox[1] + ln.bbox[3])
            if not rows:
                rows.append([ln])
                continue
            last_row = rows[-1]
            last_y_mid = 0.5 * (last_row[-1].bbox[1] + last_row[-1].bbox[3])
            line_h = ln.bbox[3] - ln.bbox[1]
            if abs(y_mid - last_y_mid) <= (0.4 * line_h + row_gap_page_tol):
                last_row.append(ln)
            else:
                rows.append([ln])

        tokens_per_row: List[int] = []
        numeric_fraction_per_row: List[float] = []

        for row in rows:
            row_tokens: List[str] = []
            row_numeric = 0
            for ln in row:
                text = ln.text_norm or ln.text or ""
                toks = _tokenize(text)
                row_tokens.extend(toks)
                for t in toks:
                    if NUMERIC_TOKEN_RE.match(t):
                        row_numeric += 1
            if row_tokens:
                tokens_per_row.append(len(row_tokens))
                numeric_fraction_per_row.append(row_numeric / float(len(row_tokens)))

        token_counts: List[int] = []
        for ln in lines:
            text = ln.text_norm or ln.text or ""
            toks = _tokenize(text)
            token_counts.append(len(toks))
            all_tokens.extend(toks)
            for t in toks:
                if NUMERIC_TOKEN_RE.match(t):
                    numeric_tokens += 1
                elif ALPHA_TOKEN_RE.match(t):
                    alpha_tokens += 1
            if LABEL_VALUE_RE.match(text):
                label_like_lines += 1

        total_tokens = max(1, len(all_tokens))
        numeric_fraction = numeric_tokens / float(total_tokens)
        alpha_fraction = alpha_tokens / float(total_tokens)
        token_count_mean = _safe_mean([float(c) for c in token_counts])
        token_count_cv = _cv_from_values([float(c) for c in token_counts])
        row_token_cv = _cv_from_values([float(c) for c in tokens_per_row])

        header_body_numeric_delta = 0.0
        if numeric_fraction_per_row:
            header = numeric_fraction_per_row[0]
            body_vals = numeric_fraction_per_row[1:] or [header]
            header_body_numeric_delta = max(0.0, _safe_mean(body_vals) - header)

        label_value_ratio = 0.0
        if lines:
            label_value_ratio = label_like_lines / float(len(lines))

        x1, y1, x2, y2 = rect.bbox
        cy = 0.5 * (y1 + y2)
        structural_position = min(1.0, max(0.0, cy / page_h))

        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        size_key = f"{int(w // size_scale)}x{int(h // size_scale)}"
        freq = size_buckets.get(size_key, 1)
        frequency_score = math.log1p(freq)

        this_center = centres[poly_id]
        dists: List[float] = []
        for other_id, other_center in centres.items():
            if other_id == poly_id:
                continue
            d = _euclidean(this_center, other_center)
            dists.append(d)

        min_dist = min(dists) if dists else page_h
        isolation_penalty = 1.0 if min_dist > 0.25 * page_h else 0.0
        semantic_neighbourhood = (1.0 - min(1.0, min_dist / (0.5 * page_h))) * (
            1.0 - min(1.0, row_token_cv)
        )

        col_gap_cv = 0.0
        if len(lines) >= 2:
            sorted_by_x = sorted(lines, key=lambda ln: 0.5 * (ln.bbox[0] + ln.bbox[2]))
            gaps: List[float] = []
            for i in range(1, len(sorted_by_x)):
                left_right = sorted_by_x[i - 1].bbox[2]
                right_left = sorted_by_x[i].bbox[0]
                gaps.append(max(0.0, right_left - left_right))
            if gaps:
                col_gap_cv = _cv_from_values(gaps)

        token_row_regularity = row_token_cv

        feats = ClusterContentFeatures(
            numeric_fraction=numeric_fraction,
            alpha_fraction=alpha_fraction,
            token_count_mean=token_count_mean,
            token_count_cv=token_count_cv,
            row_token_cv=row_token_cv,
            header_body_numeric_delta=header_body_numeric_delta,
            label_value_ratio=label_value_ratio,
            structural_position=structural_position,
            frequency_score=frequency_score,
            semantic_neighbourhood=semantic_neighbourhood,
            isolation_penalty=isolation_penalty,
        )

        enriched[poly_id] = {
            "numeric_fraction": feats.numeric_fraction,
            "alpha_fraction": feats.alpha_fraction,
            "token_count_mean": feats.token_count_mean,
            "token_count_cv": feats.token_count_cv,
            "row_token_cv": feats.row_token_cv,
            "token_row_regularity": token_row_regularity,
            "col_gap_cv": col_gap_cv,
            "header_body_numeric_delta": feats.header_body_numeric_delta,
            "label_value_ratio": feats.label_value_ratio,
            "structural_position": feats.structural_position,
            "frequency_score": feats.frequency_score,
            "semantic_neighbourhood": feats.semantic_neighbourhood,
            "isolation_penalty": feats.isolation_penalty,
        }

    return enriched


class FormatClassifier:
    """Score-based structure classifier with guardrails and ABSTAIN"""

    # Label sets
    CLUSTER_LABELS = ["TABLE", "INFO_CLUSTER"]
    REGION_LABELS = ["KEY_VALUE", "LIST", "PARAGRAPH", "STRING"]

    # Tie-breaking priority (higher index = lower priority)
    CLUSTER_PRIORITY = {"TABLE": 0, "INFO_CLUSTER": 1, "ABSTAIN": 2}
    REGION_PRIORITY = {"KEY_VALUE": 0, "LIST": 1, "PARAGRAPH": 2, "STRING": 3, "ABSTAIN": 4}

    DEFAULT_CONFIG = {
        "T_min": 2.5,  # Raised for stricter classification
        "delta_min": 2.5,  # Raised for stricter classification
        "grid_consistency_threshold": 0.6,  # Lowered from 0.8
        "thresholds": {
            "row_gap_cv_low": 0.5,
            "row_gap_cv_high": 1.0,
            "density_low": 0.01,
            "density_medium": 0.05,
            "small_area_threshold": 0.005,  # relative to page area
        },
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = dict(self.DEFAULT_CONFIG)
        if config:
            self._merge_config(config)

    def _merge_config(self, overrides: Dict[str, Any]) -> None:
        for key, val in overrides.items():
            if isinstance(val, dict) and key in self.config and isinstance(self.config[key], dict):
                self.config[key].update(val)
            else:
                self.config[key] = val

    # Main classification entry point

    def classify(
        self, inputs: DataCleaningInputsV2
    ) -> Dict[str, Any]:
        """Classify all regions in the inputs."""
        cluster_results = self._classify_clusters(inputs)
        block_results, standalone_results = self._classify_regions(inputs)

        return {
            "clusters": cluster_results,
            "blocks": block_results,
            "standalone_lines": standalone_results,
        }

    # Cluster classification (per Rectangle/Contour) — inlined experimental (grid + survival)

    def _classify_clusters(
        self, inputs: DataCleaningInputsV2
    ) -> Dict[str, Dict[str, Any]]:
        # Inlined experimental cluster classifier (no external import)
        # 1) Page geometry once
        page_ctx = compute_page_geometry_context(inputs)
        page_w = float(inputs.page.width) if inputs.page and getattr(inputs.page, "width", None) else 1.0
        page_h = float(inputs.page.height) if inputs.page and inputs.page.height else 1.0

        # 1b) Regions: distinct envelopes (gap-based) so classification can be region-relative
        poly_to_region, region_bboxes = _compute_regions(inputs.rectangles or [], page_w, page_h)

        # 2) Grids early (geometry-only merge so grid_id/row/col exist before content)
        # max_dist is adaptive to aspect ratio and line thickness for different drawing styles
        cluster_features = dict(inputs.cluster_features or {})
        grid_content = _compute_grid_level_content(
            cluster_features,
            inputs.rectangles or [],
            page_w,
            page_h,
            geometry_only=True,
            page_ctx=page_ctx,
        )
        for poly_id, extras in grid_content.items():
            base = cluster_features.get(poly_id, {})
            base.update(extras)
            cluster_features[poly_id] = base

        # 3) Content enrichment (can use grid_id)
        content_feats = enrich_cluster_content_features(inputs, page_ctx=page_ctx)
        for poly_id, extras in content_feats.items():
            base = cluster_features.get(poly_id, {})
            merged = dict(base)
            merged.update(extras)
            cluster_features[poly_id] = merged

        # 4) Refresh grid-level content aggregates (grid_numeric, grid_header_delta, grid_label_value)
        _update_grid_content_aggregates(cluster_features)

        # 4b) Region-relative structural_position and per-region stats (classification only within region)
        region_poly_ids: Dict[str, List[str]] = defaultdict(list)
        if poly_to_region and region_bboxes:
            for rect in inputs.rectangles or []:
                if not rect.poly_id or not rect.bbox or len(rect.bbox) != 4:
                    continue
                pid = rect.poly_id
                rid = poly_to_region.get(pid)
                if rid not in region_bboxes:
                    continue
                feat = cluster_features.get(pid)
                if not feat:
                    continue
                feat["region_id"] = rid
                region_poly_ids[rid].append(pid)
                x1, y1, x2, y2 = float(rect.bbox[0]), float(rect.bbox[1]), float(rect.bbox[2]), float(rect.bbox[3])
                cy = 0.5 * (y1 + y2)
                rbox = region_bboxes[rid]
                rh = rbox[3] - rbox[1]
                feat["structural_position"] = min(1.0, max(0.0, (cy - rbox[1]) / rh)) if rh > 0 else 0.5
            region_stats: Dict[str, Dict[str, float]] = {}
            for rid in region_bboxes:
                pids = region_poly_ids.get(rid, [])
                region_stats[rid] = _compute_page_content_stats(cluster_features, poly_ids_subset=pids) if pids else _compute_page_content_stats(cluster_features)
        else:
            region_stats = {}

        # 5) New survival features: ar_entropy, cell_clustering_coef, font stub, token_repetition, vertical_text_sim
        _add_survival_new_features(cluster_features, inputs.rectangles or [], inputs, page_ctx)

        results: Dict[str, Dict[str, Any]] = {}

        page_area = 1.0
        if inputs.page and getattr(inputs.page, "width", None) and getattr(inputs.page, "height", None):
            page_area = float(inputs.page.width) * float(inputs.page.height)

        dyn = _dynamic_table_thresholds(page_ctx)
        table_numeric_min = dyn["table_numeric_min"]
        table_header_delta_min = dyn["table_header_delta_min"]
        label_heavy_bar = dyn["label_heavy_bar"]

        # Page-level content stats (fallback when no regions or grid has no region_id)
        page_stats = _compute_page_content_stats(cluster_features)
        page_title_block_numeric_max = max(0.52, min(0.62, page_stats["page_p75_numeric"] + 0.05))
        page_title_block_header_delta_max = max(table_header_delta_min + 0.08, min(0.38, 0.18 + page_stats["page_median_label_value"]))

        # Grid-first: one label per grid, then propagate to all cells (survival uses grid_label when margin not met)
        # Use region-level stats when grid belongs to a region so classification is resolved only within that region
        grid_labels: Dict[str, str] = {}
        for poly_id, f in cluster_features.items():
            if f.get("is_part_of_grid") and f.get("grid_id") is not None:
                gid = f["grid_id"]
                if gid not in grid_labels:
                    rid = f.get("region_id")
                    if rid and rid in region_stats:
                        rs = region_stats[rid]
                        title_block_numeric_max = max(0.52, min(0.62, rs["page_p75_numeric"] + 0.05))
                        title_block_header_delta_max = max(table_header_delta_min + 0.08, min(0.38, 0.18 + rs["page_median_label_value"]))
                    else:
                        title_block_numeric_max = page_title_block_numeric_max
                        title_block_header_delta_max = page_title_block_header_delta_max
                    grid_labels[gid] = _classify_grid(
                        f,
                        table_numeric_min,
                        table_header_delta_min,
                        label_heavy_bar,
                        title_block_numeric_max,
                        title_block_header_delta_max,
                    )

        # Regions that contain at least one grid classified as TABLE (for ungridded title-block cells)
        region_ids_with_table_grid: set = set()
        for pid, f in cluster_features.items():
            if f.get("is_part_of_grid") and f.get("grid_id") is not None:
                if grid_labels.get(f["grid_id"]) == "TABLE":
                    rid = f.get("region_id")
                    if rid:
                        region_ids_with_table_grid.add(rid)

        # Duplicate rectangles (same bbox, different poly_id): score once, reuse result for duplicates
        seen_bbox_to_poly: Dict[Tuple[float, float, float, float], str] = {}
        effective_title_block_alignment_min = _effective_title_block_alignment_min(page_ctx)

        for rect in inputs.rectangles or []:
            bbox = tuple(float(x) for x in rect.bbox) if rect.bbox and len(rect.bbox) == 4 else (0.0, 0.0, 0.0, 0.0)
            if bbox in seen_bbox_to_poly:
                first_poly_id = seen_bbox_to_poly[bbox]
                if first_poly_id in results:
                    r0 = results[first_poly_id]
                    results[rect.poly_id] = {
                        "label": r0["label"],
                        "scores": dict(r0["scores"]),
                        "reasons": {k: list(v) for k, v in r0["reasons"].items()},
                        "low_confidence": r0["low_confidence"],
                    }
                continue
            seen_bbox_to_poly[bbox] = rect.poly_id

            feat = cluster_features.get(rect.poly_id, {})

            # Page-relative area and cell-likeness for adaptive weights
            poly_area = float(feat.get("poly_area", 0) or 0)
            feat["area_ratio"] = (poly_area / page_area) if page_area > 0 else 0.0
            poly_w = float(feat.get("poly_width", 0) or 0)
            poly_h = float(feat.get("poly_height", 0) or 0)
            ar = float(feat.get("aspect_ratio", 1.0))

            # Per-cluster derived features from page context
            feat["cell_ar_normalized"] = (
                (ar / page_ctx.page_aspect_ratio) if page_ctx.page_aspect_ratio > 0 else ar
            )
            min_dim = min(poly_w, poly_h) if (poly_w > 0 and poly_h > 0) else 0.0
            feat["cell_to_line_ratio"] = (
                (min_dim / page_ctx.page_line_thickness_px)
                if page_ctx.page_line_thickness_px > 0 else 0.0
            )

            is_part_of_grid = bool(feat.get("is_part_of_grid", False))
            grid_id = feat.get("grid_id")
            grid_label = grid_labels.get(grid_id) if grid_id else None

            # Survival mode: hard rules first, then prototype distance + confidence margin (no weighted sum).
            label: str
            reasons = {"TABLE": [], "INFO_CLUSTER": []}
            survival_debug: Optional[Dict[str, Any]] = None

            if _has_query_presence(feat, inputs) and not is_part_of_grid:
                label = "INFO_CLUSTER"
                reasons["INFO_CLUSTER"].append("query presence + not in grid (hard rule)")
            elif _has_structural_marker(feat, page_ctx, page_h):
                label = "INFO_CLUSTER"
                reasons["INFO_CLUSTER"].append("structural marker + not in grid (hard rule)")
            elif _has_two_sets_of_text_in_cell(rect.poly_id, feat, inputs):
                label = "INFO_CLUSTER"
                reasons["INFO_CLUSTER"].append("hard rule: two sets of text in one cell (label+value)")
            elif (
                (not is_part_of_grid or feat.get("grid_cell_count", 0) == 0)
                and float(feat.get("structural_position", 0)) >= 0.7
                and feat.get("region_id") in region_ids_with_table_grid
            ):
                label = "TABLE"
                reasons["TABLE"].append("title-block same region as TABLE grid (ungridded cell)")
            elif (
                float(feat.get("numeric_fraction", 1.0)) <= PER_CELL_OVERRIDE_TITLE_BLOCK_NUMERIC_MAX
                and float(feat.get("structural_position", 0)) >= TITLE_BLOCK_POSITION_MIN
                and float(feat.get("label_value_ratio", 0.0)) > PER_CELL_OVERRIDE_TITLE_BLOCK_LABEL_MIN
            ):
                label = "INFO_CLUSTER"
                reasons["INFO_CLUSTER"].append("per-cell override: metadata-like in title block")
            elif (
                float(feat.get("structural_position", 0)) >= TITLE_BLOCK_POSITION_MIN
                and float(feat.get("numeric_fraction", 1.0)) <= TITLE_BLOCK_POSITION_CONTENT_NUMERIC_MAX
                and float(feat.get("grid_numeric_fraction", 1.0)) < TITLE_BLOCK_GRID_NUMERIC_MAX
                and (
                    float(feat.get("grid_regularity_mean", 1.0)) < TITLE_BLOCK_GRID_REGULARITY_TABLE_MIN
                    or float(feat.get("grid_column_alignment_score", 1.0)) < effective_title_block_alignment_min
                )
            ):
                label = "INFO_CLUSTER"
                reasons["INFO_CLUSTER"].append("per-cell override: title block + no numeric + low grid numeric + (low regularity or alignment)")
            else:
                label, survival_debug = _classify_survival(feat, page_ctx, grid_label)
                if label == "TABLE":
                    reasons["TABLE"].append("prototype distance + margin")
                elif label == "INFO_CLUSTER":
                    reasons["INFO_CLUSTER"].append("prototype distance + margin")
                else:
                    reasons["INFO_CLUSTER"].append("ABSTAIN: margin not met, no grid label")

            low_conf = label == "ABSTAIN"
            # Populate scores for overlay: 1 - distance (higher = closer to prototype) so "what is winning" is visible
            if survival_debug:
                d_t = survival_debug.get("d_table", 0.5)
                d_i = survival_debug.get("d_info", 0.5)
                scores = {
                    "TABLE": round(max(0.0, 1.0 - min(1.0, d_t)), 2),
                    "INFO_CLUSTER": round(max(0.0, 1.0 - min(1.0, d_i)), 2),
                }
            else:
                scores = {"TABLE": 0.0, "INFO_CLUSTER": 0.0}

            out_entry = {
                "label": label,
                "scores": scores,
                "reasons": reasons,
                "low_confidence": low_conf,
            }
            if survival_debug:
                out_entry["survival_debug"] = survival_debug
            results[rect.poly_id] = out_entry

        if cluster_features is not None:
            inputs.cluster_features = cluster_features
        return results


    # Region classification (per Region_Block / standalone Region_Line)

    def _classify_regions(
        self, inputs: DataCleaningInputsV2
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        block_results: Dict[str, Dict[str, Any]] = {}
        standalone_results: Dict[str, Dict[str, Any]] = {}

        block_features = inputs.region_block_features or {}
        line_features = inputs.region_line_features or {}

        # Track which lines are in blocks
        lines_in_blocks: Set[str] = set()
        for block in inputs.region_blocks or []:
            for lid in block.line_ids:
                lines_in_blocks.add(lid)

        # Classify blocks
        for block in inputs.region_blocks or []:
            feat = block_features.get(block.id, {})
            # Get line-level features for lines in this block
            block_line_feats = [
                line_features.get(lid, {}) for lid in block.line_ids
            ]
            scores, reasons = self._score_region_block(feat, block_line_feats, inputs)
            label, low_confidence = self._select_label(
                scores, self.REGION_LABELS, self.REGION_PRIORITY
            )
            block_results[block.id] = {
                "label": label,
                "scores": scores,
                "reasons": reasons,
                "low_confidence": low_confidence,
            }

        # Classify standalone lines (not in any block)
        for line in inputs.region_lines or []:
            if line.id in lines_in_blocks:
                continue
            feat = line_features.get(line.id, {})
            scores, reasons = self._score_standalone_line(feat, line, inputs)
            label, low_confidence = self._select_label(
                scores, self.REGION_LABELS, self.REGION_PRIORITY
            )
            standalone_results[line.id] = {
                "label": label,
                "scores": scores,
                "reasons": reasons,
                "low_confidence": low_confidence,
            }

        return block_results, standalone_results

    def _score_region_block(
        self,
        feat: Dict[str, Any],
        line_feats: List[Dict[str, Any]],
        inputs: DataCleaningInputsV2,
    ) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """Score a Region_Block for KEY_VALUE, LIST, PARAGRAPH, STRING."""
        scores = {"KEY_VALUE": 0.0, "LIST": 0.0, "PARAGRAPH": 0.0, "STRING": 0.0}
        reasons: Dict[str, List[str]] = {
            "KEY_VALUE": [], "LIST": [], "PARAGRAPH": [], "STRING": []
        }

        # Extract features
        n_lines = feat.get("n_lines", 0)
        mean_len = feat.get("mean_len", 0)
        num_frac = feat.get("num_frac", 0.0)
        bullet_frac = feat.get("bullet_frac", 0.0)
        bullet_line_count = feat.get("bullet_line_count", 0)
        colon_frac = feat.get("colon_frac", 0.0)
        aspect_ratio = feat.get("aspect_ratio", 1.0)
        width_cv = feat.get("width_cv", 0.0)
        line_len_cv = feat.get("line_len_cv", 0.0)
        gap_cv = feat.get("gap_cv", 0.0)
        width_rel_page = feat.get("width_rel_page", 0.0)
        wrap_ratio = feat.get("wrap_ratio", 0.0)
        indent_levels_count = feat.get("indent_levels_count", 0)
        indent_ladder = feat.get("indent_ladder", 0.0)
        token_count = feat.get("token_count", 0)
        density = feat.get("density", 0.0)
        delimiter_band_count = feat.get("delimiter_band_count", 0)

        # Count label candidates from line features
        label_candidate_count = sum(
            1 for lf in line_feats if lf.get("is_single_line_label_candidate")
        )

        # KEY_VALUE scoring

        if colon_frac >= 0.15 or delimiter_band_count >= 1:
            scores["KEY_VALUE"] += 2
            reasons["KEY_VALUE"].append("colon_frac >= 0.15 or delimiter_band_count >= 1")

        # Two-band pattern
        if indent_levels_count == 2:
            scores["KEY_VALUE"] += 2
            reasons["KEY_VALUE"].append("two-band indent pattern")

        if label_candidate_count >= 2:
            scores["KEY_VALUE"] += 1
            reasons["KEY_VALUE"].append("multiple label candidates")

        # Penalties
        if n_lines < 2 and delimiter_band_count == 0:
            scores["KEY_VALUE"] -= 2
            reasons["KEY_VALUE"].append("n_lines < 2, no delimiter pattern")

        if bullet_frac > 0:
            scores["KEY_VALUE"] -= 1
            reasons["KEY_VALUE"].append("bullet_frac > 0 (likely LIST)")

        # LIST scoring

        if bullet_frac >= 0.25:
            scores["LIST"] += 2
            reasons["LIST"].append("bullet_frac >= 0.25")

        if bullet_line_count >= 2:
            scores["LIST"] += 2
            reasons["LIST"].append("bullet_line_count >= 2")

        if indent_ladder > 0 and indent_levels_count in (1, 2):
            scores["LIST"] += 2
            reasons["LIST"].append("indent_ladder with 1-2 levels")

        if gap_cv < 0.4:
            scores["LIST"] += 1
            reasons["LIST"].append("gap_cv < 0.4 (consistent spacing)")

        if mean_len <= 80:
            scores["LIST"] += 1
            reasons["LIST"].append("mean_len <= 80")

        # Penalties
        if n_lines < 2:
            scores["LIST"] -= 2
            reasons["LIST"].append("n_lines < 2")

        if width_rel_page > 0.8 and mean_len > 80:
            scores["LIST"] -= 1
            reasons["LIST"].append("wide block with long lines (paragraph-like)")

        if colon_frac > 0.3 and bullet_frac < 0.1:
            scores["LIST"] -= 1
            reasons["LIST"].append("high colon_frac, low bullet_frac (KV-like)")

        # PARAGRAPH scoring

        # Paragraphs need multiple lines with significant text
        if n_lines >= 3:
            if mean_len > 80:
                scores["PARAGRAPH"] += 4
                reasons["PARAGRAPH"].append("n_lines >= 3 with long lines")
            elif mean_len > 40:
                scores["PARAGRAPH"] += 3
                reasons["PARAGRAPH"].append("n_lines >= 3 with moderate lines")
            else:
                scores["PARAGRAPH"] += 2
                reasons["PARAGRAPH"].append("n_lines >= 3")
        elif n_lines == 2:
            if mean_len > 100:  # Very long lines suggest paragraph
                scores["PARAGRAPH"] += 2
                reasons["PARAGRAPH"].append("2 long lines (wrapped paragraph)")
            elif mean_len > 60:
                scores["PARAGRAPH"] += 1
                reasons["PARAGRAPH"].append("2 moderate lines")

        # High token count indicates substantial text
        if token_count >= 20:
            scores["PARAGRAPH"] += 2
            reasons["PARAGRAPH"].append("token_count >= 20")
        elif token_count >= 10:
            scores["PARAGRAPH"] += 1
            reasons["PARAGRAPH"].append("token_count >= 10")

        # Lines fill the block width (wrapped text)
        if width_rel_page >= 0.4 and 0.6 <= wrap_ratio <= 1.4:
            scores["PARAGRAPH"] += 1
            reasons["PARAGRAPH"].append("lines fill block width")

        # No structural markers (bullets, colons)
        if bullet_frac < 0.1 and colon_frac < 0.1:
            scores["PARAGRAPH"] += 1
            reasons["PARAGRAPH"].append("no structural markers")

        # PARAGRAPH Penalties - FIX 5: Reduced penalties
        if n_lines < 2:
            scores["PARAGRAPH"] -= 3
            reasons["PARAGRAPH"].append("n_lines < 2 (too short)")
        elif n_lines < 3:
            if token_count >= 15:  # Has some content
                scores["PARAGRAPH"] -= 1
                reasons["PARAGRAPH"].append("n_lines < 3 but has content")
            else:
                scores["PARAGRAPH"] -= 2
                reasons["PARAGRAPH"].append("n_lines < 3 (short)")

        if token_count < 8:
            scores["PARAGRAPH"] -= 2
            reasons["PARAGRAPH"].append("token_count < 8 (very little content)")

        if bullet_frac >= 0.2:
            scores["PARAGRAPH"] -= 2
            reasons["PARAGRAPH"].append("bullet_frac >= 0.2 (likely LIST)")

        if delimiter_band_count >= 1 and colon_frac > 0.15:
            scores["PARAGRAPH"] -= 1
            reasons["PARAGRAPH"].append("delimiter pattern (KV-like)")

        if line_len_cv > 0.9:
            scores["PARAGRAPH"] -= 1
            reasons["PARAGRAPH"].append("line_len_cv very high")

        # STRING scoring

        # Single line is strong STRING indicator
        if n_lines == 1:
            scores["STRING"] += 4
            reasons["STRING"].append("n_lines == 1")
        elif n_lines == 2:
            if mean_len < 50:  # Short lines
                scores["STRING"] += 3
                reasons["STRING"].append("n_lines == 2 with short lines")
            else:
                scores["STRING"] += 1
                reasons["STRING"].append("n_lines == 2")
        elif n_lines <= 4:
            if mean_len < 30:  # Very short lines
                scores["STRING"] += 2
                reasons["STRING"].append("few lines with very short text")

        # Low token count
        if token_count < 8:
            scores["STRING"] += 3
            reasons["STRING"].append("token_count < 8 (minimal content)")
        elif token_count < 15:
            scores["STRING"] += 2
            reasons["STRING"].append("token_count < 15")
        elif token_count < 25:
            scores["STRING"] += 1
            reasons["STRING"].append("token_count < 25")

        # Narrow block
        if width_rel_page < 0.3:
            scores["STRING"] += 2
            reasons["STRING"].append("width_rel_page < 0.3 (narrow)")
        elif width_rel_page < 0.5:
            scores["STRING"] += 1
            reasons["STRING"].append("width_rel_page < 0.5")

        # Short lines
        if mean_len < 30:
            scores["STRING"] += 2
            reasons["STRING"].append("mean_len < 30 (very short lines)")
        elif mean_len < 50:
            scores["STRING"] += 1
            reasons["STRING"].append("mean_len < 50 (short lines)")

        # STRING Penalties - FIX 5: Reduced for multi-line short blocks
        if n_lines >= 5:
            scores["STRING"] -= 3
            reasons["STRING"].append("n_lines >= 5 (too many for STRING)")
        elif n_lines >= 3:
            if mean_len > 40:  # Only penalize if lines are not short
                scores["STRING"] -= 1
                reasons["STRING"].append("n_lines >= 3 with longer lines")

        if token_count >= 30:
            scores["STRING"] -= 2
            reasons["STRING"].append("token_count >= 30 (substantial content)")

        if mean_len > 80:
            scores["STRING"] -= 2
            reasons["STRING"].append("mean_len > 80 (long lines)")

        if bullet_frac > 0.15 or colon_frac > 0.15:
            scores["STRING"] -= 1
            reasons["STRING"].append("structural patterns present")

        return scores, reasons

    def _score_standalone_line(
        self,
        feat: Dict[str, Any],
        line: RegionLine,
        inputs: DataCleaningInputsV2,
    ) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """Score a standalone Region_Line for KEY_VALUE or STRING."""
        scores = {"KEY_VALUE": 0.0, "LIST": 0.0, "PARAGRAPH": 0.0, "STRING": 0.0}
        reasons: Dict[str, List[str]] = {
            "KEY_VALUE": [], "LIST": [], "PARAGRAPH": [], "STRING": []
        }

        is_label_candidate = feat.get("is_single_line_label_candidate", False)
        is_value_candidate = feat.get("is_single_line_value_candidate", False)
        delimiter_pos = feat.get("delimiter_position_norm")
        num_frac = feat.get("num_frac", 0.0)
        is_id_like = feat.get("is_id_like", False)
        is_numeric_like = feat.get("is_numeric_like", False)

        # Compute width relative to page
        width_rel_page = 0.0
        if inputs.page and inputs.page.width:
            line_width = line.width or (line.bbox[2] - line.bbox[0])
            width_rel_page = line_width / inputs.page.width

        text = line.text_norm or line.text_raw or ""
        has_delimiter = bool(DELIMITER_PATTERN.search(text))

        # KEY_VALUE scoring (standalone line with "label: value" structure)

        if has_delimiter and delimiter_pos is not None and 0.1 < delimiter_pos < 0.6:
            scores["KEY_VALUE"] += 2
            reasons["KEY_VALUE"].append("delimiter in label position")

        if is_label_candidate and is_value_candidate:
            scores["KEY_VALUE"] += 2
            reasons["KEY_VALUE"].append("both label and value parts detected")

        if has_delimiter and len(line.tokens) >= 2:
            scores["KEY_VALUE"] += 1
            reasons["KEY_VALUE"].append("delimiter with multiple tokens")

        # STRING scoring (default for standalone lines)

        scores["STRING"] += 3  # Base bonus for single line
        reasons["STRING"].append("standalone line")

        if width_rel_page < 0.4:
            scores["STRING"] += 1
            reasons["STRING"].append("narrow line")

        if is_id_like or is_numeric_like:
            scores["STRING"] += 1
            reasons["STRING"].append("id-like or numeric-like")

        if not has_delimiter:
            scores["STRING"] += 1
            reasons["STRING"].append("no delimiter")

        # Penalties for STRING
        if has_delimiter and delimiter_pos and 0.1 < delimiter_pos < 0.6:
            scores["STRING"] -= 1
            reasons["STRING"].append("has KV-like delimiter")

        # LIST/PARAGRAPH get minimal scores for standalone lines
        if _line_has_bullet(line):
            scores["LIST"] += 1
            reasons["LIST"].append("has bullet pattern")

        return scores, reasons

    # Label selection with ABSTAIN logic

    def _select_label(
        self,
        scores: Dict[str, float],
        labels: List[str],
        priority: Dict[str, int],
    ) -> Tuple[str, bool]:
        """Select best label or ABSTAIN based on T_min and delta_min thresholds."""
        T_min = self.config["T_min"]
        delta_min = self.config["delta_min"]

        # Sort by score descending, then by priority ascending for ties
        sorted_labels = sorted(
            labels,
            key=lambda lbl: (-scores.get(lbl, 0.0), priority.get(lbl, 99)),
        )

        best_label = sorted_labels[0]
        best_score = scores.get(best_label, 0.0)
        second_score = scores.get(sorted_labels[1], 0.0) if len(sorted_labels) > 1 else 0.0

        # ABSTAIN conditions
        if best_score < T_min or (best_score - second_score) < delta_min:
            return "ABSTAIN", True

        return best_label, False


# Convenience function for classification

def classify_document_structure(inputs: DataCleaningInputsV2) -> Dict[str, Any]:
    """Classify document structure using FormatClassifier.
    
    Returns classification results for clusters, blocks, and standalone lines.
    """
    classifier = FormatClassifier()
    return classifier.classify(inputs)

# TABLE CONSOLIDATOR

@dataclass
class TableCell:
    """A single cell within a consolidated table."""
    cell_id: str
    bbox: BBox
    text: str
    poly_id: str  # Original polygon ID for traceability


@dataclass
class ConsolidatedTable:
    """A unified table object containing all its cells."""
    table_id: str
    cells: List[TableCell]
    bbox: BBox  # Bounding box of the entire table
    grid_rows: int
    grid_cols: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to output dictionary format."""
        return {
            "type": "table",
            "table_id": self.table_id,
            "bbox": list(self.bbox),
            "grid_shape": [self.grid_rows, self.grid_cols],
            "cells": [
                {
                    "cell_id": cell.cell_id,
                    "bbox": list(cell.bbox),
                    "text": cell.text,
                    "poly_id": cell.poly_id,
                }
                for cell in self.cells
            ]
        }


class TableConsolidator:
    """Consolidates TABLE-labeled cells into unified table objects."""
    
    def __init__(self, inputs: DataCleaningInputsV2, classification_results: Dict[str, Any]):
        self.inputs = inputs
        self.classification_results = classification_results
        self._table_counter = 0
    
    def consolidate(self) -> Dict[str, Any]:
        """Main entry point: consolidate all TABLE cells into table objects."""
        # Step 1: Extract TABLE-labeled cells
        table_cells = self._extract_table_cells()
        
        # Step 2: Group cells by grid membership
        cell_groups = self._group_cells_by_grid(table_cells)
        
        # Step 3: Validate and create consolidated tables
        consolidated_tables: List[ConsolidatedTable] = []
        
        for group in cell_groups:
            validated_tables = self._validate_and_split_group(group)
            consolidated_tables.extend(validated_tables)
        
        # Step 4: Merge small adjacent tables with same column count
        consolidated_tables = self._merge_adjacent_tables(consolidated_tables)
        
        # Step 5: Filter out tiny tables
        MIN_TABLE_CELLS = 4
        valid_tables = []
        rejected_poly_ids = set()
        for table in consolidated_tables:
            if len(table.cells) >= MIN_TABLE_CELLS:
                valid_tables.append(table)
            else:
                # Track rejected cells
                for cell in table.cells:
                    rejected_poly_ids.add(cell.poly_id)
        consolidated_tables = valid_tables
        
        # Step 6: Find isolated TABLE cells
        isolated_cells = self._find_isolated_cells(table_cells, consolidated_tables)
        for cell_info in isolated_cells:
            rejected_poly_ids.add(cell_info["poly_id"])
        
        # Step 7: Collect non-TABLE clusters for pass-through. TABLE cells not in any
        # consolidated table (rejected small tables + isolated cells) fall back to INFO_CLUSTER.
        non_table_clusters = self._collect_non_table_clusters(additional_poly_ids=rejected_poly_ids)
        
        return {
            "tables": [t.to_dict() for t in consolidated_tables],
            "non_table_clusters": non_table_clusters,
        }
    
    def _extract_table_cells(self) -> List[Dict[str, Any]]:
        """Extract all clusters labeled as TABLE."""
        table_cells = []
        cluster_results = self.classification_results.get("clusters", {})
        
        for poly_id, result in cluster_results.items():
            if result.get("label") != "TABLE":
                continue
            
            # Get geometry features
            features = self.inputs.cluster_features.get(poly_id, {})
            
            # Get text content from ClusterLines
            text = self._get_cell_text(poly_id)
            
            # Get bounding box
            bbox = self._get_cell_bbox(poly_id)
            
            table_cells.append({
                "poly_id": poly_id,
                "bbox": bbox,
                "text": text,
                "grid_rows": features.get("grid_rows", 0),
                "grid_cols": features.get("grid_cols", 0),
                "grid_row_idx": features.get("grid_row_idx", -1),
                "grid_col_idx": features.get("grid_col_idx", -1),
                "is_part_of_grid": features.get("is_part_of_grid", False),
                "grid_cell_count": features.get("grid_cell_count", 0),
            })
        
        return table_cells
    
    def _get_cell_text(self, poly_id: str) -> str:
        """Get concatenated text from all ClusterLines in this cell."""
        lines = []
        for cl in self.inputs.cluster_lines:
            if cl.poly_id == poly_id:
                lines.append(cl.text_raw)
        return " ".join(lines).strip()
    
    def _get_cell_bbox(self, poly_id: str) -> BBox:
        """Get bounding box for a polygon ID."""
        # Check rectangles first
        for rect in self.inputs.rectangles:
            if rect.poly_id == poly_id:
                return rect.bbox
        
        # Check contours
        for contour in self.inputs.contours:
            if contour.poly_id == poly_id:
                return contour.bbox
        
        # Fallback: compute from cluster features
        features = self.inputs.cluster_features.get(poly_id, {})
        pts = features.get("poly_points", [])
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return (min(xs), min(ys), max(xs), max(ys))
        
        return (0.0, 0.0, 0.0, 0.0)
    
    def _group_cells_by_grid(self, table_cells: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group cells that share the same grid structure."""
        if not table_cells:
            return []
        
        # Group by (grid_rows, grid_cols) signature first
        signature_groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        
        for cell in table_cells:
            if cell["is_part_of_grid"] and cell["grid_rows"] > 0 and cell["grid_cols"] > 0:
                sig = (cell["grid_rows"], cell["grid_cols"])
                signature_groups[sig].append(cell)
        
        # For each signature group, further split by spatial connectivity
        final_groups = []
        
        for sig, cells in signature_groups.items():
            spatial_groups = self._split_by_spatial_connectivity(cells)
            final_groups.extend(spatial_groups)
        
        return final_groups
    
    def _split_by_spatial_connectivity(self, cells: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Split cells into spatially connected components."""
        if len(cells) <= 1:
            return [cells] if cells else []
        
        # Build adjacency based on spatial proximity
        n = len(cells)
        adjacency = {i: set() for i in range(n)}
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._cells_are_adjacent(cells[i], cells[j]):
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        
        # Find connected components using BFS
        visited = set()
        components = []
        
        for start in range(n):
            if start in visited:
                continue
            component = []
            queue = [start]
            while queue:
                idx = queue.pop(0)
                if idx in visited:
                    continue
                visited.add(idx)
                component.append(cells[idx])
                for neighbor in adjacency[idx]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            if component:
                components.append(component)
        
        return components
    
    def _cells_are_adjacent(self, cell1: Dict[str, Any], cell2: Dict[str, Any]) -> bool:
        """Check if two cells are spatially adjacent AND structurally compatible."""
        b1 = cell1["bbox"]
        b2 = cell2["bbox"]
        
        w1 = b1[2] - b1[0]
        h1 = b1[3] - b1[1]
        w2 = b2[2] - b2[0]
        h2 = b2[3] - b2[1]
        
        ref_size = min(h1, h2)
        max_gap = ref_size * 1.0  # Tighter gap tolerance
        
        # Check horizontal adjacency (same row, side by side)
        y_overlap_amount = min(b1[3], b2[3]) - max(b1[1], b2[1])
        y_overlap = y_overlap_amount > ref_size * 0.5  # Significant Y overlap
        x_gap = min(abs(b1[2] - b2[0]), abs(b2[2] - b1[0]))
        x_adjacent = x_gap < max_gap
        
        # Check vertical adjacency (same column, stacked)
        x_overlap_amount = min(b1[2], b2[2]) - max(b1[0], b2[0])
        x_overlap = x_overlap_amount > min(w1, w2) * 0.5  # Significant X overlap
        y_gap = min(abs(b1[3] - b2[1]), abs(b2[3] - b1[1]))
        y_adjacent = y_gap < max_gap
        
        is_horizontal_adjacent = y_overlap and x_adjacent
        is_vertical_adjacent = x_overlap and y_adjacent
        
        if not (is_horizontal_adjacent or is_vertical_adjacent):
            return False
        
        # Additional check: cells in the same table should have compatible dimensions
        if is_horizontal_adjacent:
            # Same row - check height compatibility
            height_ratio = min(h1, h2) / max(h1, h2) if max(h1, h2) > 0 else 0
            if height_ratio < 0.5:  # Heights differ by more than 2x
                return False
            
            actual_x_gap = min(abs(b1[2] - b2[0]), abs(b2[2] - b1[0]))
            if actual_x_gap >= 5:
                # Cells have a gap - might be at table boundary
                # Apply stricter width check
                width_ratio = min(w1, w2) / max(w1, w2) if max(w1, w2) > 0 else 0
                if width_ratio < 0.5:
                    return False
        
        if is_vertical_adjacent:
            # Same column - check width compatibility
            width_ratio = min(w1, w2) / max(w1, w2) if max(w1, w2) > 0 else 0
            if width_ratio < 0.5:  # Widths differ by more than 2x
                return False
        
        return True
    
    def _validate_and_split_group(self, group: List[Dict[str, Any]]) -> List[ConsolidatedTable]:
        """Validate a group forms a perfect rectangular grid."""
        if not group:
            return []
        
        # Compute row and column clusters from actual positions
        row_clusters, col_clusters = self._cluster_cells_into_grid(group)
        
        n_rows = len(row_clusters)
        n_cols = len(col_clusters)
        
        # Check if it's a perfect rectangle
        if len(group) == n_rows * n_cols:
            grid_info = {"rows": n_rows, "cols": n_cols}
            table = self._create_table_from_group(group, grid_info)
            return [table]
        
        # Not a perfect rectangle - try to split into sub-grids
        # Group cells by their (row, col) position
        cell_grid = {} 
        for cell in group:
            cx = (cell["bbox"][0] + cell["bbox"][2]) / 2
            cy = (cell["bbox"][1] + cell["bbox"][3]) / 2
            col_idx = self._find_cluster_index(cx, col_clusters)
            row_idx = self._find_cluster_index(cy, row_clusters)
            cell_grid[(row_idx, col_idx)] = cell
        
        col_indices = sorted(set(c for r, c in cell_grid.keys()))
        row_indices = sorted(set(r for r, c in cell_grid.keys()))
        
        # Find ALL possible rectangles and pick the largest ones
        all_rectangles = []
        
        for start_row_i, start_row in enumerate(row_indices):
            for start_col_i, start_col in enumerate(col_indices):
                if (start_row, start_col) not in cell_grid:
                    continue
                
                # Try all possible end positions
                for end_row_i in range(start_row_i, len(row_indices)):
                    end_row = row_indices[end_row_i]
                    for end_col_i in range(start_col_i, len(col_indices)):
                        end_col = col_indices[end_col_i]
                        
                        # Check if this rectangle is complete
                        is_complete = True
                        rect_cells = []
                        for ri in range(start_row_i, end_row_i + 1):
                            for ci in range(start_col_i, end_col_i + 1):
                                r, c = row_indices[ri], col_indices[ci]
                                if (r, c) in cell_grid:
                                    rect_cells.append(cell_grid[(r, c)])
                                else:
                                    is_complete = False
                                    break
                            if not is_complete:
                                break
                        
                        if is_complete and rect_cells:
                            n_rows = end_row_i - start_row_i + 1
                            n_cols = end_col_i - start_col_i + 1
                            all_rectangles.append({
                                "cells": rect_cells,
                                "rows": n_rows,
                                "cols": n_cols,
                                "size": len(rect_cells),
                                "positions": {(row_indices[ri], col_indices[ci]) 
                                             for ri in range(start_row_i, end_row_i + 1)
                                             for ci in range(start_col_i, end_col_i + 1)}
                            })
        
        # Sort by size (largest first)
        all_rectangles.sort(key=lambda x: -x["size"])
        
        sub_tables = []
        used_positions = set()
        
        for rect in all_rectangles:
            # Check if this rectangle overlaps with already used positions
            if rect["positions"] & used_positions:
                continue
            
            # Use this rectangle
            used_positions |= rect["positions"]
            grid_info = {"rows": rect["rows"], "cols": rect["cols"]}
            table = self._create_table_from_group(rect["cells"], grid_info)
            sub_tables.append(table)
        
        return sub_tables
    
    def _cluster_cells_into_grid(self, group: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
        """Cluster cells into rows and columns based on their positions."""
        # Compute median cell height and width for tolerance
        heights = [cell["bbox"][3] - cell["bbox"][1] for cell in group]
        widths = [cell["bbox"][2] - cell["bbox"][0] for cell in group]
        median_height = sorted(heights)[len(heights) // 2] if heights else 10
        median_width = sorted(widths)[len(widths) // 2] if widths else 10
        
        # First cluster columns
        x_positions = [(cell["bbox"][0] + cell["bbox"][2]) / 2 for cell in group]
        col_clusters = self._cluster_positions_1d(x_positions, tolerance=median_width * 0.4)
        
        # Cluster rows
        y_positions = [(cell["bbox"][1] + cell["bbox"][3]) / 2 for cell in group]
        row_clusters = self._cluster_positions_1d(y_positions, tolerance=median_height * 0.4)
        
        return sorted(row_clusters), sorted(col_clusters)
    
    def _cluster_positions_1d(self, positions: List[float], tolerance: float = 10) -> List[float]:
        """Cluster 1D positions into groups."""
        if not positions:
            return []
        
        sorted_pos = sorted(positions)
        clusters = [[sorted_pos[0]]]
        
        for pos in sorted_pos[1:]:
            if pos - clusters[-1][-1] < tolerance:
                clusters[-1].append(pos)
            else:
                clusters.append([pos])
        
        # Return cluster centers
        return [sum(c) / len(c) for c in clusters]
    
    def _find_cluster_index(self, value: float, clusters: List[float]) -> int:
        """Find which cluster a value belongs to."""
        min_dist = float('inf')
        best_idx = 0
        for i, center in enumerate(clusters):
            dist = abs(value - center)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        return best_idx
    
    def _check_perfect_rectangle(self, group: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        """Check if cells form a complete rectangular grid."""
        if not group:
            return False, {}
        
        # Get expected grid dimensions from cell features
        expected_rows = group[0].get("grid_rows", 0)
        expected_cols = group[0].get("grid_cols", 0)
        expected_count = expected_rows * expected_cols
        
        # Check 1: Do we have the expected number of cells?
        actual_count = len(group)
        if expected_count > 0 and actual_count != expected_count:
            # Not a complete grid - some cells missing
            return False, {"rows": expected_rows, "cols": expected_cols, "reason": "incomplete"}
        
        # Check 2: Verify row/column indices cover the full range
        row_indices = set()
        col_indices = set()
        for cell in group:
            row_idx = cell.get("grid_row_idx", -1)
            col_idx = cell.get("grid_col_idx", -1)
            if row_idx >= 0:
                row_indices.add(row_idx)
            if col_idx >= 0:
                col_indices.add(col_idx)
        
        # Check for gaps in indices
        if row_indices and col_indices:
            expected_row_set = set(range(len(row_indices)))
            expected_col_set = set(range(len(col_indices)))
            
            # Normalize indices to 0-based
            min_row = min(row_indices)
            min_col = min(col_indices)
            normalized_rows = {r - min_row for r in row_indices}
            normalized_cols = {c - min_col for c in col_indices}
            
            rows_complete = normalized_rows == set(range(len(row_indices)))
            cols_complete = normalized_cols == set(range(len(col_indices)))
            
            if not (rows_complete and cols_complete):
                return False, {"rows": len(row_indices), "cols": len(col_indices), "reason": "gaps"}
        
        # Check 3: Verify each (row, col) position has exactly one cell
        positions = set()
        for cell in group:
            row_idx = cell.get("grid_row_idx", -1)
            col_idx = cell.get("grid_col_idx", -1)
            if row_idx >= 0 and col_idx >= 0:
                pos = (row_idx, col_idx)
                if pos in positions:
                    return False, {"reason": "duplicate_position"}
                positions.add(pos)
        
        # Compute actual grid dimensions
        actual_rows = len(row_indices) if row_indices else self._infer_rows(group)
        actual_cols = len(col_indices) if col_indices else self._infer_cols(group)
        
        # Final check: cell count matches grid dimensions
        if actual_rows * actual_cols != actual_count:
            return False, {"rows": actual_rows, "cols": actual_cols, "reason": "dimension_mismatch"}
        
        return True, {"rows": actual_rows, "cols": actual_cols}
    
    def _infer_rows(self, group: List[Dict[str, Any]]) -> int:
        """Infer number of rows from cell positions."""
        y_positions = [(c["bbox"][1] + c["bbox"][3]) / 2 for c in group]
        if not y_positions:
            return 0
        
        # Cluster Y positions
        y_positions.sort()
        tolerance = (max(y_positions) - min(y_positions)) / (len(group) + 1) * 0.5
        if tolerance < 5:
            tolerance = 5
        
        rows = 1
        last_y = y_positions[0]
        for y in y_positions[1:]:
            if y - last_y > tolerance:
                rows += 1
                last_y = y
        
        return rows
    
    def _infer_cols(self, group: List[Dict[str, Any]]) -> int:
        """Infer number of columns from cell positions."""
        x_positions = [(c["bbox"][0] + c["bbox"][2]) / 2 for c in group]
        if not x_positions:
            return 0
        
        # Cluster X positions
        x_positions.sort()
        tolerance = (max(x_positions) - min(x_positions)) / (len(group) + 1) * 0.5
        if tolerance < 5:
            tolerance = 5
        
        cols = 1
        last_x = x_positions[0]
        for x in x_positions[1:]:
            if x - last_x > tolerance:
                cols += 1
                last_x = x
        
        return cols
    
    def _compute_grid_info(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute grid info from cell positions."""
        rows = self._infer_rows(group)
        cols = self._infer_cols(group)
        return {"rows": rows, "cols": cols}
    
    def _merge_adjacent_tables(self, tables: List[ConsolidatedTable]) -> List[ConsolidatedTable]:
        """Merge small adjacent tables that have the same column count."""
        if len(tables) <= 1:
            return tables
        
        # Only consider merging small tables (< 10 cells)
        small_tables = [t for t in tables if len(t.cells) < 10]
        large_tables = [t for t in tables if len(t.cells) >= 10]
        
        if len(small_tables) <= 1:
            return tables
        
        # Try to merge small tables with same column count that are vertically adjacent
        merged = []
        used = set()
        
        for i, t1 in enumerate(small_tables):
            if i in used:
                continue
            
            # Find tables that can be merged with t1
            merge_candidates = [t1]
            
            for j, t2 in enumerate(small_tables):
                if j <= i or j in used:
                    continue
                
                # Check if same column count
                if t1.grid_cols != t2.grid_cols:
                    continue
                
                # Check if vertically adjacent (X ranges overlap, Y ranges are close)
                x_overlap = min(t1.bbox[2], t2.bbox[2]) - max(t1.bbox[0], t2.bbox[0])
                if x_overlap < min(t1.bbox[2] - t1.bbox[0], t2.bbox[2] - t2.bbox[0]) * 0.5:
                    continue
                
                # Check Y gap
                y_gap = max(t1.bbox[1], t2.bbox[1]) - min(t1.bbox[3], t2.bbox[3])
                if y_gap > 20:  # Max 20 pixel gap
                    continue
                
                # Can merge
                merge_candidates.append(t2)
                used.add(j)
            
            if len(merge_candidates) > 1:
                # Merge all candidates into one table
                used.add(i)
                merged_table = self._merge_tables(merge_candidates)
                merged.append(merged_table)
            else:
                merged.append(t1)
        
        return large_tables + merged
    
    def _merge_tables(self, tables: List[ConsolidatedTable]) -> ConsolidatedTable:
        """Merge multiple tables into one."""
        self._table_counter += 1
        table_id = f"T{self._table_counter}"
        
        # Combine all cells
        all_cells = []
        for t in tables:
            all_cells.extend(t.cells)
        
        # Renumber cells
        for idx, cell in enumerate(all_cells):
            cell.cell_id = f"cell_{idx}"
        
        # Compute combined bbox
        all_x1 = [t.bbox[0] for t in tables]
        all_y1 = [t.bbox[1] for t in tables]
        all_x2 = [t.bbox[2] for t in tables]
        all_y2 = [t.bbox[3] for t in tables]
        combined_bbox = (min(all_x1), min(all_y1), max(all_x2), max(all_y2))
        
        # Compute new grid dimensions
        total_rows = sum(t.grid_rows for t in tables)
        grid_cols = tables[0].grid_cols  # All have same column count
        
        return ConsolidatedTable(
            table_id=table_id,
            cells=all_cells,
            bbox=combined_bbox,
            grid_rows=total_rows,
            grid_cols=grid_cols,
        )
    
    def _create_table_from_group(self, group: List[Dict[str, Any]], grid_info: Dict[str, Any]) -> ConsolidatedTable:
        """Create a ConsolidatedTable from a validated group."""
        self._table_counter += 1
        table_id = f"T{self._table_counter}"
        
        # Create cells with unique IDs
        cells = []
        for idx, cell_info in enumerate(group):
            cell = TableCell(
                cell_id=f"cell_{idx}",
                bbox=cell_info["bbox"],
                text=cell_info["text"],
                poly_id=cell_info["poly_id"],
            )
            cells.append(cell)
        
        # Compute table bounding box
        all_x1 = [c["bbox"][0] for c in group]
        all_y1 = [c["bbox"][1] for c in group]
        all_x2 = [c["bbox"][2] for c in group]
        all_y2 = [c["bbox"][3] for c in group]
        table_bbox = (min(all_x1), min(all_y1), max(all_x2), max(all_y2))
        
        return ConsolidatedTable(
            table_id=table_id,
            cells=cells,
            bbox=table_bbox,
            grid_rows=grid_info.get("rows", 0),
            grid_cols=grid_info.get("cols", 0),
        )
    
    def _find_isolated_cells(
        self, 
        all_table_cells: List[Dict[str, Any]], 
        consolidated_tables: List[ConsolidatedTable]
    ) -> List[Dict[str, Any]]:
        """Find TABLE cells that weren't included in any consolidated table."""
        # Collect all poly_ids already in tables
        used_poly_ids = set()
        for table in consolidated_tables:
            for cell in table.cells:
                used_poly_ids.add(cell.poly_id)
        
        # Find cells not yet used
        isolated = []
        for cell in all_table_cells:
            if cell["poly_id"] not in used_poly_ids:
                isolated.append(cell)
        
        return isolated
    
    def _create_single_cell_table(self, cell_info: Dict[str, Any]) -> ConsolidatedTable:
        """Create a 1x1 table from an isolated cell."""
        self._table_counter += 1
        table_id = f"T{self._table_counter}"
        
        cell = TableCell(
            cell_id="cell_0",
            bbox=cell_info["bbox"],
            text=cell_info["text"],
            poly_id=cell_info["poly_id"],
        )
        
        return ConsolidatedTable(
            table_id=table_id,
            cells=[cell],
            bbox=cell_info["bbox"],
            grid_rows=1,
            grid_cols=1,
        )
    
    def _collect_non_table_clusters(self, additional_poly_ids: set = None) -> List[Dict[str, Any]]:
        """Collect all non-TABLE clusters for pass-through."""
        non_table = []
        cluster_results = self.classification_results.get("clusters", {})
        additional_poly_ids = additional_poly_ids or set()
        
        for poly_id, result in cluster_results.items():
            label = result.get("label")
            
            # Include non-TABLE clusters
            if label != "TABLE":
                non_table.append({
                    "poly_id": poly_id,
                    "label": label,
                    "bbox": list(self._get_cell_bbox(poly_id)),
                    "text": self._get_cell_text(poly_id),
                })
            # Also include rejected TABLE cells as INFO_CLUSTER
            elif poly_id in additional_poly_ids:
                non_table.append({
                    "poly_id": poly_id,
                    "label": "INFO_CLUSTER",  # Reclassify as INFO_CLUSTER
                    "bbox": list(self._get_cell_bbox(poly_id)),
                    "text": self._get_cell_text(poly_id),
                })
        
        return non_table

# Convenience function for table consolidation

def consolidate_tables(
    inputs: DataCleaningInputsV2, 
    classification_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Consolidate TABLE-labeled cells into unified table objects."""
    consolidator = TableConsolidator(inputs, classification_results)
    return consolidator.consolidate()

# QUERY SET & CANDIDATE ANCHOR COLLECTOR

# Default path for query set YAML
DEFAULT_QUERY_SET_PATH = Path(__file__).parent.parent / "config" / "query_set.yaml"

# Alias token pattern for normalization
ALIAS_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass
class QueryFieldDefinition:
    """Definition of a canonical query field with aliases and expected types."""
    name: str
    aliases: List[str]
    types: List[str]
    notes: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    alias_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    alias_tokens: Set[str] = field(default_factory=set)


@dataclass
class CandidateAnchor:
    """A candidate anchor representing a potential match for a query within a block."""
    query_name: str
    anchor_text: str
    text_norm: str
    score: float
    alias_matched: str
    source_type: str  # "table_cell", "cluster", "list", "paragraph", "key_value", "string"
    source_id: str    # e.g., "T1/cell_5" or poly_id
    block_id: str
    bbox: BBox
    tokens: Set[str] = field(default_factory=set)
    fuzz_score: float = 0.0
    cosine_score: float = 0.0
    gate_reason: str = ""  # Debug: "token_match", "short_string", "fuzz_gate"
    token_count: int = 0


class QuerySetManager:
    """Loads query set definitions and provides hybrid matching utilities."""
    
    def __init__(
        self,
        query_path: Optional[Path] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        alias_boost: float = 0.10,
        semantic_weight: float = 0.7,
        fuzz_weight: float = 0.3,
    ) -> None:
        """Initialize the QuerySetManager."""
        self.query_path = Path(query_path or DEFAULT_QUERY_SET_PATH)
        if not self.query_path.exists():
            raise FileNotFoundError(f"Query set file not found: {self.query_path}")
        
        with self.query_path.open("r", encoding="utf-8") as f:
            self.raw_config = yaml.safe_load(f)
        
        self.alias_boost = alias_boost
        self.semantic_weight = semantic_weight
        self.fuzz_weight = fuzz_weight
        self.normalizer = TextNormalizer()
        
        # Parse guardrails and fields
        self.guardrails = self.raw_config.get("globals", {}).get("guardrails", [])
        self.fields: List[QueryFieldDefinition] = self._build_fields(
            self.raw_config.get("fields", [])
        )
        
        # Build global alias token set for fast pre-filtering
        self.global_alias_tokens: Set[str] = set()
        for fld in self.fields:
            self.global_alias_tokens |= fld.alias_tokens
        
        # Resolve embedding model: use local path if present (offline), else HF name
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _local_sentence_model = os.path.join(_project_root, "models", "all-MiniLM-L6-v2")
        if os.path.isdir(_local_sentence_model):
            model_name = _local_sentence_model
        # Load embedding model and pre-compute embeddings
        self.embedder = SentenceTransformer(model_name)
        self._precompute_embeddings()
    
    def _build_fields(self, field_entries: List[Dict]) -> List[QueryFieldDefinition]:
        """Parse field definitions from YAML."""
        fields = []
        for entry in field_entries:
            field_types = entry.get("type", [])
            if isinstance(field_types, str):
                field_types = [field_types]
            aliases = entry.get("aliases", []) or []
            
            qfield = QueryFieldDefinition(
                name=entry["name"],
                aliases=aliases,
                types=field_types,
                notes=entry.get("notes"),
            )
            # Collect normalized tokens from name and all aliases
            qfield.alias_tokens = self._collect_alias_tokens([qfield.name] + qfield.aliases)
            fields.append(qfield)
        return fields
    
    def _precompute_embeddings(self) -> None:
        """Pre-embed all query names and aliases."""
        phrases: List[str] = []
        owners: List[Tuple[str, QueryFieldDefinition, Optional[str]]] = []
        
        for fld in self.fields:
            phrases.append(fld.name)
            owners.append(("name", fld, None))
            for alias in fld.aliases:
                phrases.append(alias)
                owners.append(("alias", fld, alias))
        
        if not phrases:
            return
        
        # Batch encode all phrases
        embeds = self.embedder.encode(
            phrases, convert_to_numpy=True, normalize_embeddings=True
        )
        
        for vec, (kind, fld, alias) in zip(embeds, owners):
            if kind == "name":
                fld.embedding = vec
            else:
                fld.alias_embeddings[alias] = vec
    
    def _collect_alias_tokens(self, phrases: List[str]) -> Set[str]:
        """Extract normalized tokens from phrases."""
        tokens: Set[str] = set()
        for phrase in phrases:
            tokens |= self._tokenize(phrase)
        return tokens
    
    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into lowercase alphanumeric tokens."""
        return {match.group(0).lower() for match in ALIAS_TOKEN_PATTERN.finditer(text)}
    
    def compute_hybrid_score(
        self,
        candidate_text: str,
        candidate_embedding: np.ndarray,
        target_text: str,
        target_embedding: Optional[np.ndarray],
    ) -> Tuple[float, float, float]:
        """Compute hybrid similarity score."""
        if target_embedding is None:
            return 0.0, 0.0, 0.0
        
        cosine_sim = float(np.dot(candidate_embedding, target_embedding))
        fuzz_score = fuzz.token_set_ratio(candidate_text.lower(), target_text.lower()) / 100.0
        hybrid = self.semantic_weight * cosine_sim + self.fuzz_weight * fuzz_score
        
        return max(0.0, min(1.0, hybrid)), cosine_sim, fuzz_score
    
    def score_candidate_for_field(
        self,
        field: QueryFieldDefinition,
        candidate_text: str,
        candidate_embedding: np.ndarray,
        candidate_tokens: Set[str],
    ) -> Tuple[float, str, float, float]:
        """Score a candidate against a specific query field."""
        # Check for token overlap (exact alias hit)
        has_token_overlap = bool(field.alias_tokens & candidate_tokens)
        
        # Score against field name
        best_score, best_cos, best_fuzz = self.compute_hybrid_score(
            candidate_text, candidate_embedding, field.name, field.embedding
        )
        alias_matched = field.name
        
        # Score against each alias
        for alias, alias_vec in field.alias_embeddings.items():
            score, cos, fz = self.compute_hybrid_score(
                candidate_text, candidate_embedding, alias, alias_vec
            )
            if score > best_score:
                best_score = score
                best_cos = cos
                best_fuzz = fz
                alias_matched = alias
        
        # Apply alias boost for exact token matches
        if has_token_overlap:
            best_score = min(1.0, best_score + self.alias_boost)
        
        return best_score, alias_matched, best_cos, best_fuzz
    
    def get_field_by_name(self, name: str) -> Optional[QueryFieldDefinition]:
        """Get a field definition by name."""
        for fld in self.fields:
            if fld.name == name:
                return fld
        return None


class CandidateAnchorCollector:
    """Collects candidate anchors from structural blocks and scores them against queries."""
    
    # Safety caps to prevent downstream explosion
    GLOBAL_MAX_K = 25  
    MAX_ANCHORS_PER_TABLE = 25  
    MAX_SURVIVORS_PER_BLOCK = 200  
    SHORT_STRING_TOKEN_LIMIT = 4  
    SHORT_STRING_FUZZ_CLAMP = 0.95  
    
    def __init__(
        self,
        query_manager: QuerySetManager,
        inputs: DataCleaningInputsV2,
        consolidation_results: Dict[str, Any],
        classification_results: Dict[str, Any],
        default_top_k: int = 5,
        fuzz_gate: float = 0.5,
        min_score: float = 0.85,
    ) -> None:
        """Initialize the collector."""
        self.query_manager = query_manager
        self.inputs = inputs
        self.consolidation_results = consolidation_results
        self.classification_results = classification_results
        self.default_top_k = min(default_top_k, self.GLOBAL_MAX_K)
        self.fuzz_gate = fuzz_gate
        self.min_score = min_score
        
        # Build block registry
        self._blocks = self._build_block_registry()
    
    def _build_block_registry(self) -> List[Dict[str, Any]]:
        """Build a registry of all structural blocks."""
        blocks = []
        
        # Add consolidated tables as blocks
        for table in self.consolidation_results.get("tables", []):
            blocks.append({
                "block_id": table["table_id"],
                "block_type": "TABLE",
                "data": table,
            })
        
        # Add non-table clusters as blocks
        for cluster in self.consolidation_results.get("non_table_clusters", []):
            blocks.append({
                "block_id": cluster["poly_id"],
                "block_type": cluster["label"],
                "data": cluster,
            })
        
        return blocks
    
    def collect_all(self) -> Dict[str, Any]:
        """Collect candidate anchors for all blocks and all queries."""
        all_anchors: List[CandidateAnchor] = []
        by_block: Dict[str, List[Dict]] = defaultdict(list)
        by_query: Dict[str, List[Dict]] = defaultdict(list)
        
        for block in self._blocks:
            block_anchors = self._collect_for_block(block)
            for anchor in block_anchors:
                anchor_dict = self._anchor_to_dict(anchor)
                all_anchors.append(anchor)
                by_block[anchor.block_id].append(anchor_dict)
                by_query[anchor.query_name].append(anchor_dict)
        
        return {
            "anchors": [self._anchor_to_dict(a) for a in all_anchors],
            "by_block": dict(by_block),
            "by_query": dict(by_query),
        }
    
    def _anchor_to_dict(self, anchor: CandidateAnchor) -> Dict[str, Any]:
        """Convert CandidateAnchor to dictionary."""
        return {
            "query_name": anchor.query_name,
            "anchor_text": anchor.anchor_text,
            "text_norm": anchor.text_norm,
            "score": anchor.score,
            "alias_matched": anchor.alias_matched,
            "source_type": anchor.source_type,
            "source_id": anchor.source_id,
            "block_id": anchor.block_id,
            "bbox": list(anchor.bbox),
            "fuzz_score": anchor.fuzz_score,
            "cosine_score": anchor.cosine_score,
            "gate_reason": anchor.gate_reason,
            "token_count": anchor.token_count,
        }
    
    def _collect_for_block(self, block: Dict[str, Any]) -> List[CandidateAnchor]:
        """Collect candidate anchors for a single block."""
        block_id = block["block_id"]
        block_type = block["block_type"]
        data = block["data"]
        
        # Extract candidate text spans from the block
        candidates = self._extract_candidates(block_id, block_type, data)
        
        if not candidates:
            return []
        
        # Embed all candidates ONCE per block (in-memory only)
        candidate_embeddings = self._embed_candidates_for_block(candidates)
        
        # Score candidates against all queries
        anchors: List[CandidateAnchor] = []
        
        for field in self.query_manager.fields:
            field_anchors = self._score_candidates_for_field(
                field, candidates, candidate_embeddings
            )
            # Keep top-k per field per block, with deterministic tie-breaking
            field_anchors.sort(key=lambda a: (-a.score, -a.fuzz_score, a.anchor_text))
            anchors.extend(field_anchors[:self.default_top_k])
        
        # Deduplicate: keep only the highest-scoring anchor per text_norm
        best_by_text: Dict[str, CandidateAnchor] = {}
        for anchor in anchors:
            key = anchor.text_norm
            if key not in best_by_text or anchor.score > best_by_text[key].score:
                best_by_text[key] = anchor
        anchors = list(best_by_text.values())
        
        # Apply global per-table cap to prevent explosion
        if block_type == "TABLE" and len(anchors) > self.MAX_ANCHORS_PER_TABLE:
            anchors.sort(key=lambda a: (-a.score, -a.fuzz_score, a.anchor_text))
            anchors = anchors[:self.MAX_ANCHORS_PER_TABLE]
        
        return anchors
    
    def _embed_candidates_for_block(
        self, candidates: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """Embed all unique candidate texts for a block."""
        # Get unique texts to embed
        unique_texts = list(set(c["text_norm"] for c in candidates if c["text_norm"]))
        
        if not unique_texts:
            return {}
        
        # Apply survivor cap before embedding
        survivor_cap = min(self.MAX_SURVIVORS_PER_BLOCK, 2 * len(candidates))
        if len(unique_texts) > survivor_cap:
            # Keep texts that appear most frequently or are shortest (likely labels)
            text_counts = defaultdict(int)
            for c in candidates:
                text_counts[c["text_norm"]] += 1
            unique_texts.sort(key=lambda t: (-text_counts[t], len(t)))
            unique_texts = unique_texts[:survivor_cap]
        
        # Batch embed
        try:
            embeddings = self.query_manager.embedder.encode(
                unique_texts, convert_to_numpy=True, normalize_embeddings=True
            )
            return {text: emb for text, emb in zip(unique_texts, embeddings)}
        except Exception:
            # Graceful degradation: return empty dict, scoring will use fuzz-only
            return {}
    
    def _extract_candidates(
        self, block_id: str, block_type: str, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract candidate text spans from a block."""
        candidates = []
        
        if block_type == "TABLE":
            # Extract from table cells
            for cell in data.get("cells", []):
                text_raw = cell.get("text", "").strip()
                if not text_raw:
                    continue
                
                norm = self.query_manager.normalizer.normalize(text_raw)
                tokens = self.query_manager._tokenize(norm["text_norm"])
                
                candidates.append({
                    "text_raw": text_raw,
                    "text_norm": norm["text_norm"],
                    "tokens": tokens,
                    "token_count": len(tokens),
                    "source_type": "table_cell",
                    "source_id": f"{block_id}/{cell.get('cell_id', '')}",
                    "bbox": tuple(cell.get("bbox", (0, 0, 0, 0))),
                })
        else:
            # Extract from cluster/block
            poly_id = data.get("poly_id", block_id)
            bbox = tuple(data.get("bbox", (0, 0, 0, 0)))
            
            # Get text from cluster_lines
            texts = self._get_cluster_texts(poly_id)
            
            for text_raw in texts:
                if not text_raw.strip():
                    continue
                
                norm = self.query_manager.normalizer.normalize(text_raw)
                tokens = self.query_manager._tokenize(norm["text_norm"])
                
                # Determine source type based on block label
                source_type = self._map_block_type_to_source(block_type)
                
                candidates.append({
                    "text_raw": text_raw,
                    "text_norm": norm["text_norm"],
                    "tokens": tokens,
                    "token_count": len(tokens),
                    "source_type": source_type,
                    "source_id": poly_id,
                    "bbox": bbox,
                })
                
                # Also extract key-value candidates if colon present
                kv_candidates = self._extract_key_value_spans(text_raw)
                for kv_text in kv_candidates:
                    kv_norm = self.query_manager.normalizer.normalize(kv_text)
                    kv_tokens = self.query_manager._tokenize(kv_norm["text_norm"])
                    candidates.append({
                        "text_raw": kv_text,
                        "text_norm": kv_norm["text_norm"],
                        "tokens": kv_tokens,
                        "token_count": len(kv_tokens),
                        "source_type": "key_value",
                        "source_id": poly_id,
                        "bbox": bbox,
                    })
        
        return candidates
    
    def _get_cluster_texts(self, poly_id: str) -> List[str]:
        """Get all text lines associated with a polygon ID."""
        texts = []
        for cl in self.inputs.cluster_lines:
            if cl.poly_id == poly_id:
                texts.append(cl.text_raw)
        return texts
    
    def _map_block_type_to_source(self, block_type: str) -> str:
        """Map block type label to source type."""
        mapping = {
            "INFO_CLUSTER": "cluster",
            "LIST": "list",
            "PARAGRAPH": "paragraph",
            "KEY_VALUE": "key_value",
            "STRING": "string",
            "ABSTAIN": "cluster",
        }
        return mapping.get(block_type, "cluster")
    
    def _extract_key_value_spans(self, text: str) -> List[str]:
        """Extract key portions from key-value patterns."""
        candidates = []
        if ':' in text:
            parts = text.split(':')
            if len(parts) >= 2:
                left = parts[0].strip()
                if 1 <= len(left) <= 80:
                    candidates.append(left)
            if text.rstrip().endswith(':'):
                label = text.rstrip(':').strip()
                if label:
                    candidates.append(label)
        return candidates
    
    def _score_candidates_for_field(
        self,
        field: QueryFieldDefinition,
        candidates: List[Dict[str, Any]],
        candidate_embeddings: Dict[str, np.ndarray],
    ) -> List[CandidateAnchor]:
        """Score all candidates against a specific query field."""
        # Pre-filter candidates using token gating and fuzz soft-gate
        prefiltered = []
        for cand in candidates:
            token_count = cand.get("token_count", len(cand["tokens"]))
            
            # Token gating (cheap O(1) check)
            has_token_match = bool(field.alias_tokens & cand["tokens"])
            is_short_string = token_count <= self.SHORT_STRING_TOKEN_LIMIT
            
            if has_token_match:
                gate_reason = "token_match"
            elif is_short_string:
                gate_reason = "short_string"
            else:
                # RapidFuzz soft-gate for longer strings without token match
                max_fuzz = 0.0
                for alias in [field.name] + field.aliases:
                    fz = fuzz.token_set_ratio(cand["text_norm"].lower(), alias.lower()) / 100.0
                    max_fuzz = max(max_fuzz, fz)
                
                if max_fuzz >= self.fuzz_gate:
                    gate_reason = "fuzz_gate"
                else:
                    continue  # Rejected by gating
            
            prefiltered.append({**cand, "gate_reason": gate_reason})
        
        if not prefiltered:
            return []
        
        # Score each candidate using pre-computed embeddings
        anchors = []
        for cand in prefiltered:
            text_norm = cand["text_norm"]
            embed_vec = candidate_embeddings.get(text_norm)
            token_count = cand.get("token_count", len(cand["tokens"]))
            
            # Compute hybrid score
            if embed_vec is not None:
                score, alias_matched, cos_score, fuzz_score = self.query_manager.score_candidate_for_field(
                    field, text_norm, embed_vec, cand["tokens"]
                )
                
                # Clamp fuzz contribution for short strings to prevent false spikes
                if token_count <= self.SHORT_STRING_TOKEN_LIMIT:
                    clamped_fuzz = min(fuzz_score, self.SHORT_STRING_FUZZ_CLAMP)
                    # Recompute hybrid with clamped fuzz
                    score = (
                        self.query_manager.semantic_weight * cos_score +
                        self.query_manager.fuzz_weight * clamped_fuzz
                    )
                    # Re-apply alias boost if applicable
                    if field.alias_tokens & cand["tokens"]:
                        score = min(1.0, score + self.query_manager.alias_boost)
            else:
                # Fallback: fuzz-only scoring (graceful degradation)
                max_fuzz = 0.0
                best_alias = field.name
                for alias in [field.name] + field.aliases:
                    fz = fuzz.token_set_ratio(text_norm.lower(), alias.lower()) / 100.0
                    if fz > max_fuzz:
                        max_fuzz = fz
                        best_alias = alias
                
                # Clamp for short strings
                if token_count <= self.SHORT_STRING_TOKEN_LIMIT:
                    max_fuzz = min(max_fuzz, self.SHORT_STRING_FUZZ_CLAMP)
                
                score = max_fuzz
                alias_matched = best_alias
                cos_score = 0.0
                fuzz_score = max_fuzz
                
                # Apply alias boost
                if field.alias_tokens & cand["tokens"]:
                    score = min(1.0, score + self.query_manager.alias_boost)
            
            # Apply score threshold filter
            if score < self.min_score:
                continue
            
            anchors.append(CandidateAnchor(
                query_name=field.name,
                anchor_text=cand["text_raw"],
                text_norm=text_norm,
                score=score,
                alias_matched=alias_matched,
                source_type=cand["source_type"],
                source_id=cand["source_id"],
                block_id=cand["source_id"].split("/")[0] if "/" in cand["source_id"] else cand["source_id"],
                bbox=cand["bbox"],
                tokens=cand["tokens"],
                fuzz_score=fuzz_score,
                cosine_score=cos_score,
                gate_reason=cand["gate_reason"],
                token_count=token_count,
            ))
        
        return anchors

# Convenience function for candidate anchor collection

def collect_candidate_anchors(
    inputs: DataCleaningInputsV2,
    consolidation_results: Dict[str, Any],
    classification_results: Dict[str, Any],
    query_path: Optional[Path] = None,
    top_k: int = 5,
    min_score: float = 0.85,
) -> Dict[str, Any]:
    """Collect candidate anchors from all structural blocks."""
    query_manager = QuerySetManager(query_path=query_path)
    collector = CandidateAnchorCollector(
        query_manager=query_manager,
        inputs=inputs,
        consolidation_results=consolidation_results,
        classification_results=classification_results,
        default_top_k=top_k,
        min_score=min_score,
    )
    return collector.collect_all()
