import os
import sys

# CRITICAL: Set model paths before any torch/doctr/paddle/huggingface imports anywhere in the process
# Use project-relative models/ folder (same layout on dev and offline server)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_MODELS_ROOT = os.path.join(_PROJECT_ROOT, "models")
_DOCTR_ROOT = os.path.join(_MODELS_ROOT, "doctr")
os.environ["TORCH_HOME"] = _DOCTR_ROOT
os.environ["DOCTR_CACHE_DIR"] = _DOCTR_ROOT  # DocTR downloads to DOCTR_CACHE_DIR, not TORCH_HOME
os.environ["HF_HOME"] = _MODELS_ROOT  # Hugging Face cache (sentence-transformers, etc.)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ["PADDLEOCR_DOWNLOAD_DISABLE"] = "1"

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Add subdirectories to path
_base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_base_dir, 'input'))
sys.path.insert(0, os.path.join(_base_dir, 'workers'))
sys.path.insert(0, os.path.join(_base_dir, 'cleaners'))
sys.path.insert(0, os.path.join(_base_dir, 'bootstrap'))

# Local imports
from process_input import (
    BatchData,
    DocumentData,
    PageData,
    process_input_local,
    preprocess_image,
    get_batch_summary,
    iterate_pages,
)

# Worker imports
from frame_detection import detect_frame
from rectangle_detection import detect_rectangles
from cluster_ocr import process_rectangles
from mask import mask_rectangles_in_memory
from region_ocr import data_extraction_in_memory, prewarm_region_doctr

# Cleaner imports
from data_cleaningv2 import (
    build_data_cleaning_inputs_v2,
    classify_document_structure,
    consolidate_tables,
    collect_candidate_anchors,
    PageMetadata,
    DataCleaningInputsV2,
)
from relationship_resolver_v2_enhanced import resolve_relationships_v2, assemble_table_records
from cross_page_stitcher import CrossPageStitcher
from internal_val import InternalValidator

# Queue management
from queue_manager import (
    PageQueue,
    DocumentQueue,
    BatchQueue,
    build_batch_queue_from_batches,
    get_queue_summary,
)

# Bootstrap integration (Module A + Module B)
try:
    from bootstrap_adapter import (
        BOOTSTRAP_AVAILABLE,
        run_module_a,
        run_module_b,
        ModuleAResult,
        ModuleBResult,
    )
except ImportError:
    BOOTSTRAP_AVAILABLE = False
    run_module_a = None
    run_module_b = None
    ModuleAResult = None
    ModuleBResult = None


# =============================================================================
# PAGE-LEVEL EXTRACTION RESULTS (IN-MEMORY)
# =============================================================================

# CONSTANTS

A4_AREA = 2480 * 3508  # A4 at 300 DPI

@dataclass
class PageExtractionResult:
    """In-memory container for all extraction results from a single page."""
    page_id: str
    document_id: str
    batch_id: str
    
    # Frame Detection (only for >= A4 pages)
    frame_rect: Optional[tuple] = None  
    frame_mask: Optional[Any] = None    
    
    # Rectangle Detection
    rect_data: Optional[Dict[str, Any]] = None  
    rect_vis: Optional[Any] = None              
    
    # Cluster OCR
    cluster_results: Optional[Dict[str, Any]] = None  
    
    # Masking
    masked_image: Optional[Any] = None   
    mask_info: Optional[Dict[str, Any]] = None  
    
    # Region OCR
    region_results: Optional[Dict[str, Any]] = None
    
    # Data Cleaning (all scoped to this page via page_id)
    cleaning_inputs: Optional[Any] = None  
    classification: Optional[Dict[str, Any]] = None  
    consolidation: Optional[Dict[str, Any]] = None   
    anchors: Optional[Dict[str, Any]] = None         
    
    # Stage 6: Relationship Resolution
    relationships: Optional[Dict[str, Any]] = None
    
    # Bootstrap (Module A + Module B) results
    bootstrap_result: Optional[Any] = None  
    corrected_tokens: Optional[List[Any]] = None  
    proto_entities: Optional[List[Any]] = None  
    
    # Status tracking
    status: str = "PENDING"
    errors: List[str] = field(default_factory=list)


@dataclass
class DocumentExtractionResult:
    """In-memory container for all extraction results from a document."""
    document_id: str
    batch_id: str
    page_results: Dict[str, PageExtractionResult] = field(default_factory=dict)
    aggregated_results: Optional[Dict[str, Any]] = None
    status: str = "PENDING"


@dataclass
class BatchExtractionResult:
    """In-memory container for all extraction results from a batch."""
    batch_id: str
    document_results: Dict[str, DocumentExtractionResult] = field(default_factory=dict)
    validation_results: Optional[Dict[str, Any]] = None
    status: str = "PENDING"

# PIPELINE STAGE PROCESSORS

def process_page_extraction(page: PageData) -> PageExtractionResult:
    """Run extraction stages (rect detection, cluster OCR, region OCR) on a single page.
    All results stay in memory."""
    result = PageExtractionResult(
        page_id=page.page_id,
        document_id=page.document_id,
        batch_id=page.batch_id,
        status="EXTRACTING"
    )
    
    try:
        img = page.image
        h, w = img.shape[:2]
        page_area = w * h
        
        # Stage 1: Frame Detection (only if page >= A4 size)

        if page_area >= A4_AREA:
            logging.debug(f"Page {page.page_id}: Running frame detection (area={page_area})")
            frame_result, frame_vis = detect_frame(img, page_id=page.page_id, debug=False, basename=None)
            result.frame_rect = frame_result  # dict with 'bbox' and 'page_id', or None
            result.frame_mask = frame_vis  # Keep visualization in memory
            logging.info(f"Page {page.page_id}: Frame detected = {frame_result}")
        else:
            logging.debug(f"Page {page.page_id}: Skipping frame detection (area={page_area} < A4)")
            result.frame_rect = None
            result.frame_mask = None
        

        # Stage 2: Rectangle Detection
        # - If frame was detected: pass frame_rect to crop ROI
        # - If frame detection returned skip signal: process full image (no frame crop)
        # - If no frame detection ran: process full page image

        logging.debug(f"Page {page.page_id}: Running rectangle detection")
        # Extract bbox tuple from frame_rect dict if present (skip signal means no bbox)
        frame_bbox = result.frame_rect['bbox'] if result.frame_rect and 'bbox' in result.frame_rect else None
        rect_data, rect_vis = detect_rectangles(
            image=img,
            frame_rect=frame_bbox,  # tuple (x,y,w,h) or None
            page_id=page.page_id,
            adaptive_thresh=True,
            debug=False,
            basename=None
        )
        result.rect_data = rect_data
        result.rect_vis = rect_vis
        logging.info(f"Page {page.page_id}: Detected {len(rect_data.get('rectangles', []))} rectangles, {len(rect_data.get('clusters', []))} clusters")
        

        # Stage 3: Cluster OCR
        # - Receives rectangles from rect_data (each has 'crop' of the rectangle)
        # - Performs PaddleOCR on each rectangle crop (models: C:\OCULI\models\paddleocr)
        # - Outputs: clusters, Cluster_Line, Cluster_Block, Rectangles, Contours

        rectangles = result.rect_data.get('rectangles', []) if result.rect_data else []
        if rectangles:
            logging.debug(f"Page {page.page_id}: Running cluster OCR on {len(rectangles)} rectangles")
            cluster_results = process_rectangles(rectangles, page_id=page.page_id)
            result.cluster_results = cluster_results
            logging.info(f"Page {page.page_id}: Cluster OCR produced {len(cluster_results.get('Cluster_Line', []))} lines, {len(cluster_results.get('Cluster_Block', []))} blocks")
        else:
            logging.debug(f"Page {page.page_id}: No rectangles to process for cluster OCR")
            result.cluster_results = {
                'clusters': [],
                'Cluster_Line': [],
                'Cluster_Block': [],
                'Rectangles': [],
                'Contours': []
            }
        

        # Stage 4: Masking
        # - Receives page image, rect_data, and frame_rect
        # - Masks detected rectangles with white to isolate region text
        # - Outputs masked image for region OCR

        logging.debug(f"Page {page.page_id}: Running masking")
        mask_result = mask_rectangles_in_memory(
            image=img,
            rect_data=result.rect_data,
            frame_rect=result.frame_rect,
            page_id=page.page_id,
            debug=False
        )
        result.masked_image = mask_result['masked_image']
        result.mask_info = mask_result['mask_info']
        logging.info(f"Page {page.page_id}: Masked {mask_result['mask_info']['rectangles_masked']} rectangles")
        

        # Stage 5: Region OCR
        # - Receives masked image from masking stage
        # - Performs DocTR detection + PaddleOCR recognition (same engine as cluster_ocr)
        # - Outputs: Region_Block, Region_Line with page_id

        logging.debug(f"Page {page.page_id}: Running region OCR")
        region_results = data_extraction_in_memory(
            masked_image=result.masked_image,
            page_id=page.page_id,
            batch_id=page.batch_id,
            document_id=page.document_id
        )
        result.region_results = region_results
        logging.info(f"Page {page.page_id}: Region OCR produced {len(region_results.get('Region_Line', []))} lines, {len(region_results.get('Region_Block', []))} blocks")
        

        # Stage 5.5: Module A (NRM) - OCR Correction
        # - Converts OCR outputs to OcrToken format
        # - Processes through Module A (correction, canonicalization, entity hints)
        # - Corrected tokens stored for use in Module B (after data cleaning)
        # Pipeline: Region OCR → **Module A** → Data Cleaning

        if BOOTSTRAP_AVAILABLE and run_module_a:
            logging.debug(f"Page {page.page_id}: Running Module A (NRM)")
            module_a_result = run_module_a(
                cluster_results=result.cluster_results or {},
                region_results=result.region_results or {},
                page_id=page.page_id,
                document_id=page.document_id,
                enable_logging=False,
            )
            result.corrected_tokens = module_a_result.corrected_tokens
            
            logging.info(
                f"Page {page.page_id}: Module A complete - "
                f"{module_a_result.stats.get('ocr_tokens_count', 0)} tokens, "
                f"{module_a_result.stats.get('corrections_made', 0)} corrections, "
                f"{module_a_result.stats.get('entity_hints_detected', 0)} entity hints"
            )
        else:
            logging.debug(f"Page {page.page_id}: Bootstrap not available, skipping Module A")
        
        result.status = "EXTRACTED"
        
    except Exception as e:
        result.status = "ERROR"
        result.errors.append(str(e))
        logging.error(f"Extraction failed for page {page.page_id}: {e}")
    
    return result


def process_page_data_cleaning(
    page: PageData,
    extraction_result: PageExtractionResult
) -> PageExtractionResult:
    """Run data cleaning stages on a single page's extraction results."""
    try:
        extraction_result.status = "CLEANING"
        
        logging.info(f"Page {page.page_id}: Starting data cleaning (isolated)")
        

        # Step 1: Build PageMetadata for this page

        page_meta = PageMetadata(
            page_id=page.page_id,
            width=float(page.width),
            height=float(page.height),
            batch_id=page.batch_id,
            document_id=page.document_id,
        )
        
        # Step 2: Build DataCleaningInputsV2 from THIS PAGE's extraction data
        # - rect_data: rectangles detected on THIS page
        # - cluster_results: cluster OCR from THIS page's rectangles
        # - region_results: region OCR from THIS page's masked image
        # All structures already carry page_id from extraction stage

        logging.debug(f"Page {page.page_id}: Building cleaning inputs")
        cleaning_inputs = build_data_cleaning_inputs_v2(
            rect_data=extraction_result.rect_data,
            cluster_results=extraction_result.cluster_results,
            region_results=extraction_result.region_results,
            page_meta=page_meta,
        )
        extraction_result.cleaning_inputs = cleaning_inputs
        
        logging.info(
            f"Page {page.page_id}: Cleaning inputs built - "
            f"{len(cleaning_inputs.region_lines)} region lines, "
            f"{len(cleaning_inputs.cluster_lines)} cluster lines, "
            f"{len(cleaning_inputs.rectangles)} rectangles"
        )
        
        # Step 3: Classify document structures (per page)
        # - Classifies clusters as TABLE, INFO_CLUSTER, etc.
        # - Classifies region blocks as LIST, PARAGRAPH, KEY_VALUE, STRING

        logging.debug(f"Page {page.page_id}: Classifying structures")
        classification = classify_document_structure(cleaning_inputs)
        extraction_result.classification = classification
        
        logging.info(
            f"Page {page.page_id}: Classification complete - "
            f"{len(classification.get('cluster_labels', {}))} clusters, "
            f"{len(classification.get('block_labels', {}))} blocks"
        )

        # Step 4: Consolidate tables (per page)
        # - Groups TABLE-labeled cells into unified table objects
        # - Separates non-table clusters

        logging.debug(f"Page {page.page_id}: Consolidating tables")
        consolidation = consolidate_tables(cleaning_inputs, classification)
        extraction_result.consolidation = consolidation
        
        logging.info(
            f"Page {page.page_id}: Consolidation complete - "
            f"{len(consolidation.get('tables', []))} tables, "
            f"{len(consolidation.get('non_table_clusters', []))} non-table clusters"
        )

        # Step 5: Collect candidate anchors (per page)
        # - Matches query fields against text in tables, clusters, blocks
        # - All anchors are scoped to this page via page_id

        logging.debug(f"Page {page.page_id}: Collecting candidate anchors")
        anchors = collect_candidate_anchors(
            inputs=cleaning_inputs,
            consolidation_results=consolidation,
            classification_results=classification,
        )
        extraction_result.anchors = anchors
        
        logging.info(
            f"Page {page.page_id}: Anchor collection complete - "
            f"{len(anchors.get('anchors', []))} total anchors"
        )
        
        extraction_result.status = "CLEANED"
        logging.info(f"Page {page.page_id}: Data cleaning complete")
        
    except Exception as e:
        extraction_result.status = "ERROR"
        extraction_result.errors.append(str(e))
        logging.error(f"Data cleaning failed for page {page.page_id}: {e}")
    
    return extraction_result


def process_page_relationship_resolution(
    page: PageData,
    extraction_result: PageExtractionResult
) -> PageExtractionResult:
    """Run relationship resolution on a single page."""
    try:
        extraction_result.status = "RESOLVING"
        
        logging.info(f"Page {page.page_id}: Starting relationship resolution (isolated)")
        
        # Verify we have the required inputs from data cleaning
        if not extraction_result.cleaning_inputs:
            logging.warning(f"Page {page.page_id}: No cleaning_inputs, skipping relationship resolution")
            extraction_result.status = "COMPLETE"
            return extraction_result
        
        if not extraction_result.anchors:
            logging.warning(f"Page {page.page_id}: No anchors, skipping relationship resolution")
            extraction_result.status = "COMPLETE"
            return extraction_result
        
        # Stage 6.5: Module B (NER) - Entity Extraction
        # - Takes corrected tokens from Module A (stored after extraction)
        # - Generates proto-entities with types and confidence
        # - Entities passed to relationship resolver for entity-aware scoring
        # Pipeline: Data Cleaning → **Module B** → Relationship Resolver

        if BOOTSTRAP_AVAILABLE and run_module_b and extraction_result.corrected_tokens:
            logging.debug(f"Page {page.page_id}: Running Module B (NER)")
            module_b_result = run_module_b(
                corrected_tokens=extraction_result.corrected_tokens,
                page_id=page.page_id,
                document_id=page.document_id,
            )
            extraction_result.proto_entities = module_b_result.proto_entities
            
            logging.info(
                f"Page {page.page_id}: Module B complete - "
                f"{module_b_result.stats.get('proto_entities_count', 0)} entities generated"
            )
        else:
            if not extraction_result.corrected_tokens:
                logging.debug(f"Page {page.page_id}: No corrected tokens, skipping Module B")
            else:
                logging.debug(f"Page {page.page_id}: Bootstrap not available, skipping Module B")
        
        # Stage 7: Resolve relationships for THIS PAGE
        # - inputs: DataCleaningInputsV2 for this page
        # - classification: structure labels for this page
        # - anchors: candidate anchors found on this page
        # - consolidation: consolidated tables from this page
        # - proto_entities: Module B entities (if available)

        logging.debug(f"Page {page.page_id}: Resolving relationships")
        relationships = resolve_relationships_v2(
            inputs=extraction_result.cleaning_inputs,
            classification_results=extraction_result.classification or {},
            anchor_results=extraction_result.anchors,
            consolidation_results=extraction_result.consolidation,
            proto_entities=extraction_result.proto_entities,
        )
        extraction_result.relationships = relationships
        
        # Count resolved fields
        resolved_count = sum(1 for r in relationships.values() if r is not None)
        logging.info(
            f"Page {page.page_id}: Relationship resolution complete - "
            f"{resolved_count}/{len(relationships)} fields resolved"
        )
        
        extraction_result.status = "COMPLETE"
        logging.info(f"Page {page.page_id}: Processing complete")
        
    except Exception as e:
        extraction_result.status = "ERROR"
        extraction_result.errors.append(str(e))
        logging.error(f"Relationship resolution failed for page {page.page_id}: {e}")
    
    return extraction_result


def process_cross_page_stitching(
    doc_result: DocumentExtractionResult,
    document: DocumentData
) -> DocumentExtractionResult:
    """Stitch data structures split across page boundaries."""
    try:
        # Extract all pages' data in order
        pages_data = []
        for page in sorted(document.pages, key=lambda p: p.page_number):
            if page.page_id not in doc_result.page_results:
                continue
            
            page_result = doc_result.page_results[page.page_id]
            
            # Skip pages without cleaning results
            if not page_result.consolidation or not page_result.classification:
                logging.warning(f"Page {page.page_id}: Missing consolidation/classification, skipping stitching")
                continue
            
            pages_data.append({
                'page_id': page.page_id,
                'page_number': page.page_number,
                'consolidation': page_result.consolidation,
                'classification': page_result.classification,
                'cleaning_inputs': page_result.cleaning_inputs,
            })
        
        if len(pages_data) < 2:
            logging.debug("Less than 2 pages with valid data, skipping stitching")
            return doc_result
        
        # Stitch tables, lists, paragraphs
        stitcher = CrossPageStitcher()
        stitched_pages = stitcher.stitch_all(pages_data)
        
        # Update doc_result with stitched consolidation results
        for stitched_page in stitched_pages:
            page_id = stitched_page['page_id']
            if page_id in doc_result.page_results:
                doc_result.page_results[page_id].consolidation = stitched_page['consolidation']
                logging.debug(f"Page {page_id}: Updated with stitched consolidation results")
        
        logging.info(f"Document {document.document_id}: Cross-page stitching complete")
        
    except Exception as e:
        logging.error(f"Cross-page stitching failed for document {document.document_id}: {e}")
        # Don't fail the pipeline - continue with unstitched data
    
    return doc_result



# DOCUMENT-LEVEL PROCESSING


def process_document(document: DocumentData) -> DocumentExtractionResult:
    """Process all pages in a document through the full pipeline."""
    doc_result = DocumentExtractionResult(
        document_id=document.document_id,
        batch_id=document.batch_id,
        status="PROCESSING"
    )
    
    # PHASE 1: Build page queue

    page_queue = PageQueue(document_id=document.document_id)
    for page in document.pages:
        if page.status != "UNSUPPORTED":
            page_queue.add_page(page.page_id, page)
    
    logging.info(f"Document {document.document_id}: {page_queue.count()['total']} pages queued for extraction")

    # PHASE 2: Extract all pages (queue-based)

    while page_queue.has_pending():
        page_item = page_queue.get_next()
        if page_item is None:
            break
        
        page = page_item.page_data
        page_queue.update_status(page_item.id, "PROCESSING")
        
        logging.info(f"Processing page {page.page_id} ({page.page_number}/{len(document.pages)})")
        
        # Run extraction
        page_result = process_page_extraction(page)
        
        # Mark complete
        page_queue.mark_complete(page_item.id, page_result)
    
    # Transfer extracted results to doc_result
    doc_result.page_results = page_queue.get_results()
    
    # GATE: Verify all pages extracted before proceeding

    if not page_queue.all_complete():
        doc_result.status = "EXTRACTION_INCOMPLETE"
        logging.error(f"Document {document.document_id}: Extraction incomplete")
        return doc_result
    
    logging.info(f"Document {document.document_id}: All {len(doc_result.page_results)} pages extracted. Moving to data cleaning.")
    
    # PHASE 3: Data cleaning (per page - ISOLATED)
    # Each page is cleaned independently using only its own extraction data

    logging.info(f"Document {document.document_id}: Running data cleaning")
    for page in document.pages:
        if page.page_id not in doc_result.page_results:
            continue
        
        # Get THIS PAGE's extraction result only
        page_result = doc_result.page_results[page.page_id]
        
        logging.debug(f"Cleaning page {page.page_id} (isolated)")
        
        # Clean THIS PAGE only - page_result contains only this page's data
        # cluster_results, region_results, rect_data all have page_id tags
        cleaned_result = process_page_data_cleaning(page, page_result)
        
        # Store back - still isolated by page_id
        doc_result.page_results[page.page_id] = cleaned_result
    
    # PHASE 3.5: Cross-page stitching (document-level)
    # Merge tables, lists, and paragraphs split across page boundaries

    if len(document.pages) > 1:
        logging.info(f"Document {document.document_id}: Stitching cross-page structures")
        doc_result = process_cross_page_stitching(doc_result, document)
    else:
        logging.debug(f"Document {document.document_id}: Single page, skipping cross-page stitching")
    
    # PHASE 4: Relationship resolution (per page - ISOLATED)
    # Each page resolves relationships using only its own cleaned data
    # Note: consolidation may now contain stitched tables from cross-page merging

    logging.info(f"Document {document.document_id}: Resolving relationships")
    for page in document.pages:
        if page.page_id not in doc_result.page_results:
            continue
        
        # Get THIS PAGE's cleaned result only
        page_result = doc_result.page_results[page.page_id]
        
        logging.debug(f"Resolving relationships for page {page.page_id} (isolated)")
        
        # Resolve THIS PAGE only
        resolved_result = process_page_relationship_resolution(page, page_result)
        
        # Store back - still isolated by page_id
        doc_result.page_results[page.page_id] = resolved_result
    
    # PHASE 5: Aggregate results (preserving page isolation in output)

    doc_result.aggregated_results = aggregate_document_results(doc_result)
    doc_result.status = "COMPLETE"
    
    logging.info(f"Document {document.document_id}: Processing complete")
    
    return doc_result


def all_pages_extracted(doc_result: DocumentExtractionResult) -> bool:
    """Check if all pages have completed extraction stage."""
    return all(
        pr.status in ["EXTRACTED", "CLEANED", "COMPLETE"]
        for pr in doc_result.page_results.values()
    )


def _get_confidence(score: float, fallback_used: bool = False) -> str:
    """Determine confidence level based on score and fallback usage."""
    if fallback_used or score < 0.6:
        return "low"
    return "high"


def _normalize_format(fmt: str) -> str:
    """Normalize format string to standard enum values."""
    fmt_upper = (fmt or "").upper()
    mapping = {
        "TABLE": "TABLE",
        "KV": "KEY_VALUE",
        "KEY_VALUE": "KEY_VALUE",
        "LIST": "LIST",
        "PARAGRAPH": "PARAGRAPH",
        "STRING": "STRING",
        "CLUSTER": "INFO_CLUSTER",
        "INFO_CLUSTER": "INFO_CLUSTER",
    }
    return mapping.get(fmt_upper, "STRING")


def _build_result_entry(
    document_id: str,
    page_id: str,
    query_name: str,
    rel_result: Any,
) -> Dict[str, Any]:
    """Build a standardised result entry from a RelationshipResult."""
    # Extract fields from RelationshipResult (dict or dataclass)
    if isinstance(rel_result, dict):
        fmt = rel_result.get("format", "string")
        values = rel_result.get("values", [])
        score = rel_result.get("score", 0.0)
        meta = rel_result.get("meta", {})
    else:
        fmt = getattr(rel_result, "format", "string")
        values = getattr(rel_result, "values", [])
        score = getattr(rel_result, "score", 0.0)
        meta = getattr(rel_result, "meta", {})
    
    data_format = _normalize_format(fmt)
    fallback_used = meta.get("fallback_used", False) if isinstance(meta, dict) else False
    confidence = _get_confidence(score, fallback_used)
    
    # Base entry (common fields)
    entry = {
        "document_id": document_id,
        "page_id": page_id,
        "data_format": data_format,
        "query_name": query_name,
        "confidence": confidence,
    }
    
    # Add multi-page tracking for stitched content
    if isinstance(meta, dict):
        page_span = meta.get("page_span")
        if page_span:
            entry["page_span"] = {
                "start_page": page_span.get("start_page"),
                "end_page": page_span.get("end_page"),
                "spans_pages": True,
                "source_page_ids": page_span.get("page_ids", []),
            }
    
    # TABLE format: orientation-aware row assembly from TableRecordAssembler
    if data_format == "TABLE":
        orientation_case = meta.get("orientation_case", 1) if isinstance(meta, dict) else 1
        orientation = "vertical" if orientation_case in (1, 3) else "horizontal"
        entry["orientation"] = orientation
        
        assembled_records = meta.get("assembled_records", []) if isinstance(meta, dict) else []
        
        row_records = []
        if assembled_records:
            # Use pre-assembled records from TableRecordAssembler
            for i, record in enumerate(assembled_records):
                if isinstance(record, dict):
                    # Already a dict (from TableRecord.fields or serialized)
                    fields_dict = record.get("fields", record)
                    row_confidence = record.get("confidence", confidence)
                else:
                    # TableRecord dataclass
                    fields_dict = getattr(record, "fields", {})
                    row_confidence = getattr(record, "confidence", 1.0)
                    row_confidence = _get_confidence(row_confidence, fallback_used)
                
                row_records.append({
                    "row_index": i,
                    "record": fields_dict,  # Full row: {"Part Number": "X", "Quantity": "1", "Material": "Steel"}
                    "confidence": row_confidence if isinstance(row_confidence, str) else _get_confidence(row_confidence, fallback_used),
                })
        else:
            # Fallback: build from values list (single column)
            for i, val in enumerate(values):
                if isinstance(val, dict):
                    row_records.append({
                        "row_index": i,
                        "record": val,
                        "confidence": confidence,
                    })
                else:
                    row_records.append({
                        "row_index": i,
                        "record": {query_name: val},
                        "confidence": confidence,
                    })
        
        entry["value"] = row_records
        
        # Compute top-level confidence from row confidences
        if row_records:
            row_conf_values = [
                0.8 if r.get("confidence") == "high" else 0.4 
                for r in row_records
            ]
            avg_conf = sum(row_conf_values) / len(row_conf_values)
            entry["confidence"] = "high" if avg_conf >= 0.6 else "low"
    else:
        # Non-table: scalar value (first value or joined)
        entry["orientation"] = None
        if len(values) == 1:
            entry["value"] = values[0]
        elif len(values) > 1:
            entry["value"] = values[0]  # Primary value
        else:
            entry["value"] = None
    
    return entry


def aggregate_document_results(doc_result: DocumentExtractionResult) -> Dict[str, Any]:
    """Aggregate all page results into standardized output format."""
    document_id = doc_result.document_id
    
    aggregated = {
        "document_id": document_id,
        "batch_id": doc_result.batch_id,
        "total_pages": len(doc_result.page_results),
        "pages": {},
        "document_summary": {
            "entries": [],
            "total_entries": 0,
            "pages_completed": 0,
            "pages_with_errors": 0,
        }
    }
    
    all_entries: List[Dict[str, Any]] = []
    stitched_refs: Dict[str, List[Dict[str, Any]]] = {}  # secondary_page_id -> [references]
    
    # First pass: collect results and track stitched content
    for page_id, page_result in doc_result.page_results.items():
        page_data = {
            "page_id": page_id,
            "status": page_result.status,
            "results": [],
            "errors": page_result.errors if page_result.errors else None,
        }
        
        # Build result entries for this page
        if page_result.relationships:
            for query_name, rel_result in page_result.relationships.items():
                if rel_result is not None:
                    entry = _build_result_entry(document_id, page_id, query_name, rel_result)
                    page_data["results"].append(entry)
                    all_entries.append(entry)
                    
                    # Track stitched content for cross-references
                    page_span = entry.get("page_span")
                    if page_span and page_span.get("spans_pages"):
                        for secondary_page_id in page_span.get("source_page_ids", []):
                            if secondary_page_id != page_id:
                                if secondary_page_id not in stitched_refs:
                                    stitched_refs[secondary_page_id] = []
                                stitched_refs[secondary_page_id].append({
                                    "type": "stitched_reference",
                                    "query_name": query_name,
                                    "data_format": entry.get("data_format"),
                                    "primary_page": page_id,
                                    "page_range": f"{page_span['start_page']}-{page_span['end_page']}",
                                    "note": f"Content continues from page {page_span['start_page']}"
                                })
        
        aggregated["pages"][page_id] = page_data
        
        # Update stats
        if page_result.status == "COMPLETE":
            aggregated["document_summary"]["pages_completed"] += 1
        if page_result.errors:
            aggregated["document_summary"]["pages_with_errors"] += 1
    
    # Second pass: add cross-references to secondary pages
    for page_id, refs in stitched_refs.items():
        if page_id in aggregated["pages"]:
            aggregated["pages"][page_id]["stitched_references"] = refs
    
    # Document-level flattened summary
    aggregated["document_summary"]["entries"] = all_entries
    aggregated["document_summary"]["total_entries"] = len(all_entries)
    
    return aggregated


# BATCH-LEVEL PROCESSING


def process_batch(batch: BatchData) -> BatchExtractionResult:
    """Process all documents in a batch through the full pipeline."""
    batch_result = BatchExtractionResult(
        batch_id=batch.batch_id,
        status="PROCESSING"
    )
    
    logging.info(f"Processing batch {batch.batch_id}: {batch.document_count} documents, {batch.total_pages} total pages")
    
    # Build document queue
    doc_queue = DocumentQueue(batch_id=batch.batch_id)
    for document in batch.documents:
        doc_queue.add_document(document.document_id, document)
    
    logging.info(f"Batch {batch.batch_id}: {doc_queue.count()['total']} documents queued")
    
    # Process all documents (queue-based)

    while doc_queue.has_pending():
        doc_item = doc_queue.get_next()
        if doc_item is None:
            break
        
        document = doc_item.document_data
        doc_queue.update_status(doc_item.id, "PROCESSING")
        
        logging.info(f"Processing document {document.document_id} ({len(document.pages)} pages)")
        
        # Process document
        doc_result = process_document(document)
        
        # Mark complete
        doc_queue.mark_complete(doc_item.id, doc_result)
    
    # Transfer completed results to batch_result
    batch_result.document_results = doc_queue.get_results()
    
    # Verify all documents completed

    if not doc_queue.all_complete():
        batch_result.status = "INCOMPLETE"
        logging.error(f"Batch {batch.batch_id}: Processing incomplete")
        return batch_result
    
    batch_result.status = "COMPLETE"
    logging.info(
        f"Batch {batch.batch_id}: Processing complete - "
        f"{len(batch_result.document_results)} documents processed"
    )
    
    # PHASE 6: Internal Validation (batch-level)
    # Validate metadata quality, detect missing data, check schema consistency

    logging.info(f"Batch {batch.batch_id}: Running internal validation")
    try:
        validator = InternalValidator(batch_result)
        validation_results = validator.validate()
        batch_result.validation_results = validation_results
        
        logging.info(
            f"Batch {batch.batch_id}: Validation complete - "
            f"{validation_results['validation_summary']['total_queries_found']} queries found, "
            f"{validation_results['validation_summary']['missing_data_candidates']} missing data candidates, "
            f"{validation_results['validation_summary']['total_discrepancies']} discrepancies"
        )
    except Exception as e:
        logging.error(f"Batch {batch.batch_id}: Validation failed - {e}")
        batch_result.validation_results = {
            "maiden_id": batch.batch_id,
            "validation_status": "failed",
            "error": str(e),
        }
    
    return batch_result


# OUTPUT: FINAL RESULTS ONLY

def save_final_results(
    batch_result: BatchExtractionResult,
    output_dir: str = "outputs"
) -> Dict[str, str]:
    """Save final extraction results to disk.
    This is the ONLY place where results are written to files."""
    output_paths = {}
    
    batch_dir = os.path.join(output_dir, batch_result.batch_id)
    os.makedirs(batch_dir, exist_ok=True)
    
    for doc_id, doc_result in batch_result.document_results.items():
        doc_dir = os.path.join(batch_dir, doc_id)
        os.makedirs(doc_dir, exist_ok=True)
        
        output_path = os.path.join(doc_dir, "extraction_results.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_result.aggregated_results, f, indent=2, ensure_ascii=False)
        
        output_paths[doc_id] = output_path
        logging.info(f"Saved results: {output_path}")
    
    # Save batch-level validation results
    if batch_result.validation_results:
        validation_path = os.path.join(batch_dir, "validation_results.json")
        with open(validation_path, 'w', encoding='utf-8') as f:
            json.dump(batch_result.validation_results, f, indent=2, ensure_ascii=False)
        output_paths["_validation"] = validation_path
        logging.info(f"Saved validation results: {validation_path}")
    
    return output_paths

# MAIN PIPELINE ENTRY POINT

def process_batch_queue(batch_queue: BatchQueue, output_dir: str = "outputs") -> Dict[str, Any]:
    """Process all batches in the queue sequentially. """
    all_results = {
        "batches_processed": 0,
        "total_documents": 0,
        "total_pages": 0,
        "batch_summaries": {},
        "output_paths": {},
    }
    
    # Initialise OCR models once when pipeline starts (offline server paths)
    try:
        prewarm_cluster_ocr()
        logging.info("Cluster OCR (PaddleOCR) prewarmed")
    except Exception as e:
        logging.warning(f"Cluster OCR prewarm failed (will lazy-load on first use): {e}")
    try:
        prewarm_region_doctr()
        logging.info("Region OCR (DocTR) prewarmed")
    except Exception as e:
        logging.warning(f"Region DocTR prewarm failed (will lazy-load on first use): {e}")

    logging.info(f"Starting batch queue processing ({batch_queue.count()['pending']} batches)")
    
    while batch_queue.has_pending():
        batch_item = batch_queue.get_next()
        if batch_item is None:
            break
        
        batch = batch_item.batch_data
        batch_queue.update_status(batch_item.id, "PROCESSING")
        
        logging.info(
            f"Processing batch {batch.batch_id} "
            f"({batch.document_count} documents, {batch.total_pages} pages)"
        )
        
        # Process this batch
        batch_result = process_batch(batch)
        
        # Save results
        output_paths = save_final_results(batch_result, output_dir)
        
        # Mark complete
        batch_queue.mark_complete(batch_item.id, batch_result)
        
        # Update summary
        all_results["batches_processed"] += 1
        all_results["total_documents"] += len(batch_result.document_results)
        all_results["total_pages"] += batch.total_pages
        all_results["batch_summaries"][batch.batch_id] = {
            "status": batch_result.status,
            "documents": len(batch_result.document_results),
            "pages": batch.total_pages,
        }
        all_results["output_paths"].update(output_paths)
    
    return all_results


def pipeline_main(input_dir: str, output_dir: str = "outputs") -> Dict[str, Any]:
    """ Main entry point for the document processing pipeline."""
    logging.info(f"Starting pipeline for input: {input_dir}")
    
    # Ingest
    batch = process_input_local(input_dir)
    logging.info(f"Ingested: {get_batch_summary(batch)}")
    
    # Build queue using queue_manager
    batch_queue = build_batch_queue_from_batches([batch])
    
    # Process
    results = process_batch_queue(batch_queue, output_dir)
    
    logging.info(
        f"Pipeline complete: {results['batches_processed']} batches, "
        f"{results['total_documents']} documents, {results['total_pages']} pages"
    )
    
    return results


def pipeline_main_multi_batch(input_dirs: List[str], output_dir: str = "outputs") -> Dict[str, Any]:
    """ Process multiple input directories as separate batches."""
    logging.info(f"Starting multi-batch pipeline for {len(input_dirs)} input directories")
    
    # Collect batches
    batches = []
    for input_dir in input_dirs:
        if os.path.exists(input_dir):
            batch = process_input_local(input_dir)
            batches.append(batch)
            logging.info(f"Queued batch from {input_dir}: {get_batch_summary(batch)}")
        else:
            logging.warning(f"Input directory not found, skipping: {input_dir}")
    
    if not batches:
        logging.error("No valid batches to process")
        return {"error": "No valid input directories"}
    
    # Build queue using queue_manager
    batch_queue = build_batch_queue_from_batches(batches)
    
    # Process
    results = process_batch_queue(batch_queue, output_dir)
    
    logging.info(
        f"Multi-batch pipeline complete: {results['batches_processed']} batches, "
        f"{results['total_documents']} documents, {results['total_pages']} pages"
    )
    
    return results

# CLI ENTRY POINT

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "debug"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    
    if os.path.exists(input_dir):
        result = pipeline_main(input_dir, output_dir)
        print(json.dumps(result, indent=2))
    else:
        print(f"Input directory not found: {input_dir}")
