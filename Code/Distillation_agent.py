import json
import logging
import re
from copy import deepcopy
from typing import Any, Dict, List

from openai import OpenAI
from base_agent import BaseAgent
from document_utils import format_chunk, split_text

logger = logging.getLogger(__name__)


DISTILLATION_CHUNK_SIZE = 18000
DISTILLATION_CHUNK_OVERLAP = 500
DISTILLATION_DEFAULT_RETRIES = 2
TRACEABILITY_MAX_FEEDBACK_ITEMS = 25


class ContentDistillationAgent(BaseAgent):
    def __init__(self, client: OpenAI, model_name: str, max_retries: int = DISTILLATION_DEFAULT_RETRIES):
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        self.max_retries = max_retries

        system_prompt = """You are an expert Methodologist and Senior Data Extractor conducting a PRISMA-compliant Systematic Review and Meta-Analysis. Your objective is to systematically and exhaustively extract trial data from the provided academic full-text, strictly aligning with the user-defined "Research Topic".

[CRITICAL EXTRACTION PROTOCOLS]:
1. EXHAUSTIVE & UNBIASED EXPLORATION: Scrutinize the full text, figures, tables, and the appended PDF Layout Audit if present. Capture all baseline demographics, methodological parameters, primary/secondary outcomes, and effect estimates relevant to the Research Topic. Extract ALL data relevant to the topic, but strictly EXCLUDE tertiary exploratory variables, anecdotal observations, or raw patient-level listings unless they directly support the primary/secondary outcomes.
2. ADAPTIVE SEMI-STRUCTURED ARCHITECTURE: Your output MUST be a valid JSON object. You should prioritize universal root-level domains such as: "study_metadata", "baseline_characteristics", "interventions_or_exposures", "clinical_outcomes", and "adverse_events". HOWEVER, to accommodate diverse meta-analysis archetypes (e.g., Diagnostic Accuracy, Prognostic, Pharmacological), you are explicitly authorized to synthesize customized root-level keys (e.g., "diagnostic_performance_metrics", "survival_analysis", "pharmacokinetic_parameters") IF the paper's specific data paradigm fundamentally demands it.
3. STUDY ARM & SUBGROUP STRATIFICATION: Data MUST be rigorously stratified by study arms (e.g., Intervention vs. Control, Exposed vs. Unexposed, or Target Test vs. Reference Standard). If metrics are reported across subgroups (e.g., Male/Female) or longitudinal timepoints (e.g., Baseline, 12-week), create distinctly named nested keys (e.g., "sensitivity_male_cohort", "systolic_bp_control_12wk"). Do NOT aggregate independent strata.
4. DATA PROVENANCE (TRACEABILITY): Every terminal extracted metric MUST be structured as a nested dictionary containing EXACTLY two keys: "value" and "source_location".
Example: "patient_age_intervention": {"value": "45.2 ± 3.1", "source_location": "Table 1, Column 2, Row 3"}
The source_location must be precise and reviewable, e.g., Table/Figure number, page number, section, paragraph, row/column, or appended layout-audit table block when applicable.
5. STATISTICAL FIDELITY: Extract effect measures and measures of dispersion exactly as published (e.g., "Mean ± SD", "Median (IQR)", "HR 0.85 (95% CI 0.70-0.95), p=0.04"). Do not impute, compute, or decompose composite statistical strings unless explicitly separated in the source text.
6. Extract solely what is explicitly documented in the supplied segment. If this segment contains no data relevant to the Research Topic, output an empty JSON object: {}. Do not infer that the rest of the paper lacks relevant data.
7. READABLE OUTPUT: Output ONLY a valid, well-formed JSON object. Do not include markdown wrappers, prefatory text, or concluding remarks. If a specific metric, subgroup, or category is not mentioned in the text, DO NOT create a key for it with a 'Not Reported' or 'N/A' value. Completely omit the key to keep the JSON concise and dense.
8. PROGRAMMATIC AUDIT COMPATIBILITY: The downstream deterministic traceability auditor will reject terminal scalar leaves, missing source_location fields, empty source locations, and values that cannot be textually supported. Therefore, do not output raw strings/numbers/lists as terminal metrics; always wrap terminal metrics as {"value": ..., "source_location": ...}."""
        super().__init__(client, model_name, system_prompt)

    def _normalise_for_matching(self, text: Any) -> str:
        """Normalize text for approximate deterministic value-support checks."""
        if text is None:
            return ""
        text = str(text)
        replacements = {
            "\u00a0": " ",
            "−": "-",
            "–": "-",
            "—": "-",
            "﹣": "-",
            "±": "+/-",
            "·": ".",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()

    def _is_missing_like(self, value: Any) -> bool:
        missing_tokens = {"", "n/a", "na", "not reported", "nr", "none", "null", "unclear"}
        return self._normalise_for_matching(value) in missing_tokens

    def _value_supported_by_text(self, value: Any, source_text: str) -> bool:
        """Check approximate textual support using normalized text and token matching."""
        value_norm = self._normalise_for_matching(value)
        text_norm = self._normalise_for_matching(source_text)
        if not value_norm or value_norm in text_norm:
            return True

        # Check numeric-token overlap when exact matching fails.
        numeric_tokens = re.findall(r"(?<![A-Za-z])-?\d+(?:\.\d+)?%?", value_norm.replace(",", ""))
        if numeric_tokens:
            matched = sum(1 for token in numeric_tokens if token in text_norm)
            return matched / max(len(numeric_tokens), 1) >= 0.6

        # Check word overlap for non-numeric labels.
        words = [w for w in re.findall(r"[a-zA-Z]{4,}", value_norm) if
                 w not in {"with", "from", "that", "this", "were", "study"}]
        if not words:
            return True
        matched = sum(1 for word in words if word in text_norm)
        return matched / max(len(words), 1) >= 0.5

    def _validate_traceable_json(self, data: Any, source_text: str) -> Dict[str, Any]:
        """
        Programmatically validate the source-traceable extraction contract.

        Required contract:
        - every terminal metric is a dict with exactly {"value", "source_location"};
        - source_location is non-empty and reviewable;
        - numeric/textual values are at least approximately supported by the parsed paper text.
        """
        errors: List[str] = []
        warnings: List[str] = []
        metric_count = 0
        checked_values = 0
        supported_values = 0

        if not isinstance(data, dict) or "error" in data:
            return {
                "decision": "reject",
                "metric_count": 0,
                "checked_values": 0,
                "supported_values": 0,
                "errors": ["Extraction must be a JSON object of traceable metrics; error objects are not data."],
                "warnings": [],
            }

        def visit(obj: Any, path: str) -> None:
            nonlocal metric_count, checked_values, supported_values
            if isinstance(obj, dict):
                keys = set(obj.keys())
                if "value" in keys or "source_location" in keys:
                    metric_count += 1
                    if keys != {"value", "source_location"}:
                        errors.append(
                            f"{path}: terminal metric must contain exactly ['value', 'source_location']; found {sorted(keys)}"
                        )
                    value = obj.get("value")
                    source_location = obj.get("source_location")
                    if self._is_missing_like(value):
                        errors.append(
                            f"{path}.value: missing-like values should be omitted during extraction, not reported as a metric.")
                    if self._is_missing_like(source_location):
                        errors.append(f"{path}.source_location: missing or non-reviewable source location.")
                    else:
                        loc_norm = self._normalise_for_matching(source_location)
                        reviewable_markers = [
                            "table", "fig", "figure", "page", "paragraph", "section", "result", "method",
                            "appendix", "supplement", "row", "column", "m3 pdf layout audit"
                        ]
                        if not any(marker in loc_norm for marker in reviewable_markers):
                            warnings.append(
                                f"{path}.source_location: location may be too coarse for rapid human review: {source_location}"
                            )
                    if not self._is_missing_like(value):
                        checked_values += 1
                        if self._value_supported_by_text(value, source_text):
                            supported_values += 1
                        else:
                            errors.append(
                                f"{path}.value: value cannot be approximately matched to the parsed paper text: {value}"
                            )
                    return

                if not obj:
                    warnings.append(f"{path}: empty object encountered.")
                for key, value in obj.items():
                    next_path = f"{path}.{key}" if path else str(key)
                    visit(value, next_path)
                return

            if isinstance(obj, list):
                if not obj:
                    warnings.append(f"{path}: empty list encountered.")
                for idx, item in enumerate(obj):
                    visit(item, f"{path}[{idx}]")
                return

            # Any scalar leaf is a violation of the source-traceability contract.
            errors.append(
                f"{path}: terminal scalar/list item is not source-traceable; wrap it as {{'value', 'source_location'}}.")

        if "value" in data or "source_location" in data:
            errors.append("Root object must contain named extraction fields, not an unnamed terminal metric.")
        visit(data, "root")
        return {
            "decision": "reject" if errors else "accept",
            "metric_count": metric_count,
            "checked_values": checked_values,
            "supported_values": supported_values,
            "support_rate": round(supported_values / checked_values, 4) if checked_values else 1.0,
            "errors": errors,
            "warnings": warnings,
        }

    def _traceability_feedback(self, audit: Dict[str, Any]) -> str:
        errors = audit.get("errors", [])[:TRACEABILITY_MAX_FEEDBACK_ITEMS]
        warnings = audit.get("warnings", [])[:10]
        parts = [
            "The deterministic source-traceability audit rejected the extraction.",
            f"Metric count: {audit.get('metric_count', 0)}; checked values: {audit.get('checked_values', 0)}; support rate: {audit.get('support_rate', 'N/A')}",
        ]
        if errors:
            parts.append("Errors:\n- " + "\n- ".join(errors))
        if warnings:
            parts.append("Warnings:\n- " + "\n- ".join(warnings))
        parts.append(
            "Regenerate the JSON so that every terminal metric has exactly {'value', 'source_location'}, "
            "every source_location is precise, and reported values are explicitly supported by the paper text."
        )
        return "\n".join(parts)

    def _run_checker(self, text: str, data: dict, topic: str, programmatic_audit: Dict[str, Any]) -> dict:
        checker_prompt = f"""Audit this extracted data for accuracy and relevance to the topic: "{topic}"
        Data: {json.dumps(data, indent=2, ensure_ascii=False)}
        Programmatic Traceability Audit: {json.dumps(programmatic_audit, indent=2, ensure_ascii=False)}
        Text Context: {text}

      [AUDIT RULES]:
        1. Data Accuracy: Whether the data in the integrated table exactly matches the original tables/text/layout-audit table blocks.
        2. Semantic Consistency: Whether the data meanings remain consistent with the source tables, figures, or source paragraphs.
        3. Source Traceability: Whether every terminal metric uses exactly {{"value", "source_location"}}; whether the source_location is precise enough for human review; and whether the value is supported by the cited source.
        4. Segment Coverage: Review the entire supplied segment, including tables or layout-audit blocks. An empty extraction may be accepted ONLY when this segment contains no relevant extractable data. Never reject solely because a different segment of the paper is absent. Preserve arm, subgroup, outcome and timepoint distinctions.
        5. The programmatic audit must also accept. An error object, malformed candidate, or unsupported value cannot be accepted.
        - REJECT IF: The data contains hallucinated values not in the text, violates the JSON traceability format, contains terminal scalar leaves, uses empty/coarse source locations for critical outcomes, or misses MASSIVE amounts of critical primary outcomes.
        - ACCEPT IF: The extraction is generally accurate and source-traceable, even if minor secondary variables are missed. (You can provide suggestions in 'feedback' for the next run, but do not reject for minor omissions).

        Output JSON: {{ 
          "decision": "accept/reject", 
          "feedback": "Detailed feedback. If rejected, clearly state the hallucination, traceability violation, or major error."
        }}"""
        try:
            content, _, _, _ = self._make_llm_call(checker_prompt, temperature=0, json_mode=True)
            result = json.loads(self._clean_json_string(content))
            if not isinstance(result, dict) or "error" in result or result.get("decision") not in {"accept", "reject"}:
                raise ValueError("Checker must return an explicit accept or reject decision")
            return result
        except Exception as exc:
            logger.warning("LLM Checker failed: %s", exc)
            return {
                "decision": "reject",
                "feedback": "LLM checker failed, so the extraction was not auto-accepted. Regenerate strict valid JSON with exact value/source_location traceability.",
            }

    def _clean_json_string(self, content: str) -> str:
        return content.replace("```json", "").replace("```", "").strip()

    @staticmethod
    def _merge_data(left: Any, right: Any) -> Any:
        """Combine extracted fields without overwriting distinct metric records."""
        if left == right:
            return deepcopy(left)
        left_metric = isinstance(left, dict) and bool({"value", "source_location"} & left.keys())
        right_metric = isinstance(right, dict) and bool({"value", "source_location"} & right.keys())
        if isinstance(left, dict) and isinstance(right, dict) and not left_metric and not right_metric:
            merged = deepcopy(left)
            for key, value in right.items():
                merged[key] = ContentDistillationAgent._merge_data(merged[key], value) if key in merged else deepcopy(value)
            return merged
        records = deepcopy(left) if isinstance(left, list) else [deepcopy(left)]
        additions = right if isinstance(right, list) else [right]
        for value in additions:
            if value not in records:
                records.append(deepcopy(value))
        return records

    def _extract_chunk(self, chunk, topic: str, total_chunks: int) -> Dict[str, Any]:
        segment_text = format_chunk(chunk)
        base_prompt = f"""Research Topic: "{topic}"

[Paper Segment {chunk.index} of {total_chunks}]:
{segment_text}

Extract all relevant data in this segment. Segments overlap; retain the original table,
section, arm, subgroup, outcome and timepoint in each field and source_location.
Do not treat segment boundaries as study boundaries or infer absent table cells.
Return {{}} only if this entire segment has no relevant extractable data.
Output ONLY a JSON object."""
        curr_prompt = base_prompt
        attempts = []
        last_candidate = None
        for attempt in range(self.max_retries + 1):
            logger.info("Extracting segment %s/%s, attempt %s/%s", chunk.index, total_chunks,
                        attempt + 1, self.max_retries + 1)
            candidate = None
            parse_error = None
            try:
                content, _, _, _ = self._make_llm_call(curr_prompt, temperature=0.0, json_mode=True)
                candidate = json.loads(self._clean_json_string(content))
                if candidate == {"error": "No relevant data found for this topic"}:
                    candidate = {}
            except Exception as exc:
                parse_error = str(exc)
                logger.warning("Extraction failed for segment %s: %s", chunk.index, exc)

            last_candidate = candidate
            programmatic_audit = self._validate_traceable_json(candidate, chunk.text)
            if parse_error:
                programmatic_audit["errors"].append(parse_error)
            checker_candidate = candidate if candidate is not None else {"error": "No parseable extraction candidate"}
            check = self._run_checker(segment_text, checker_candidate, topic, programmatic_audit)
            attempts.append({
                "attempt": attempt + 1,
                "programmatic_audit": programmatic_audit,
                "semantic_check": check,
            })
            if programmatic_audit["decision"] == "accept" and check.get("decision") == "accept":
                return {"status": "accepted", "data": candidate, "attempts": attempts}

            feedback = []
            if programmatic_audit["decision"] == "reject":
                feedback.append(self._traceability_feedback(programmatic_audit))
            feedback.append("Semantic Checker: " + str(check.get("feedback", "The candidate was not accepted.")))
            curr_prompt = base_prompt + "\n\n[CHECKER FEEDBACK]:\n" + "\n".join(feedback)
            curr_prompt += "\nRegenerate the complete segment extraction as valid JSON."
        return {"status": "rejected", "attempts": attempts, "last_candidate": last_candidate}

    def process(self, markdown_text: str, topic: str) -> dict:
        if not isinstance(markdown_text, str) or not markdown_text.strip():
            return {
                "error": "Content distillation requires non-empty paper text",
                "validation_status": "incomplete", "partial_data": {},
                "audit": {"chunks": []},
                "coverage": {"total_characters": len(markdown_text) if isinstance(markdown_text, str) else 0,
                             "chunk_count": 0, "processed_chunks": 0, "accepted_chunks": 0,
                             "failed_chunks": 0, "complete": False, "all_chunks_processed": False},
            }

        chunks = split_text(markdown_text, max_chars=DISTILLATION_CHUNK_SIZE,
                            overlap=DISTILLATION_CHUNK_OVERLAP)
        merged_data = {}
        chunk_audits = []
        accepted_chunks = 0
        for chunk in chunks:
            result = self._extract_chunk(chunk, topic, len(chunks))
            chunk_audit = {"index": chunk.index, "start": chunk.start, "end": chunk.end,
                           "status": result["status"], "attempts": result["attempts"]}
            if result["status"] == "accepted":
                accepted_chunks += 1
                merged_data = self._merge_data(merged_data, result["data"])
            else:
                chunk_audit["last_candidate"] = result["last_candidate"]
            chunk_audits.append(chunk_audit)

        complete = accepted_chunks == len(chunks)
        coverage = {
            "total_characters": len(markdown_text), "chunk_count": len(chunks),
            "processed_chunks": len(chunk_audits), "accepted_chunks": accepted_chunks,
            "failed_chunks": len(chunks) - accepted_chunks,
            "complete": complete, "all_chunks_processed": len(chunk_audits) == len(chunks),
        }
        result = {"validation_status": "accepted" if complete else "incomplete" if accepted_chunks else "rejected",
                  "audit": {"chunks": chunk_audits}, "coverage": coverage}
        if complete:
            result["data"] = merged_data
        else:
            result["error"] = "One or more paper segments did not pass extraction validation"
            result["partial_data"] = merged_data
        return result
