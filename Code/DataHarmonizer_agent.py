import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List

from openai import OpenAI
from base_agent import BaseAgent
from document_utils import batch_records

logger = logging.getLogger(__name__)

HARMONIZER_BATCH_CHARS = 40000
HARMONIZER_REGISTRY_PAGE_CHARS = 12000
CONTEXT_FIELDS = ("arm", "timepoint", "unit", "statistic", "population")
DESCRIPTOR_FIELDS = ("variable",) + CONTEXT_FIELDS


class DataHarmonizerAgent(BaseAgent):
    """Align complete batches of metrics, then assemble values from their sources."""

    def __init__(self, client: OpenAI, model_name: str):
        system_prompt = """You are a medical data harmonizer for meta-analysis.
Map field names and identify semantically equivalent variables. Never invent, convert,
aggregate or return replacement values. Preserve differences in treatment arm,
population/subgroup, measurement time, statistic and unit. Distinguish event counts,
sample sizes, proportions and effect estimates. Use only the supplied identifiers.
Return the requested JSON object with one response entry for every supplied target."""
        super().__init__(client, model_name, system_prompt)

    def _is_traceable_leaf(self, obj: Any) -> bool:
        return isinstance(obj, dict) and "value" in obj and "source_location" in obj

    def _untraceable_paths(self, obj: Any, path: str = "data") -> list:
        if isinstance(obj, dict):
            if "value" in obj or "source_location" in obj:
                return [] if set(obj) == {"value", "source_location"} else [path]
            return [issue for key, value in obj.items()
                    for issue in self._untraceable_paths(value, f"{path}.{key}")]
        if isinstance(obj, list):
            return [issue for index, value in enumerate(obj)
                    for issue in self._untraceable_paths(value, f"{path}[{index}]")]
        return [path]

    def _collect_traceable_metrics(self, obj: Any, source: str, path: str = "data", context=None) -> List[Dict[str, Any]]:
        context = context or []
        if self._is_traceable_leaf(obj):
            record = {"source": source, "original_key": path,
                      "value": obj["value"], "source_location": obj["source_location"]}
            related = [item for item in context if item["original_key"] != path]
            if related:
                record["extraction_context"] = related
            return [record]
        records = []
        if isinstance(obj, dict):
            # Carry explicit contextual labels with their numeric siblings.
            contextual = []
            for key, value in obj.items():
                if self._is_traceable_leaf(value) and re.search(
                    r"arm|group|intervention|exposure|time|follow|unit|population|"
                    r"outcome|endpoint|statistic|measure|cohort|treatment|visit|label|name",
                    str(key), re.IGNORECASE,
                ):
                    contextual.append({"original_key": f"{path}.{key}",
                                       "value": value["value"], "source_location": value["source_location"]})
            for key, value in obj.items():
                records.extend(self._collect_traceable_metrics(value, source, f"{path}.{key}", context + contextual))
        elif isinstance(obj, list):
            for index, value in enumerate(obj):
                records.extend(self._collect_traceable_metrics(value, source, f"{path}[{index}]", context))
        return records

    def _build_traceability_index(self, raw_extractions_list: list) -> List[Dict[str, Any]]:
        records = []
        for paper_index, item in enumerate(raw_extractions_list, 1):
            source = item.get("source", f"paper_{paper_index}") if isinstance(item, dict) else f"paper_{paper_index}"
            data = item.get("data", item) if isinstance(item, dict) else item
            for record in self._collect_traceable_metrics(data, source):
                record["paper_id"] = f"paper_{paper_index:05d}"
                record["metric_id"] = f"metric_{len(records) + 1:08d}"
                records.append(record)
        return records

    @staticmethod
    def _normalise_label(value: str) -> str:
        return " ".join(value.strip().casefold().split())

    @classmethod
    def _descriptor_label(cls, field: str, value: str) -> str:
        # Unit symbols can be case-sensitive (for example, mM and mm).
        return " ".join(value.strip().split()) if field == "unit" else cls._normalise_label(value)

    @staticmethod
    def _load_response(content: str) -> dict:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object.")
        if "error" in result:
            raise ValueError("Harmonization returned an error object.")
        return result

    @staticmethod
    def _validate_ids(entries: Any, id_key: str, expected_ids: list) -> None:
        if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
            raise ValueError("Response entries must be an array of objects.")
        actual = [entry.get(id_key) for entry in entries]
        if any(not isinstance(item, str) for item in actual):
            raise ValueError(f"Every response entry must have a string {id_key}.")
        if len(actual) != len(set(actual)):
            raise ValueError(f"Duplicate {id_key} in response.")
        missing, unknown = sorted(set(expected_ids) - set(actual)), sorted(set(actual) - set(expected_ids))
        if missing or unknown:
            raise ValueError(f"Identifier coverage mismatch: missing={missing}, unknown={unknown}.")

    def _map_batch(self, records: list, topic: str) -> dict:
        prompt = f"""[Task: map_metrics]
Research topic: {topic}
For each complete source metric below, name the clinical variable and its context.
Read original_key, value and any extraction_context when determining context.
Context records carry explicit labels from the same parent or ancestor object;
match the relevant object paths and preserve conflicts instead of guessing.
Return this schema:
{{"mappings":[{{"metric_id":"supplied id","variable":"canonical variable name",
"arm":"treatment arm or overall","timepoint":"measurement time or unspecified",
"unit":"original unit or unspecified","statistic":"mean/event count/etc. or unspecified",
"population":"population/subgroup or overall"}}]}}
Use consistent concise English labels. Include every metric_id exactly once.
Do not confuse randomized population with analyzed population, target with achieved
blood pressure, baseline with follow-up, or intensive with standard treatment.
Use unspecified when the original extraction does not identify a context; do not
infer missing units or context from clinical convention. Do not convert units.
All six descriptor fields are required non-empty strings. Do not return values.
[Records JSON]
{json.dumps(records, ensure_ascii=False)}"""
        content, _, _, _ = self._make_llm_call(prompt, temperature=0.0, json_mode=True)
        entries = self._load_response(content).get("mappings")
        self._validate_ids(entries, "metric_id", [record["metric_id"] for record in records])
        mappings = {}
        for entry in entries:
            descriptor = {}
            for field in DESCRIPTOR_FIELDS:
                value = entry.get(field)
                if not isinstance(value, str) or not value.strip() or len(value) > 256:
                    raise ValueError(f"Invalid {field} for {entry['metric_id']}.")
                descriptor[field] = " ".join(value.split())
            mappings[entry["metric_id"]] = descriptor
        return mappings

    def _align_pages(self, targets: list, registry: list) -> list:
        prompt = f"""[Task: align_descriptors]
Identify synonyms for the SAME clinical variable among these complete descriptors.
For each target descriptor_id return every semantically equivalent descriptor_id
in the registry page, or an empty list if none. Context must match: do not merge
different arms, times, units, statistics, populations, outcome definitions, randomized
versus analyzed population, or target versus achieved measurements.
Return {{"alignments":[{{"descriptor_id":"target id","equivalent_ids":["registry id"]}}]}}.
Include every target id exactly once, and use only registry ids in equivalent_ids.
[Targets JSON]
{json.dumps(targets, ensure_ascii=False)}
[Registry JSON]
{json.dumps(registry, ensure_ascii=False)}"""
        content, _, _, _ = self._make_llm_call(prompt, temperature=0.0, json_mode=True)
        entries = self._load_response(content).get("alignments")
        self._validate_ids(entries, "descriptor_id", [entry["descriptor_id"] for entry in targets])
        registry_ids = {entry["descriptor_id"] for entry in registry}
        for entry in entries:
            matches = entry.get("equivalent_ids")
            if (not isinstance(matches, list) or any(not isinstance(value, str) for value in matches)
                    or len(matches) != len(set(matches)) or not set(matches) <= registry_ids):
                raise ValueError(f"Invalid equivalent_ids for {entry['descriptor_id']}.")
        return entries

    def _canonicalise(self, mappings: dict, errors: list) -> tuple:
        """Compare all registry pages within an identical context, including across batches."""
        descriptors, signatures, metric_descriptors = {}, {}, {}
        for metric_id, descriptor in sorted(mappings.items()):
            signature = tuple(self._descriptor_label(field, descriptor[field]) for field in DESCRIPTOR_FIELDS)
            if signature not in signatures:
                descriptor_id = f"descriptor_{len(descriptors) + 1:07d}"
                signatures[signature] = descriptor_id
                descriptors[descriptor_id] = {"descriptor_id": descriptor_id, **descriptor}
            metric_descriptors[metric_id] = signatures[signature]
        parents = {key: key for key in descriptors}

        def find(key):
            while parents[key] != key:
                parents[key] = parents[parents[key]]
                key = parents[key]
            return key

        context_groups = defaultdict(list)
        for descriptor in descriptors.values():
            context = tuple(self._descriptor_label(field, descriptor[field]) for field in CONTEXT_FIELDS)
            context_groups[context].append(descriptor)
        unresolved = set()
        comparisons = successful = 0
        for group in context_groups.values():
            # Never send records from incompatible arms/times/units in the same registry.
            pages = batch_records(group, HARMONIZER_REGISTRY_PAGE_CHARS)
            for page_index, targets in enumerate(pages):
                for registry in pages[page_index:]:
                    if targets is registry and len(targets) == 1:
                        continue
                    comparisons += 1
                    affected = {entry["descriptor_id"] for entry in targets + registry}
                    try:
                        entries = self._align_pages(targets, registry)
                        # The model cannot introduce descriptors outside this context group.
                        for entry in entries:
                            for other in entry["equivalent_ids"]:
                                left, right = find(entry["descriptor_id"]), find(other)
                                parents[max(left, right)] = min(left, right)
                        successful += 1
                    except Exception as exc:
                        unresolved.update(affected)
                        errors.append({"stage": "registry_alignment", "descriptor_ids": sorted(affected),
                                       "details": str(exc)})
        canonical = {metric_id: find(descriptor_id) for metric_id, descriptor_id in metric_descriptors.items()
                     if descriptor_id not in unresolved}
        return canonical, descriptors, {key: find(key) for key in descriptors}, comparisons, successful

    @staticmethod
    def _column_name(descriptor: dict) -> str:
        # Keep context visible in the matrix; no numeric conversions are performed.
        parts = [descriptor["variable"]] + [f"{key}_{descriptor[key]}" for key in CONTEXT_FIELDS]
        return "__".join(re.sub(r"[^\w]+", "_", part, flags=re.UNICODE).strip("_").lower() or "field"
                         for part in parts)

    def process(self, raw_extractions_list: list, topic: str) -> dict:
        provenance = self._build_traceability_index(raw_extractions_list)
        errors, mappings, eligible = [], {}, []
        unresolved_inputs = []
        invalid_papers = set()
        papers = []
        metrics_by_paper = defaultdict(list)
        for record in provenance:
            metrics_by_paper[record["paper_id"]].append(record)
        for index, item in enumerate(raw_extractions_list, 1):
            paper_id = f"paper_{index:05d}"
            source = item.get("source", f"paper_{index}") if isinstance(item, dict) else f"paper_{index}"
            data = item.get("data", item) if isinstance(item, dict) else item
            papers.append({"_paper_id": paper_id, "source": source})
            issues = self._untraceable_paths(data)
            if issues:
                invalid_papers.add(paper_id)
                errors.append({"stage": "input", "paper_id": paper_id, "paths": issues,
                               "details": "Input contains fields outside the traceable metric contract."})
                unresolved_inputs.append({"paper_id": paper_id, "source": source,
                                          "data": data, "invalid_paths": issues})
            if not metrics_by_paper[paper_id]:
                errors.append({"stage": "input", "paper_id": paper_id, "details": "No traceable metrics found."})
        for record in provenance:
            try:
                batch_records([record], HARMONIZER_BATCH_CHARS)
                eligible.append(record)
            except ValueError as exc:
                errors.append({"stage": "metric_batching", "metric_ids": [record["metric_id"]], "details": str(exc)})
        batches = batch_records(eligible, HARMONIZER_BATCH_CHARS)
        successful_batches = 0
        for index, batch in enumerate(batches, 1):
            try:
                mappings.update(self._map_batch(batch, topic))
                successful_batches += 1
            except Exception as exc:
                errors.append({"stage": "metric_mapping", "batch": index,
                               "metric_ids": [record["metric_id"] for record in batch], "details": str(exc)})
                logger.error("Harmonization batch %s failed: %s", index, exc)
        try:
            canonical, descriptors, roots, comparisons, aligned = self._canonicalise(mappings, errors)
        except Exception as exc:
            # An oversized descriptor registry must not become an unreported partial result.
            errors.append({"stage": "registry_batching", "details": str(exc)})
            canonical, descriptors, roots, comparisons, aligned = {}, {}, {}, 0, 0
        used_roots = sorted(set(canonical.values()))
        columns, used_names = {}, set()
        for descriptor_id in used_roots:
            name = self._column_name(descriptors[descriptor_id])
            if name in used_names or name in {"source", "_paper_id", "_unresolved_metrics", "_harmonization_status"}:
                name += f"__{descriptor_id}"
            while name in used_names:
                name += f"__{descriptor_id}"
            columns[descriptor_id] = name
            used_names.add(name)
        dataset, covered_ids = [], []
        collisions = 0
        complete_papers = 0
        for paper in papers:
            paper_metrics = metrics_by_paper[paper["_paper_id"]]
            unresolved = [record for record in paper_metrics if record["metric_id"] not in canonical]
            row_complete = bool(paper_metrics) and not unresolved and paper["_paper_id"] not in invalid_papers
            complete_papers += int(row_complete)
            row = {**paper, "_harmonization_status": "complete" if row_complete else "incomplete"}
            row.update({name: "Not Reported" if row_complete else "Unresolved" for name in columns.values()})
            grouped = defaultdict(list)
            for record in paper_metrics:
                if record["metric_id"] in canonical:
                    grouped[columns[canonical[record["metric_id"]]]].append(dict(record))
                    covered_ids.append(record["metric_id"])
            for name, records in grouped.items():
                row[name] = records[0] if len(records) == 1 else {"values": records}
                collisions += int(len(records) > 1)
            if unresolved:
                row["_unresolved_metrics"] = [dict(record) for record in unresolved]
            dataset.append(row)
        covered_set = set(covered_ids)
        unresolved_ids = [record["metric_id"] for record in provenance if record["metric_id"] not in covered_set]
        # Count source records rather than columns, so collisions cannot inflate coverage.
        if len(covered_ids) != len(covered_set):
            raise AssertionError("A source metric was emitted more than once.")
        status = "complete" if not errors and not unresolved_ids else ("incomplete" if covered_ids else "error")
        variables = []
        for descriptor_id in used_roots:
            aliases = [descriptor["variable"] for key, descriptor in descriptors.items()
                       if roots[key] == descriptor_id]
            variables.append({"harmonized_key": columns[descriptor_id], **descriptors[descriptor_id],
                              "aliases": sorted(set(aliases))})
        audit = {
            "status": status,
            "input_papers": len(papers), "complete_papers": complete_papers,
            "input_metrics": len(provenance), "mapped_metrics": len(covered_ids),
            "unresolved_metrics": len(unresolved_ids), "unresolved_metric_ids": unresolved_ids,
            "metric_coverage": len(covered_ids) / len(provenance) if provenance else 0.0,
            "mapping_batches": len(batches), "successful_mapping_batches": successful_batches,
            "registry_comparisons": comparisons, "successful_registry_comparisons": aligned,
            "collision_cells": collisions, "checked_cells": len(covered_ids),
            "traceable_cells": len(covered_ids),
            "traceability_preservation_rate": 1.0 if covered_ids else None,
            "warnings": [error["details"] for error in errors],
        }
        result = {
            "status": status, "harmonized_dataset": dataset,
            "traceability_index": provenance, "harmonization_audit": audit,
            "harmonized_variables": variables, "errors": errors,
            "unresolved_inputs": unresolved_inputs,
            "harmonization_notes": "All values and provenance are copied unchanged from the complete source index. "
                                   "Units are preserved, not converted. Context labels must match before semantic alignment. "
                                   "Unresolved records are retained explicitly; collisions retain all source records. "
                                   "Complete status describes record coverage and successful mapping checks, not clinical validation.",
        }
        if status != "complete":
            result["error"] = "Harmonization incomplete; inspect errors and unresolved records."
        return result
