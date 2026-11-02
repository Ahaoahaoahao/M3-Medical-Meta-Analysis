"""Validation boundaries for saved extraction and harmonization inputs."""

from copy import deepcopy


def extraction_record(source, screening_result, extraction_result):
    if not isinstance(extraction_result, dict):
        extraction_result = {
            "error": "Extraction returned a non-object result",
            "validation_status": "incomplete",
        }
    validation = {key: deepcopy(value) for key, value in extraction_result.items() if key != "data"}
    data = extraction_result.get("data")
    coverage = extraction_result.get("coverage", {})
    accepted = (
        extraction_result.get("validation_status") == "accepted"
        and "error" not in extraction_result
        and isinstance(data, dict)
        and "error" not in data
        and isinstance(coverage, dict)
        and coverage.get("complete") is True
    )
    if not accepted:
        if validation.get("validation_status") not in {"rejected", "incomplete"}:
            validation["validation_status"] = "incomplete"
        validation.setdefault("error", "Extraction was not fully validated")
        if data is not None:
            validation["candidate_data"] = deepcopy(data)
        data = {"error": validation["error"], "validation_status": validation["validation_status"]}
    return {
        "source": source,
        "screening_dossier": screening_result,
        "distilled_data": data,
        "extraction_validation": validation,
    }


def harmonization_inputs(results):
    accepted = []
    excluded = []
    for record in results:
        source = record.get("source")
        data = record.get("distilled_data")
        validation = record.get("extraction_validation", {})
        coverage = validation.get("coverage", {}) if isinstance(validation, dict) else {}
        valid = (
            "error" not in record
            and isinstance(validation, dict)
            and validation.get("validation_status") == "accepted"
            and "error" not in validation
            and isinstance(coverage, dict)
            and coverage.get("complete") is True
            and isinstance(data, dict)
            and "error" not in data
        )
        if valid and data:
            accepted.append({"source": source, "data": data})
        else:
            excluded.append({
                "source": source,
                "reason": "no_relevant_data" if valid else "extraction_not_validated",
            })
    return accepted, excluded
