import json
import logging
import re
from typing import Any, Dict, Optional

from openai import OpenAI
from base_agent import BaseAgent
from qa_prompts import QA_PROMPTS_DICT
from evidence_context import DocumentCoverageError, DocumentEvidenceAgent

logger = logging.getLogger(__name__)

QA_TOOL_DESCRIPTIONS = {
    "NOS_Cohort": "Observational cohort or case-control study; Newcastle-Ottawa Scale.",
    "RoB2_RCT": "Randomized controlled trial; Cochrane Risk of Bias 2.0.",
    "QUADAS_2_Diagnostic": "Diagnostic accuracy study with index test and reference standard; QUADAS-2.",
    "AMSTAR_2": "Systematic review, meta-analysis, or umbrella review; AMSTAR 2.",
    "ROBINS_I": "Non-randomized study of interventions; ROBINS-I.",
    "QUIPS": "Prognostic factor or prediction study, often reporting survival/risk prediction; QUIPS.",
    "JBI_Prevalence": "Cross-sectional or prevalence study; JBI Critical Appraisal Tool for Prevalence Studies.",
}


def _safe_json_loads(content: str) -> Dict[str, Any]:
    """Load model JSON robustly when the model wraps the object in code fences."""
    cleaned = content.replace("```json", "").replace("```", "").strip()
    result = json.loads(cleaned)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object for QA selection or assessment.")
    return result


def _normalise_qa_tool(qa_tool: Optional[str]) -> Optional[str]:
    """None/AUTO requests per-paper selection; a supported key fixes the tool."""
    if qa_tool is None or (isinstance(qa_tool, str) and qa_tool.strip().upper() == "AUTO"):
        return None
    if qa_tool not in QA_PROMPTS_DICT:
        raise ValueError(f"Unsupported QA Tool: '{qa_tool}'. Available tools: {list(QA_PROMPTS_DICT.keys())}")
    return qa_tool


class QAToolSelectorAgent(BaseAgent):
    """Selects the appropriate risk-of-bias / quality-assessment tool from the paper itself."""

    def __init__(self, client: OpenAI, model_name: str):
        tools = "\n".join([f"- {k}: {v}" for k, v in QA_TOOL_DESCRIPTIONS.items()])
        system_prompt = f"""You are an evidence-based medicine methodologist.
Your task is to select exactly one quality-assessment or risk-of-bias tool for a medical primary study or review.

Available tools:
{tools}

Selection principles:
1. Prefer the study design explicitly stated in the paper over the user's broad PICOS description.
2. If the paper is an RCT, choose RoB2_RCT.
3. If the paper is an observational cohort/case-control study without an intervention assignment, choose NOS_Cohort.
4. If the paper evaluates an index diagnostic test against a reference standard, choose QUADAS_2_Diagnostic.
5. If the paper itself is a systematic review/meta-analysis/umbrella review, choose AMSTAR_2.
6. If the paper is a non-randomized intervention study, choose ROBINS_I.
7. If the paper studies prognostic factors, prediction, or survival risk, choose QUIPS.
8. If the paper is cross-sectional/prevalence-oriented, choose JBI_Prevalence.
9. If evidence is insufficient, select the most defensible tool and set confidence below 0.6.

Return ONLY valid JSON with this schema:
{{
  "selected_tool": "one key from the available tools",
  "detected_study_design": "concise study-design label",
  "evidence_quote": "short exact quote from the paper supporting the design, or N/A",
  "confidence": 0.0,
  "rationale": "brief rationale, not a hidden chain-of-thought"
}}"""
        super().__init__(client, model_name, system_prompt)
        self.evidence_agent = DocumentEvidenceAgent(client, model_name)

    def _heuristic_select(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # These low-confidence fallbacks are used only after selector failure.
        # Match whole design terms: "roc" inside "process" is not diagnostic
        # evidence, and "non-randomized" must not match the randomized rule.
        rules = [
            ("AMSTAR_2", [r"\bsystematic\s+review\b", r"\bmeta[-\s]?analys[ie]s\b", r"\bumbrella\s+review\b"]),
            ("QUADAS_2_Diagnostic", [r"\bdiagnostic\s+(?:test\s+)?accuracy\b", r"\bindex\s+test\b", r"\breference\s+standard\b"]),
            ("ROBINS_I", [r"\bnon[-\s]?randomi[sz]ed\b", r"\bquasi[-\s]experimental\b"]),
            ("RoB2_RCT", [r"\brandomi[sz]ed\b", r"\brandom\s+allocation\b", r"\brandomly\s+assigned\b", r"\bdouble[-\s]blind\b", r"\bplacebo[-\s]controlled\b"]),
            # Repeated measurements before/after treatment do not establish
            # a non-randomized design, especially when randomization is explicit.
            ("ROBINS_I", [r"\bbefore[-\s](?:and[-\s])?after\s+(?:study|studies|design)\b"]),
            ("QUIPS", [r"\bprognostic\b", r"\bprediction\s+model\b"]),
            ("JBI_Prevalence", [r"\bcross[-\s]sectional\b", r"\bprevalence\b", r"\bsurvey\b"]),
            ("NOS_Cohort", [r"\bcohort\b", r"\bcase[-\s]control\b", r"\bobservational\b", r"\bretrospective\b", r"\bprospective\b"]),
        ]
        # Prefer this paper's text; metadata is only a secondary source of cues.
        for cue_source, cue_text in (
            ("paper text", text),
            ("screening metadata", json.dumps(metadata or {}, ensure_ascii=False)),
        ):
            normalised = re.sub(r"[\u2010-\u2015\u2212]", "-", cue_text).lower()
            for tool, patterns in rules:
                if any(re.search(pattern, normalised) for pattern in patterns):
                    return {
                        "selected_tool": tool,
                        "detected_study_design": "heuristically inferred",
                        "evidence_quote": "N/A",
                        "confidence": 0.55,
                        "rationale": f"Fallback heuristic matched design terms in {cue_source} for {tool} after selector failure.",
                    }
        return {
            "selected_tool": "NOS_Cohort",
            "detected_study_design": "unclear observational/default",
            "evidence_quote": "N/A",
            "confidence": 0.30,
            "rationale": "Fallback default used because no reliable study-design signal was found.",
        }

    def process(self, markdown_text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context, coverage = self.evidence_agent.process(
            markdown_text,
            "Find the design of this paper, allocation methods, whether interventions were assigned, "
            "whether the study is a review, diagnostic-accuracy evaluation, prognostic study, "
            "or prevalence survey. Distinguish its actual methods from cited studies. Preserve "
            "explicit methodological statements and conflicts. Available QA tools: "
            + json.dumps(QA_TOOL_DESCRIPTIONS, ensure_ascii=False),
        )
        prompt = f"""[Metadata from screening stage]
{json.dumps(metadata or {}, ensure_ascii=False, indent=2)}

[Paper evidence]
{context}

Select the appropriate QA tool."""
        try:
            content, _, _, _ = self._make_llm_call(prompt, temperature=0, json_mode=True)
            selection = _safe_json_loads(content)
            selected_tool = selection.get("selected_tool")
            if selected_tool not in QA_PROMPTS_DICT:
                raise ValueError(f"Selector returned unsupported tool: {selected_tool}")
            selection["document_coverage"] = coverage
            return selection
        except Exception as exc:
            if coverage["mode"] == "evidence_reduction":
                coverage["status"] = "failed"
                coverage["failure"] = str(exc)
                raise DocumentCoverageError(f"Whole-document QA tool selection failed: {exc}", coverage) from exc
            logger.warning("QA tool selector failed; falling back to heuristic selector: %s", exc)
            selection = self._heuristic_select(markdown_text, metadata)
            selection["document_coverage"] = coverage
            selection["selection_method"] = "heuristic_fallback"
            return selection


class QualityAssessmentAgent(BaseAgent):
    """Risk-of-bias / quality-assessment agent with optional automatic tool routing."""

    def __init__(self, client: OpenAI, model_name: str, qa_tool: Optional[str] = None):
        # Configured tool; None enables per-paper selection.
        self.qa_tool = _normalise_qa_tool(qa_tool)
        self.selector = QAToolSelectorAgent(client, model_name)

        system_prompt = QA_PROMPTS_DICT[self.qa_tool] if self.qa_tool else "You are an evidence-based medicine quality-assessment agent."
        super().__init__(client, model_name, system_prompt)
        self.evidence_agent = DocumentEvidenceAgent(client, model_name)

    def _summarize(self, qa_tool: str, qa_result: Dict[str, Any]) -> str:
        if qa_tool == "NOS_Cohort":
            return f"NOS score: {qa_result.get('total_nos_score', 'N/A')}/9; category: {qa_result.get('quality_category', 'N/A')}"
        if qa_tool == "RoB2_RCT":
            return f"RoB 2.0 overall risk: {qa_result.get('overall_risk_of_bias', 'N/A')}"
        if qa_tool == "QUADAS_2_Diagnostic":
            return f"QUADAS-2 conclusion: {qa_result.get('overall_quadas_conclusion', 'N/A')}"
        if qa_tool == "AMSTAR_2":
            return f"AMSTAR 2 confidence: {qa_result.get('overall_confidence_rating', 'N/A')}"
        if qa_tool == "ROBINS_I":
            return f"ROBINS-I overall risk: {qa_result.get('overall_risk_of_bias', 'N/A')}"
        if qa_tool == "QUIPS":
            return f"QUIPS overall risk: {qa_result.get('overall_quips_risk', 'N/A')}"
        if qa_tool == "JBI_Prevalence":
            return f"JBI appraisal: {qa_result.get('overall_appraisal', 'N/A')}"
        return "QA completed."

    def process(
        self,
        markdown_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        qa_tool: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Select for each paper in AUTO mode; per-call overrides are temporary."""
        selection = None
        requested_tool = self.qa_tool if qa_tool is None else qa_tool
        selected_tool = _normalise_qa_tool(requested_tool)
        if selected_tool is None:
            try:
                selection = self.selector.process(markdown_text, metadata=metadata)
            except DocumentCoverageError as exc:
                return {
                    "selected_qa_tool": None,
                    "error": "QA tool selection failed",
                    "details": str(exc),
                    "document_coverage": {"selection": exc.coverage},
                }
            selected_tool = selection["selected_tool"]
        else:
            selection = {
                "selected_tool": selected_tool,
                "detected_study_design": (metadata or {}).get("Study Design", "user-specified"),
                "evidence_quote": "N/A",
                "confidence": 1.0,
                "rationale": "QA tool was specified by the user or pipeline configuration.",
            }

        self.system_prompt = QA_PROMPTS_DICT[selected_tool]
        coverage = None
        logger.info("Running whole-document Quality Assessment using [%s]", selected_tool)

        try:
            context, coverage = self.evidence_agent.process(
                markdown_text,
                f"Collect evidence for every item/domain of {selected_tool}. Find study methods, "
                "allocation, recruitment, comparability, blinding, outcome ascertainment, missing "
                "data, follow-up, selective reporting, and any other criteria below. Preserve "
                "contradictions and exact source quotes; do not assign scores. Criteria reference:\n"
                + QA_PROMPTS_DICT[selected_tool],
            )
            prompt = f"""Please perform a rigorous quality assessment using the {selected_tool} criteria.

[Selected QA Tool]
{json.dumps(selection, ensure_ascii=False, indent=2)}

[Paper evidence]
{context}

[CORE RULES]
1. Base every scored item/domain on explicit evidence from the paper text above.
2. Never hallucinate data, quotes, or page/paragraph locations.
3. For EVERY scored item/domain, extract a short exact quote supporting the judgment; if no relevant text is found, write "N/A" and mark the item as Unclear/0/Some concerns/No Information according to the tool rules.
4. Provide concise, audit-style rationales that directly connect the evidence to the judgment.
5. Return ONLY valid JSON following the schema required by the selected tool prompt.
"""
            content, _, _, _ = self._make_llm_call(prompt, temperature=0, json_mode=True)
            qa_result = _safe_json_loads(content)
            logger.info("Quality Assessment completed for [%s]", selected_tool)
            return {
                "selected_qa_tool": selected_tool,
                "qa_selection": selection,
                "quality_assessment": qa_result,
                "summary": self._summarize(selected_tool, qa_result),
                "document_coverage": {
                    "selection": selection.get("document_coverage", {"mode": "user_specified"}),
                    "assessment": coverage,
                },
            }
        except json.JSONDecodeError:
            logger.error("JSON decode failed for %s quality assessment", selected_tool)
            return {
                "selected_qa_tool": selected_tool,
                "qa_selection": selection,
                "error": "QA result JSON decode failed",
                "details": "Invalid JSON response from LLM",
                "document_coverage": {
                    "selection": selection.get("document_coverage", {"mode": "user_specified"}),
                    "assessment": coverage,
                },
            }
        except Exception as e:
            logger.error("Quality Assessment failed for [%s]: %s", selected_tool, str(e), exc_info=True)
            return {
                "selected_qa_tool": selected_tool,
                "qa_selection": selection,
                "error": "Quality Assessment failed",
                "details": str(e),
                "document_coverage": {
                    "selection": selection.get("document_coverage", {"mode": "user_specified"}),
                    "assessment": e.coverage if isinstance(e, DocumentCoverageError) else coverage,
                },
            }
