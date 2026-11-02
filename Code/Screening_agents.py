import time
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from base_agent import BaseAgent, LLM_CALL_TIMEOUT
from evidence_context import DocumentCoverageError, DocumentEvidenceAgent

logger = logging.getLogger(__name__)

class AdvocateAgent(BaseAgent):
    def __init__(self, client: OpenAI, model_name: str):
        system_prompt = """You are an optimistic and rigorous Medical Literature Screening Specialist. Your primary goal is to MAXIMIZE SENSITIVITY (Recall). 
You firmly believe that if a paper shares conceptual overlap with the target, the full text MIGHT contain the required data. Give the paper the "benefit of the doubt."

[CRITICAL INSTRUCTION ON STUDY DESIGN]:
Meta-analyses vary greatly. Some require Randomized Controlled Trials (RCTs), some require Observational Studies (Cohort/Case-Control), and "Umbrella Reviews" specifically require previous Systematic Reviews and Meta-analyses. 
You MUST NOT have any inherent bias against "Reviews" or any specific study type. Your ONLY standard is the user-provided "Study_Design" criteria.

Your task:
1. Cross-reference the paper's text against the Target PICOS criteria.
2. Actively search for ANY hints, synonyms, or implicit clues that suggest it MIGHT meet the criteria, especially regarding the study design.
3. Formulate a strong, academically grounded argument defending WHY this paper should be INCLUDED.
4. Conclude with a "Pro-Inclusion Score" (0-100), evaluating the potential value of reading its full text.

[KEY RULE]: If the paper matches AT LEAST ONE element in the Outcome list, it is a VALID match. 
Do not exclude a paper just because it focuses on a single disease (e.g., only obesity). 
Your mission is to find reasons to keep the paper. If the abstract is unclear, always argue for "Include"."""
        super().__init__(client, model_name, system_prompt)

    def process(self, text: str, picos: dict) -> str:
        prompt = f"[Target PICOS]:\n{json.dumps(picos, indent=2, ensure_ascii=False)}\n\n[Paper Text]:\n{text}\n\nArgue for its inclusion."
        logger.info("AdvocateAgent processing screening evidence")
        content, _, _, _ = self._make_llm_call(prompt)
        return content

class CriticAgent(BaseAgent):
    def __init__(self, client: OpenAI, model_name: str):
        system_prompt = """You are a highly critical, rigorous, and skeptical Medical Literature Auditor. Your goal is to filter out obvious noise.

        [CRITICAL INSTRUCTION ON ABSTRACT LIMITATIONS - READ CAREFULLY]:
        You are evaluating a portion of a paper.
        - A "Fatal Flaw" is a DIRECT CONTRADICTION of the user's criteria (e.g., the criteria asks for Human studies, and the abstract explicitly says "in a rat model").
        - OMISSION IS NOT A FATAL FLAW: If the abstract simply fails to mention a specific outcome or subgroup, DO NOT assume it is absent in the full text. You cannot use "lack of information" as a reason to exclude.

        Your task:
        1. Scrutinize the paper's text against the Target PICOS criteria.
        2. Actively hunt for DEFINITIVE, EXPLICIT reasons to EXCLUDE the paper.
        3. Formulate a sharp argument attacking WHY this paper violates the criteria.
        4. Conclude with a "Pro-Exclusion Score" (0-100)."""
        super().__init__(client, model_name, system_prompt)

    def process(self, text: str, picos: dict) -> str:
        prompt = f"[Target PICOS]:\n{json.dumps(picos, indent=2, ensure_ascii=False)}\n\n[Paper Text]:\n{text}\n\nArgue for its exclusion."
        logger.info("CriticAgent processing screening evidence")
        content, _, _, _ = self._make_llm_call(prompt)
        return content

class JudgeAgent(BaseAgent):
    def __init__(self, client: OpenAI, model_name: str):
        system_prompt = """You are the Senior Chief Editor overseeing a Medical Meta-Analysis. Make the FINAL DECISION ("Include" or "Exclude").

[STRICT BOOLEAN GATING PROTOCOL]:
Evaluate across 5 dimensions: P, I, C, O, S(Yes/No/Unclear).
- 'Yes': The abstract explicitly confirms the criterion.
- 'Unclear': The abstract uses broad terms (e.g., "clinical outcomes" instead of specifying "IPSS score") or omits the detail entirely.
- 'No': The abstract EXPLICITLY CONTRADICTS the criterion.

[DECISION RULE]:
- The "Benefit of the Doubt" Rule: If ANY gate is 'Unclear', you MUST lean towards inclusion. The goal of abstract screening is high recall.
- If ALL gates are 'Yes' or 'Unclear' -> Include (Score >= 85).
- ONLY if a gate is a definitive 'No' (Fatal Flaw) -> Exclude (Score < 85).
Required JSON structure:
{
  "boolean_gates": {"P": "Yes/No/Unclear", "I": "Yes/No/Unclear", "C": "Yes/No/Unclear", "O": "Yes/No/Unclear", "S": "Yes/No/Unclear"},
  "judge_reasoning": "A rigorous 2-sentence explanation.",
  "picos_alignment_score": <Integer between 0-50 for Exclude, or 85-100 for Include>,
  "decision": "Include" or "Exclude",
  "exclusion_reason": "Provide PRISMA reason if Exclude, otherwise 'N/A'"
}"""
        super().__init__(client, model_name, system_prompt)

    def process(self, adv: str, cri: str, picos: dict, evidence_context: str = "") -> dict:
        prompt = f"[Target PICOS]:\n{json.dumps(picos, indent=2, ensure_ascii=False)}\n[Paper evidence]:\n{evidence_context}\n[Advocate]:\n{adv}\n[Critic]:\n{cri}\nFinal Verdict in JSON."
        logger.info(f"JudgeAgent making final decision")
        content, _, _, _ = self._make_llm_call(prompt, json_mode=True)
        return json.loads(content)

class MetadataExtractionAgent(BaseAgent):
    def __init__(self, client: OpenAI, model_name: str):
        system_prompt = "Extract Metadata (Author, Year, Study Design, Population, Intervention, Outcome) strictly in JSON. If any field is not found, fill with 'N/A'."
        super().__init__(client, model_name, system_prompt)
        self.evidence_agent = DocumentEvidenceAgent(client, model_name)

    def process(self, text: str) -> dict:

        context, coverage = self.evidence_agent.process(
            text,
            "Find author, publication year, the actual study design, population, intervention, "
            "comparator, and outcomes for this paper. Preserve conflicts and distinguish cited studies.",
        )
        prompt = f"Extract metadata from the following paper evidence:\n\n{context}"
        logger.info("MetadataExtractionAgent processing whole-document evidence")
        content, _, _, _ = self._make_llm_call(prompt, json_mode=True)
        metadata = json.loads(content)
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a JSON object.")
        metadata["document_coverage"] = coverage
        return metadata

class ScreeningDebateSystem:
    def __init__(self, client: OpenAI, model_name: str):
        self.advocate = AdvocateAgent(client, model_name)
        self.critic = CriticAgent(client, model_name)
        self.judge = JudgeAgent(client, model_name)
        self.metadata_agent = MetadataExtractionAgent(client, model_name)
        self.evidence_agent = DocumentEvidenceAgent(client, model_name)

    def process(self, text: str, picos: dict) -> dict:
        context, coverage = self.evidence_agent.process(
            text,
            "Find all evidence supporting or explicitly contradicting each target PICOS criterion. "
            "Preserve study design, arms, population, outcomes, and eligibility-relevant details. "
            "Do not treat information absent from one segment as absent from the study. Target PICOS: "
            + json.dumps(picos, ensure_ascii=False),
        )
        logger.info("ScreeningDebateSystem processing whole-document evidence")


        with ThreadPoolExecutor(max_workers=1) as executor:

            f_adv = executor.submit(self.advocate.process, context, picos)

            logger.info("Staggering API calls: Waiting 3 seconds before submitting CriticAgent...")
            time.sleep(3)

            f_cri = executor.submit(self.critic.process, context, picos)

            try:
                adv_arg = f_adv.result(timeout=LLM_CALL_TIMEOUT)
                cri_arg = f_cri.result(timeout=LLM_CALL_TIMEOUT)
            except TimeoutError:
                logger.error("Thread execution timeout for Advocate/Critic Agent")
                raise

        verdict = self.judge.process(adv_arg, cri_arg, picos, evidence_context=context)
        verdict["document_coverage"] = coverage

        if verdict.get("decision") == "Include":
            logger.info("Paper PASSED. Extracting structural metadata...")
            try:
                verdict["paper_metadata"] = self.metadata_agent.process(text)
            except DocumentCoverageError:
                raise
            except Exception as e:
                logger.warning(f"Metadata extraction failed: {str(e)}")
                verdict["paper_metadata"] = {"error": "Metadata extraction failed", "details": str(e)}

        return verdict
