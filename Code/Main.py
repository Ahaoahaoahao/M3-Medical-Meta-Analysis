import os
import json
import logging
import traceback
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from Pdf_parser import parse_pdf_to_markdown
from DataHarmonizer_agent import DataHarmonizerAgent
from pipeline_results import extraction_record, harmonization_inputs
from Screening_agents import ScreeningDebateSystem
from Distillation_agent import ContentDistillationAgent
from Quality_assessment_agent import QualityAssessmentAgent

# Configuration
RESEARCH_TOPIC = "Effect of intensive versus standard blood pressure control on cardiovascular outcomes."
TARGET_QA_TOOL = "AUTO"  # set to a concrete tool key to override automatic QA routing
PICOS_CRITERIA = {
"Population": "Hypertensive patients or potential hypertensive populations, or other populations associated with hypertension, or blood pressure control related to normal individuals, orpharmacological trial related to blood pressure.",
"Intervention/Exposure": "Blood pressure control, defined per trial protocol as aiming for a systolic/diastolic target.",
"Comparison": "Blood pressure control, defined per trial protocol as aiming for a systolic/diastolic target.",
"Outcome": "Reporting at least one major cardiovascular event (MACE), myocardial infarction (MI), stroke, all-cause mortality, or cardiac death and other possible related events.",
"Study_Design": "Include Randomized Controlled Trials (RCTs), observational studies. EXCLUDE: , retrospective cohorts, case reports, systematic reviews, and meta-analyses."
}
# Paths
INPUT_PDF_DIR = Path(r"YOUR_Path_INPUT_PDF")
OUTPUT_DIR = Path(r"YOUR_Path_OUTPUT_DISTILLED")
RAW_MD_DIR = OUTPUT_DIR / "raw_md"
FINAL_OUTPUT_FILE = OUTPUT_DIR / "MASTER_EXTRACTION_RESULTS.json"
FINAL_QA_FILE = OUTPUT_DIR / "QUALITY_ASSESSMENT_RESULTS.json"
FINAL_HARMONIZED_FILE = OUTPUT_DIR / "HARMONIZED_DATASET.json"
LOG_FILE = OUTPUT_DIR / "pipeline.log"
ENCODING = "utf-8"

def setup_env():
    load_dotenv()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RAW_MD_DIR, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(LOG_FILE, encoding=ENCODING), logging.StreamHandler()]
    )

def save_json_file(data, file_path):
    with open(file_path, 'w', encoding=ENCODING) as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    setup_env()
    logger = logging.getLogger(__name__)

    pdf_files = list(INPUT_PDF_DIR.glob('*.pdf'))
    if not pdf_files:
        logger.warning("No PDF files found.")
        exit(0)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
    model = os.getenv("OPENAI_MODEL_NAME")

    # Agent initialization
    debate_system = ScreeningDebateSystem(client, model)
    distillation_agent = ContentDistillationAgent(client, model)
    qa_agent = QualityAssessmentAgent(client, model, qa_tool=None if TARGET_QA_TOOL == "AUTO" else TARGET_QA_TOOL)
    harmonizer_agent = DataHarmonizerAgent(client, model)

    # Processing
    all_final_results = []
    all_qa_results = []
    included_count = 0

    for idx, pdf_path in enumerate(pdf_files, 1):
        pdf_name = pdf_path.name
        logger.info("\n" + "=" * 60)
        logger.info(f"Processing Paper [{idx}/{len(pdf_files)}]: {pdf_name}")

        try:
            md_text = parse_pdf_to_markdown(str(pdf_path))
            md_output_path = RAW_MD_DIR / f"{pdf_path.stem}.md"
            with open(md_output_path, "w", encoding=ENCODING) as f:
                f.write(md_text)

            screening_result = debate_system.process(md_text, PICOS_CRITERIA)

            if screening_result.get("decision") == "Include":
                logger.info("Paper included; starting data extraction and quality assessment.")
                included_count += 1

                try:
                    extraction_result = distillation_agent.process(md_text, RESEARCH_TOPIC)
                except Exception as exc:
                    extraction_result = {
                        "error": "Content distillation failed", "details": str(exc),
                        "validation_status": "incomplete", "coverage": {"complete": False},
                    }
                all_final_results.append(extraction_record(pdf_name, screening_result, extraction_result))
                save_json_file(all_final_results, FINAL_OUTPUT_FILE)

                try:
                    qa_result = qa_agent.process(md_text, metadata=screening_result.get("paper_metadata"))
                except Exception as exc:
                    qa_result = {"error": "Quality assessment failed", "details": str(exc)}

                selected_tool = qa_result.get("selected_qa_tool", "N/A")
                logger.info(f"QA completed using [{selected_tool}]: {qa_result.get('summary', 'N/A')}")

                all_qa_results.append({
                    **qa_result,
                    "source": pdf_name,
                    "qa_tool": qa_result.get("selected_qa_tool", "N/A"),
                    "qa_selection": qa_result.get("qa_selection"),
                    "quality_assessment": qa_result.get("quality_assessment", qa_result)
                })
            else:
                logger.info(f"EXCLUDED. Reason: {screening_result.get('exclusion_reason')}")

            save_json_file(all_final_results, FINAL_OUTPUT_FILE)
            save_json_file(all_qa_results, FINAL_QA_FILE)

        except Exception as e:
            logger.error(f"Error processing {pdf_name}: {str(e)}")
            error_dict = {"source": pdf_name, "error": "Processing failed", "details": str(e)}
            if hasattr(e, "coverage"):
                error_dict["document_coverage"] = e.coverage
            all_final_results.append(error_dict)
            all_qa_results.append(error_dict)
            save_json_file(all_final_results, FINAL_OUTPUT_FILE)
            save_json_file(all_qa_results, FINAL_QA_FILE)

    # Harmonization
    if included_count > 0:
        logger.info("\n" + "*" * 60)
        logger.info("Starting Data Harmonization for all included papers...")

        raw_extractions_to_harmonize, excluded_extractions = harmonization_inputs(all_final_results)
        if raw_extractions_to_harmonize:
            harmonized_result = harmonizer_agent.process(raw_extractions_to_harmonize, RESEARCH_TOPIC)
        else:
            harmonized_result = {
                "status": "not_run",
                "harmonized_dataset": [],
                "details": "No validated non-empty extraction was available.",
            }
        harmonized_result["input_extraction_audit"] = {
            "included_papers": included_count,
            "validated_nonempty_papers": len(raw_extractions_to_harmonize),
            "excluded_extractions": excluded_extractions,
        }
        failed_qa_sources = [
            record.get("source") for record in all_qa_results
            if "error" in record or (
                isinstance(record.get("quality_assessment"), dict)
                and "error" in record["quality_assessment"]
            )
        ]
        harmonized_result["qa_execution_audit"] = {"failed_sources": failed_qa_sources}
        harmonized_result["pipeline_status"] = (
            "incomplete" if any(item["reason"] == "extraction_not_validated" for item in excluded_extractions)
            or "error" in harmonized_result or harmonized_result.get("status") == "incomplete"
            or failed_qa_sources
            else harmonized_result.get("status", "complete")
        )
        save_json_file(harmonized_result, FINAL_HARMONIZED_FILE)

        logger.info("Data harmonization status: %s. Output saved to: %s",
                    harmonized_result["pipeline_status"], FINAL_HARMONIZED_FILE)
    else:
        logger.warning("No papers were included, skipping Data Harmonization.")
        save_json_file({
            "status": "not_run", "harmonized_dataset": [],
            "pipeline_status": "incomplete" if all_final_results else "not_run",
            "details": "No included papers were available for harmonization.",
        }, FINAL_HARMONIZED_FILE)

    logger.info("\n" + "=" * 60)
    logger.info(f"Pipeline finished. Included: {included_count}/{len(pdf_files)}")
