# M³: Multi-Agent Medical Meta-Analysis

M³ implements a multi-agent workflow for evidence processing before statistical pooling in medical meta-analysis. It screens studies against a PICOS specification, extracts study data with source locations, performs study-design-aware quality assessment, and aligns variables across included studies while preserving their reported values and provenance.

The implementation accompanies **M³: Multi-agent paradigm for medical evidence synthesis**, by Jiahao Chen, Xu Dong, Ge Li, Boning Li, and Lihong Hu.

## Workflow

1. **PDF parsing:** recover text and tables and append page-level locations of detected tables and raster images.
2. **Study screening:** an Advocate and a Critic assess eligibility, followed by a Judge decision. The current entry point runs one round of these roles, with Advocate and Critic tasks executed sequentially.
3. **Data extraction:** extract topic-relevant fields as structured records containing a value and its source location. Programmatic checks and a model-based Checker provide feedback for bounded retries.
4. **Quality assessment:** select a tool for each paper in `AUTO` mode, or use a specified tool, then generate a structured assessment from the corresponding prompt template.
5. **Data harmonization:** align variables within compatible study-arm, time-point, unit, statistic, and population contexts. Preserve original values, source locations, unresolved records, and conflicting entries.

Long documents are processed in overlapping segments. Screening and quality assessment use a bounded evidence context assembled from those segments; the final calls do not necessarily receive the complete document verbatim.

This repository does not calculate pooled effect estimates, perform statistical meta-analysis, or generate forest plots. Harmonization does not convert units or derive unreported values.

## Files

| File | Purpose |
| --- | --- |
| `Main.py` | Configure and run the pipeline; save results and logs. |
| `base_agent.py` | Shared model API calls, timeouts, and usage logging. |
| `Pdf_parser.py` | Extract PDF text, recoverable tables, and layout locations. |
| `Screening_agents.py` | Advocate, Critic, Judge, and study-metadata extraction. |
| `Distillation_agent.py` | Structured extraction, traceability checks, feedback, and retries. |
| `Quality_assessment_agent.py` | Per-paper tool selection and quality assessment. |
| `qa_prompts.py` | Quality-assessment prompt templates and output structures. |
| `DataHarmonizer_agent.py` | Context-aware variable alignment and provenance-preserving output. |
| `document_utils.py` | Overlapping text segments and complete-record JSON batches. |
| `evidence_context.py` | Source-anchored evidence collection and context reduction. |
| `pipeline_results.py` | Separate validated extraction data from failed or incomplete results. |
| `requirements.txt` | Dependency versions supplied with this distribution. |

Keep these Python files together. Imports use the filenames shown above.

## Installation

Use Python with `pip` and `venv`. The source distribution does not specify a validated Python version; choose an interpreter compatible with the listed dependencies and record its version when reporting experiments.

From the directory containing `Main.py`, create an isolated environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

**Dependency preparation:** the supplied `requirements.txt` currently contains the incomplete entry `packaging @ file:`. Replace that line with `packaging`, or with a verified version pin from your experiment environment, before running the installation command. The remaining pins have not been independently validated as a complete cross-platform environment.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The source directly uses `openai`, `python-dotenv`, `pymupdf4llm`, and `PyMuPDF`. The `fitz` import is provided by `PyMuPDF`.

## Configuration

### Model access

Create a local `.env` file alongside `Main.py`:

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_MODEL_NAME=your_model_identifier
```

For a compatible alternative endpoint, also set:

```dotenv
OPENAI_BASE_URL=https://your-provider.example/v1
```

Replace the example values with your own configuration. Omit `OPENAI_BASE_URL` when using the SDK's default endpoint. Do not upload a populated `.env` file or include credentials in logs shared publicly.

The endpoint must support Chat Completions with system/user messages, the `temperature` parameter, and `response_format={"type": "json_object"}`. Responses must contain text and token-usage information. Model availability, API charges, and endpoint behavior depend on the provider. The configured request timeout is 180 seconds.

### Study protocol and paths

Edit the configuration block in `Main.py` before running:

| Setting | What to configure |
| --- | --- |
| `RESEARCH_TOPIC` | The review question used for extraction and harmonization. |
| `PICOS_CRITERIA` | Population, intervention/exposure, comparison, outcomes, and study designs. |
| `INPUT_PDF_DIR` | Directory containing the input PDFs. |
| `OUTPUT_DIR` | A separate output directory for this run. |
| `TARGET_QA_TOOL` | `"AUTO"` for per-paper selection, or an exact tool key below. |

For example, replace the two `YOUR_Path_*` values with:

```python
INPUT_PDF_DIR = Path("data/pdfs")
OUTPUT_DIR = Path("outputs/run_001")
```

Relative paths are resolved from the working directory. Create `data/pdfs` and place the input papers there. The pipeline creates the output directory and `raw_md` subdirectory.

The bundled research question concerns blood-pressure control. Adapt the PICOS criteria to the actual review protocol, including distinct intervention and comparison definitions. The example criteria are not a universal eligibility standard.

### Quality-assessment tools

| Configuration key | Prompt-template scope |
| --- | --- |
| `AUTO` | Select a tool separately for each paper. |
| `NOS_Cohort` | Newcastle–Ottawa cohort assessment template. |
| `RoB2_RCT` | Risk of Bias 2 template for randomized trials. |
| `QUADAS_2_Diagnostic` | Diagnostic-accuracy assessment template. |
| `AMSTAR_2` | Systematic-review appraisal template. |
| `ROBINS_I` | Non-randomized intervention-study template. |
| `QUIPS` | Prognostic-study assessment template. |
| `JBI_Prevalence` | Prevalence-study appraisal template. |

The implemented questions and output fields are defined in `qa_prompts.py`. These are prompt-based templates, not complete validated implementations of every official instrument. In particular, `NOS_Cohort` contains cohort items rather than a separate case-control instrument. Inspect the selected tool and assessment fields before interpreting a result. Inspect the selection rationale and confidence when present; heuristic fallback decisions are explicitly marked and require particular care.

## Run

From the directory containing `Main.py`, with the environment activated and the configuration completed:

```bash
python Main.py
```

The entry point reads `*.pdf` files directly inside `INPUT_PDF_DIR`; it does not recursively scan subdirectories. Start with a small set of text-readable PDFs and inspect the parsed Markdown and results before processing a larger collection.

Each execution starts with empty in-memory result lists and overwrites the corresponding JSON output files as processing proceeds. Automatic resume is not implemented. Use a new `OUTPUT_DIR` for each experiment to preserve previous results. The log file is appended to when an existing output directory is reused.

## Outputs

| Output | Contents |
| --- | --- |
| `raw_md/<paper>.md` | Parsed paper text and generated layout annotations. |
| `MASTER_EXTRACTION_RESULTS.json` | Included-paper extraction records, screening dossiers, validation information, and processing errors. |
| `QUALITY_ASSESSMENT_RESULTS.json` | Tool selections, assessment objects, coverage information, or errors. |
| `HARMONIZED_DATASET.json` | Aligned data, provenance, unresolved records, mapping audits, and pipeline status. |
| `pipeline.log` | Progress, API usage and timing, exclusion reasons, and errors. |

An extraction record contains `source`, `screening_dossier`, `distilled_data`, and `extraction_validation`. Each terminal extracted metric uses `value` and `source_location`. Failed or incompletely validated extractions are retained as diagnostic records and excluded from harmonization input.

When harmonization is performed, the output includes `harmonized_dataset`, `harmonized_variables`, `traceability_index`, `harmonization_audit`, `errors`, and `unresolved_inputs`. The entry point also adds `input_extraction_audit`, `qa_execution_audit`, and `pipeline_status`. Skipped stages may return a smaller status or error object.

Inspect validation and error fields as well as the data. `accepted`, `complete`, coverage fractions, and provenance-preservation rates describe internal processing checks; they are not independent estimates of extraction accuracy or clinical correctness. Excluded studies are primarily recorded in `pipeline.log`, so the extraction JSON is not a complete screening ledger for every input paper.

## Practical limitations and reproducibility

- PDF parsing depends on recoverable text and tables. Raster-image locations are recorded, but image contents are not interpreted by this implementation and no explicit OCR workflow is configured. Inspect `raw_md` for empty, scanned, or poorly extracted papers.
- The programmatic value-support check uses approximate matching. Numeric substrings and partially matching compound values can pass this check; the separate model-based Checker does not establish guaranteed correctness.
- Screening and quality-assessment outputs need inspection for missing fields and invalid decisions. The current code does not fully validate all decision and assessment schemas.
- Context reduction preserves selected source quotations and segment coverage, not necessarily every detail in the original paper. Harmonization preserves reported units and values instead of performing statistical transformations.
- Record the exact model identifier, provider endpoint, dependency versions, protocol configuration, input-document versions, and output directory for each experiment. There is no globally configured random seed, and results can vary across model versions and runs.
- This directory supplies the pipeline implementation and prompts. It does not include the complete benchmark datasets, reference annotations, or evaluation scripts needed to reproduce every result in the manuscript.

Before sharing the repository, exclude `.env`, local virtual environments, caches, and any documents or data that are not intended for redistribution. A license file is not included in this distribution; reuse permissions should be clarified with the authors.
