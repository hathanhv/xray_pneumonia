import unittest
import tempfile
from pathlib import Path

from src.reporting.clinical_schema import (
    build_clinical_report,
    validate_clinical_report,
)
from src.reporting.clinical_report import build_report_facts, render_template_report
from src.reporting.llm_report import parse_llm_response
from src.reporting.report_storage import (
    prepare_report_bundle,
    save_report_bundle,
    save_report_revision,
)
from src.reporting.report_render import render_report_html, export_report_pdf


class ClinicalSchemaTests(unittest.TestCase):
    def test_builds_pneumonia_focused_report_without_mutating_ai_result(self):
        source = {
            "study": {"study_id": "case_001"},
            "classification": {"prediction": "PNEUMONIA", "confidence": 0.93},
            "findings": [
                {"finding": "lung_opacity", "status": "present", "probability": 0.9}
            ],
            "cxformer_multilabel": {
                "findings": [
                    {"finding": "Pleural effusion", "status": "present", "probability": 0.8},
                    {"finding": "Cardiomegaly", "status": "present", "probability": 0.7},
                ]
            },
            "provenance": {"classification_model": "classifier-v1"},
        }

        report = build_clinical_report(source)

        self.assertEqual(report["schema_version"], "1.2")
        self.assertEqual(report["primary_diagnosis"]["status"], "suspected")
        self.assertEqual(len(report["pneumonia_evidence"]["findings"]), 2)
        self.assertEqual(len(report["additional_findings"]), 1)
        self.assertNotIn("schema_version", source)
        self.assertIn("suspicious for pneumonia", report["report_draft"]["impression_text"])
        self.assertIn("lung opacity", report["report_draft"]["findings_text"].lower())
        validate_clinical_report(report)

    def test_normal_prediction_is_unlikely(self):
        report = build_clinical_report(
            {
                "classification": {"prediction": "NORMAL", "confidence": 0.91},
                "findings": [],
                "cxformer_multilabel": {"findings": []},
            }
        )
        self.assertEqual(report["primary_diagnosis"]["status"], "unlikely")

    def test_template_does_not_claim_localization_when_missing(self):
        facts = build_report_facts(
            {
                "primary_diagnosis": {
                    "status": "indeterminate",
                    "probability": None,
                },
                "pneumonia_evidence": {
                    "findings": [
                        {"finding": "consolidation", "status": "present"}
                    ]
                },
                "additional_findings": [],
            }
        )
        draft = render_template_report(facts)
        self.assertIn("consolidation is detected", draft["findings_text"].lower())
        self.assertIn("not available", draft["impression_text"])

    def test_llm_draft_is_used_only_when_contract_is_valid(self):
        source = {
            "classification": {"prediction": "PNEUMONIA", "confidence": 0.93},
            "findings": [],
            "cxformer_multilabel": {"findings": []},
        }

        def client(_prompt):
            return '{"findings_text":"Opacity noted.","impression_text":"Pneumonia suspected.","recommendation_text":"Review required.","limitations_text":"AI estimate."}'

        report = build_clinical_report(source, llm_client=client)

        self.assertEqual(report["report_draft"]["generated_by"], "llm")
        self.assertEqual(report["report_draft"]["prompt_version"], "pneumonia-report-v1")

    def test_invalid_llm_output_falls_back_to_template(self):
        source = {
            "classification": {"prediction": "PNEUMONIA", "confidence": 0.93},
            "findings": [],
            "cxformer_multilabel": {"findings": []},
        }

        report = build_clinical_report(source, llm_client=lambda _prompt: "not-json")

        self.assertEqual(report["report_draft"]["generated_by"], "template-fallback")
        self.assertIn("pneumonia", report["report_draft"]["impression_text"].lower())

    def test_llm_cannot_introduce_unsupported_numbers_or_fields(self):
        facts = {
            "primary_diagnosis": {"probability": 0.93},
            "pneumonia_findings": [],
            "additional_findings": [],
        }
        with self.assertRaises(ValueError):
            parse_llm_response(
                '{"findings_text":"Burden 99%.","impression_text":"Pneumonia.","recommendation_text":"","limitations_text":"","extra":"x"}',
                facts,
            )

    def test_llm_can_format_existing_probability_as_percent(self):
        facts = {
            "primary_diagnosis": {"probability": 0.93},
            "pneumonia_findings": [],
            "additional_findings": [],
        }
        draft = parse_llm_response(
            '{"findings_text":"","impression_text":"Probability 93%.","recommendation_text":"","limitations_text":""}',
            facts,
        )
        self.assertIn("93%", draft["impression_text"])

    def test_report_bundle_keeps_ai_result_immutable_and_versions_revisions(self):
        report = build_clinical_report(
            {
                "classification": {"prediction": "PNEUMONIA", "confidence": 0.93},
                "findings": [],
                "cxformer_multilabel": {"findings": []},
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            report_path = save_report_bundle(
                report,
                output_root=directory,
                study_id="case/001",
            )
            stored = Path(report_path)
            self.assertTrue((stored.parent / "ai_result.json").is_file())
            report["report_draft"]["impression_text"] = "Edited by physician."
            save_report_revision(
                report,
                report_path=stored,
                report_status="in_review",
            )
            revised = __import__("json").loads(stored.read_text(encoding="utf-8"))
            self.assertEqual(revised["metadata"]["revision_number"], 2)
            self.assertEqual(revised["metadata"]["report_status"], "in_review")
            self.assertTrue((stored.parent / "ai_result.json").is_file())

    def test_html_renderer_uses_draft_until_approval(self):
        report = build_clinical_report(
            {
                "study": {"study_id": "case_001"},
                "classification": {"prediction": "PNEUMONIA", "confidence": 0.93},
                "findings": [],
                "cxformer_multilabel": {"findings": []},
            }
        )
        html = render_report_html(report)
        self.assertIn("Chest X-ray Report", html)
        self.assertIn("suspicious for pneumonia", html)
        self.assertIn("DRAFT", html)

    def test_ctr_measurement_is_rendered_in_draft_and_html(self):
        report = build_clinical_report(
            {
                "study": {"study_id": "case_001"},
                "classification": {"prediction": "PNEUMONIA", "confidence": 0.93},
                "findings": [],
                "cxformer_multilabel": {"findings": []},
                "measurements": {
                    "ctr": {
                        "value": 0.5123,
                        "method": "cardiac_to_internal_thoracic_width",
                        "valid": True,
                    }
                },
            }
        )
        html = render_report_html(report)
        self.assertIn("Cardiothoracic ratio (CTR) is 0.512", report["report_draft"]["findings_text"])
        self.assertIn("Anatomy Measurements", html)
        self.assertIn("0.512", html)

    def test_html_renderer_includes_raw_study_image(self):
        report = build_clinical_report(
            {
                "study": {"study_id": "case_001"},
                "classification": {"prediction": "NORMAL", "confidence": 0.91},
                "findings": [],
                "cxformer_multilabel": {"findings": []},
            }
        )
        html = render_report_html(report, image_uri="raw_image.png")
        self.assertIn('src="raw_image.png"', html)

    def test_pdf_export_requires_approval(self):
        report = build_clinical_report(
            {
                "classification": {"prediction": "NORMAL", "confidence": 0.91},
                "findings": [],
                "cxformer_multilabel": {"findings": []},
            }
        )
        with self.assertRaises(ValueError):
            export_report_pdf(report, "unused.pdf")
