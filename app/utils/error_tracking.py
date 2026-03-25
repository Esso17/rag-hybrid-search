"""Error tracking and reporting for ingestion processes"""

import json
import logging
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ErrorTracker:
    """Track and report errors during document ingestion"""

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize error tracker

        Args:
            output_dir: Directory to save error reports (defaults to ./error_reports)
        """
        self.errors: list[dict[str, Any]] = []
        self.error_categories = defaultdict(list)
        self.output_dir = Path(output_dir) if output_dir else Path("./error_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = datetime.now()

    def track_error(
        self,
        error: Exception,
        document_idx: int,
        document_info: dict[str, Any],
        phase: str = "ingestion",
    ) -> None:
        """
        Track an error with full context

        Args:
            error: The exception that occurred
            document_idx: Index of the document being processed
            document_info: Document metadata and information
            phase: Phase where error occurred (loading, ingestion, etc.)
        """
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()

        # Build error record
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "document_idx": document_idx,
            "error_type": error_type,
            "error_message": error_message,
            "stack_trace": stack_trace,
            "document_info": {
                "title": document_info.get("title", "Unknown"),
                "source_file": document_info.get("metadata", {}).get("source_file", "Unknown"),
                "doc_id": document_info.get("doc_id", "Unknown"),
                "content_preview": (
                    document_info.get("content", "")[:200] + "..."
                    if "content" in document_info
                    else "N/A"
                ),
            },
        }

        # Store error
        self.errors.append(error_record)
        self.error_categories[error_type].append(error_record)

        # Log real-time detailed error
        self._log_error_realtime(error_record)

    def _log_error_realtime(self, error_record: dict[str, Any]) -> None:
        """Log error in real-time with full details"""
        doc_info = error_record["document_info"]

        logger.error("=" * 80)
        logger.error(f"⚠️  ERROR #{len(self.errors)} - {error_record['phase'].upper()} PHASE")
        logger.error(f"Document #{error_record['document_idx']}: {doc_info['title']}")
        logger.error(f"Source File: {doc_info['source_file']}")
        logger.error(f"Error Type: {error_record['error_type']}")
        logger.error(f"Error Message: {error_record['error_message']}")
        logger.error("-" * 80)
        logger.error("Stack Trace:")
        logger.error(error_record["stack_trace"])
        logger.error("=" * 80)

    def get_summary(self) -> dict[str, Any]:
        """Get error summary statistics"""
        total_errors = len(self.errors)

        if total_errors == 0:
            return {
                "total_errors": 0,
                "error_rate": 0,
                "categories": {},
                "message": "No errors occurred ✓",
            }

        # Count by category
        category_counts = {
            error_type: len(errors) for error_type, errors in self.error_categories.items()
        }

        # Most common errors
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            "total_errors": total_errors,
            "unique_error_types": len(self.error_categories),
            "category_counts": dict(sorted_categories),
            "most_common_error": sorted_categories[0] if sorted_categories else None,
            "duration": str(datetime.now() - self.start_time),
        }

    def print_summary(self) -> None:
        """Print error summary to console"""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("ERROR SUMMARY")
        print("=" * 80)

        if summary["total_errors"] == 0:
            print("✓ No errors occurred during ingestion")
            print("=" * 80)
            return

        print(f"Total Errors: {summary['total_errors']}")
        print(f"Unique Error Types: {summary['unique_error_types']}")
        print(f"Duration: {summary['duration']}")
        print("\n" + "-" * 80)
        print("ERROR BREAKDOWN BY TYPE:")
        print("-" * 80)

        for error_type, count in summary["category_counts"].items():
            percentage = (count / summary["total_errors"]) * 100
            print(f"  {error_type}: {count} ({percentage:.1f}%)")

        if summary["most_common_error"]:
            error_type, count = summary["most_common_error"]
            print(f"\n⚠️  Most Common Error: {error_type} ({count} occurrences)")

        print("=" * 80)

    def save_detailed_report(self, filename: Optional[str] = None) -> Path:
        """
        Save detailed error report to JSON file

        Args:
            filename: Optional custom filename (defaults to timestamp-based name)

        Returns:
            Path to saved report file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ingestion_errors_{timestamp}.json"

        filepath = self.output_dir / filename

        report = {
            "report_generated": datetime.now().isoformat(),
            "ingestion_start": self.start_time.isoformat(),
            "duration": str(datetime.now() - self.start_time),
            "summary": self.get_summary(),
            "errors": self.errors,
            "errors_by_category": dict(self.error_categories.items()),
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Detailed error report saved to: {filepath}")
        return filepath

    def get_investigation_guide(self) -> str:
        """Generate investigation guide for errors"""
        if not self.errors:
            return "No errors to investigate ✓"

        guide = [
            "\n" + "=" * 80,
            "ERROR INVESTIGATION GUIDE",
            "=" * 80,
        ]

        for error_type, errors in self.error_categories.items():
            guide.append(f"\n▶ {error_type} ({len(errors)} occurrences)")
            guide.append("-" * 80)

            # Show first 3 examples
            for i, error in enumerate(errors[:3], 1):
                doc_info = error["document_info"]
                guide.append(f"\n  Example {i}:")
                guide.append(f"    Document: {doc_info['title']}")
                guide.append(f"    Source: {doc_info['source_file']}")
                guide.append(f"    Message: {error['error_message']}")

            if len(errors) > 3:
                guide.append(f"\n  ... and {len(errors) - 3} more similar errors")

            # Add investigation tips
            guide.append("\n  💡 Investigation Tips:")
            if error_type == "KeyError":
                guide.append("    - Check if document structure matches expected schema")
                guide.append("    - Verify all required metadata fields are present")
            elif error_type == "ValueError":
                guide.append("    - Validate data types and formats")
                guide.append("    - Check for empty or malformed content")
            elif error_type == "UnicodeDecodeError":
                guide.append("    - Verify file encoding (try UTF-8 with error handling)")
                guide.append("    - Check for binary or corrupted files")
            elif "Embedding" in error_type or "API" in error_type:
                guide.append("    - Check API credentials and rate limits")
                guide.append("    - Verify network connectivity")
                guide.append("    - Review content length (may exceed model limits)")
            else:
                guide.append("    - Review stack trace in detailed error report")
                guide.append("    - Check document content and metadata")

        guide.append("\n" + "=" * 80)
        return "\n".join(guide)

    def print_investigation_guide(self) -> None:
        """Print investigation guide to console"""
        print(self.get_investigation_guide())


def create_error_tracker(output_dir: Optional[str] = None) -> ErrorTracker:
    """
    Factory function to create an error tracker

    Args:
        output_dir: Directory to save error reports

    Returns:
        Configured ErrorTracker instance
    """
    return ErrorTracker(output_dir=output_dir)
