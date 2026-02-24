"""
Artifact generation service for PDFs, code, and other content.

Provides functionality to generate various artifact types from content.
"""

import asyncio
import os
import io
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from reportlab.lib.pagesizes import letter, A4, legal
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

from app.models.artifact import (
    ArtifactCreateRequest,
    ArtifactResponse,
    ArtifactType,
    ArtifactStatus,
    PDFArtifactOptions,
    CodeArtifactOptions,
)


class ArtifactService:
    """
    Service for generating various artifact types.

    Supports:
    - PDF generation from text/markdown
    - Code file generation with syntax
    - Markdown documents
    - HTML documents
    - Plain text files
    """

    # Storage directory for generated artifacts
    STORAGE_DIR = os.getenv("ARTIFACT_STORAGE_DIR", "./artifacts")
    BASE_URL = os.getenv("ARTIFACT_BASE_URL", "/api/artifacts")

    def __init__(self):
        """Initialize the artifact service."""
        self.storage_dir = Path(self.STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories by type
        for artifact_type in ArtifactType:
            (self.storage_dir / artifact_type.value).mkdir(exist_ok=True)

    def _generate_filename(
        self,
        artifact_type: ArtifactType,
        title: Optional[str],
        original_filename: Optional[str],
    ) -> str:
        """Generate a unique filename for the artifact."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]

        # Use original filename if provided
        if original_filename:
            name, ext = os.path.splitext(original_filename)
            return f"{name}_{timestamp}_{unique_id}{ext}"

        # Generate based on type
        extensions = {
            ArtifactType.PDF: ".pdf",
            ArtifactType.CODE: ".txt",
            ArtifactType.MARKDOWN: ".md",
            ArtifactType.HTML: ".html",
            ArtifactType.TEXT: ".txt",
        }

        base_title = (title or "artifact").replace(" ", "_").lower()
        ext = extensions.get(artifact_type, ".txt")

        return f"{base_title}_{timestamp}_{unique_id}{ext}"

    async def generate_pdf(
        self,
        content: str,
        options: Optional[PDFArtifactOptions] = None,
    ) -> bytes:
        """
        Generate a PDF from content.

        Args:
            content: Text/markdown content
            options: PDF generation options

        Returns:
            PDF file bytes
        """
        options = options or PDFArtifactOptions()

        # Page size mapping
        page_sizes = {
            "A4": A4,
            "Letter": letter,
            "Legal": legal,
        }
        page_size = page_sizes.get(options.page_size, A4)

        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=page_size,
            leftMargin=1 * inch,
            rightMargin=1 * inch,
            topMargin=1 * inch,
            bottomMargin=1 * inch,
        )

        # Build styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            alignment=TA_CENTER,
            spaceAfter=30,
        )

        # Parse margin
        try:
            margin_value = float(options.margin.replace("in", "")) * inch
        except Exception:
            margin_value = 1 * inch

        # Build story (content flowables)
        story = []

        # Add title if provided
        if options.title:
            story.append(Paragraph(options.title, title_style))
            story.append(Spacer(1, 0.3 * inch))

        # Add content
        # Simple paragraph wrapping - for markdown, would need more parsing
        lines = content.split("\n\n")
        for line in lines:
            if line.strip():
                # Escape special characters
                escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(escaped, styles["BodyText"]))
                story.append(Spacer(1, 0.1 * inch))

        # Add page numbers
        if options.include_page_numbers:
            # In a full implementation, would add page number templates
            pass

        # Build PDF
        doc.build(story)

        return buffer.getvalue()

    async def generate_code(
        self,
        content: str,
        options: Optional[CodeArtifactOptions] = None,
    ) -> str:
        """
        Generate a code file with optional formatting.

        Args:
            content: Code content
            options: Code generation options

        Returns:
            Formatted code string
        """
        options = options or CodeArtifactOptions()

        lines = content.split("\n")

        if options.include_line_numbers:
            numbered = []
            width = len(str(len(lines)))
            for i, line in enumerate(lines, 1):
                numbered.append(f"{i:>{width}} | {line}")
            return "\n".join(numbered)

        return content

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system usage."""
        # Remove or replace dangerous characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")
        return filename

    async def create_artifact(self, request: ArtifactCreateRequest) -> ArtifactResponse:
        """
        Create an artifact from the request.

        Args:
            request: Artifact creation request

        Returns:
            ArtifactResponse with artifact details
        """
        artifact_id = f"art_{uuid.uuid4().hex}"

        # Generate filename
        filename = self._sanitize_filename(
            self._generate_filename(
                request.artifact_type,
                request.title,
                request.filename,
            )
        )

        # Full file path
        type_dir = self.storage_dir / request.artifact_type.value
        file_path = type_dir / filename

        # Generate artifact content
        content_bytes: Optional[bytes] = None
        content_preview: Optional[str] = None
        file_size: Optional[int] = None

        try:
            if request.artifact_type == ArtifactType.PDF:
                pdf_options = PDFArtifactOptions(**request.options) if request.options else None
                content_bytes = await self.generate_pdf(request.content, pdf_options)
                content_preview = request.content[:200] + "..." if len(request.content) > 200 else request.content

            elif request.artifact_type == ArtifactType.CODE:
                code_options = CodeArtifactOptions(**request.options) if request.options else None
                formatted_code = await self.generate_code(request.content, code_options)
                content_bytes = formatted_code.encode("utf-8")
                content_preview = formatted_code[:200] + "..." if len(formatted_code) > 200 else formatted_code

            elif request.artifact_type == ArtifactType.MARKDOWN:
                content_bytes = request.content.encode("utf-8")
                content_preview = request.content[:200] + "..." if len(request.content) > 200 else request.content

            elif request.artifact_type == ArtifactType.HTML:
                # Wrap content in basic HTML structure
                html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{request.title or 'Artifact'}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }}
    </style>
</head>
<body>
{request.content}
</body>
</html>"""
                content_bytes = html_content.encode("utf-8")
                content_preview = request.content[:200] + "..." if len(request.content) > 200 else request.content

            else:  # TEXT
                content_bytes = request.content.encode("utf-8")
                content_preview = request.content[:200] + "..." if len(request.content) > 200 else request.content

            # Write to file
            with open(file_path, "wb") as f:
                f.write(content_bytes)

            file_size = len(content_bytes)

            return ArtifactResponse(
                id=artifact_id,
                artifact_type=request.artifact_type,
                status=ArtifactStatus.COMPLETED,
                title=request.title,
                filename=filename,
                download_url=f"{self.BASE_URL}/{artifact_id}/download",
                content_preview=content_preview,
                file_size_bytes=file_size,
                metadata=request.metadata,
                created_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
            )

        except Exception as e:
            return ArtifactResponse(
                id=artifact_id,
                artifact_type=request.artifact_type,
                status=ArtifactStatus.FAILED,
                title=request.title,
                filename=filename,
                error_message=str(e),
                metadata=request.metadata,
                created_at=datetime.utcnow(),
            )

    async def get_artifact(self, artifact_id: str) -> Optional[ArtifactResponse]:
        """
        Get artifact by ID.

        Args:
            artifact_id: Artifact identifier

        Returns:
            ArtifactResponse if found, None otherwise
        """
        # Search for artifact file
        for artifact_type in ArtifactType:
            type_dir = self.storage_dir / artifact_type.value
            for file_path in type_dir.glob(f"*{artifact_id}*"):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    return ArtifactResponse(
                        id=artifact_id,
                        artifact_type=artifact_type,
                        status=ArtifactStatus.COMPLETED,
                        filename=file_path.name,
                        download_url=f"{self.BASE_URL}/{artifact_id}/download",
                        file_size_bytes=file_size,
                    )

        return None

    async def get_artifact_content(self, artifact_id: str) -> Optional[bytes]:
        """
        Get artifact file content.

        Args:
            artifact_id: Artifact identifier

        Returns:
            File content bytes if found, None otherwise
        """
        for artifact_type in ArtifactType:
            type_dir = self.storage_dir / artifact_type.value
            for file_path in type_dir.glob(f"*{artifact_id}*"):
                if file_path.is_file():
                    return file_path.read_bytes()

        return None

    async def list_artifacts(
        self,
        artifact_type: Optional[ArtifactType] = None,
    ) -> List[ArtifactResponse]:
        """
        List all artifacts.

        Args:
            artifact_type: Filter by artifact type (optional)

        Returns:
            List of ArtifactResponse
        """
        results = []

        types_to_search = [artifact_type] if artifact_type else list(ArtifactType)

        for art_type in types_to_search:
            type_dir = self.storage_dir / art_type.value
            for file_path in type_dir.iterdir():
                if file_path.is_file():
                    # Extract artifact ID from filename
                    # Format: title_TIMESTAMP_ID.ext
                    parts = file_path.stem.split("_")
                    if len(parts) >= 3:
                        potential_id = parts[-1]
                        if len(potential_id) == 8:  # UUID fragment length
                            file_size = file_path.stat().st_size
                            results.append(
                                ArtifactResponse(
                                    id=f"art_{potential_id}",
                                    artifact_type=art_type,
                                    status=ArtifactStatus.COMPLETED,
                                    filename=file_path.name,
                                    download_url=f"{self.BASE_URL}/art_{potential_id}/download",
                                    file_size_bytes=file_size,
                                )
                            )

        return results

    async def delete_artifact(self, artifact_id: str) -> bool:
        """
        Delete an artifact.

        Args:
            artifact_id: Artifact identifier

        Returns:
            True if deleted, False otherwise
        """
        for artifact_type in ArtifactType:
            type_dir = self.storage_dir / artifact_type.value
            for file_path in type_dir.glob(f"*{artifact_id}*"):
                if file_path.is_file():
                    file_path.unlink()
                    return True

        return False
