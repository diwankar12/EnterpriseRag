import streamlit as st
from pathlib import Path
import logging

from .app_config import get_services, tribe_supports_squads
from ..vector_store_interface import DocumentType
from .pdf import trigger_reseed_for_upload


def _store_documents_squad_aware(
    services,
    doc_type,
    source_key,
    documents,
    metadatas,
    target_squad=None,
):
    """
    Store documents in appropriate collection based on squad selection.
    """

    # Get tribe from session state and convert to lowercase
    current_tribe_title = st.session_state.get("tribe_selector", "Mortgage")
    current_tribe = (
        current_tribe_title.lower() if current_tribe_title else "mortgage"
    )

    # If no squad targeting or tribe doesn't support squads, use default storage
    if not target_squad or not tribe_supports_squads(current_tribe):
        services.vector_store.upsert_documents(
            doc_type=doc_type,
            source_key=source_key,
            documents=documents,
            metadatas=metadatas,
        )
        return

    # Add squad + tribe information to metadata
    enhanced_metadatas = []
    for metadata in metadatas:
        enhanced_metadata = metadata.copy()
        enhanced_metadata["target_squad"] = target_squad
        enhanced_metadata["tribe"] = current_tribe
        enhanced_metadatas.append(enhanced_metadata)

    # Store with enhanced metadata
    services.vector_store.upsert_documents(
        doc_type=doc_type,
        source_key=f"{target_squad}_{source_key}",
        documents=documents,
        metadatas=enhanced_metadatas,
    )


def _clear_kb_cache():
    """Helper function to clear knowledge base cache"""
    for key in list(st.session_state.keys()):
        if key.startswith("kb_stats_") or key == "kb_stats_cache":
            del st.session_state[key]


def process_uploaded_files(uploaded_files, target_squad=None):
    """Process uploaded PDF, PPTX, DOCX and image files"""

    services = get_services()

    if not services.is_initialized():
        st.error("Services not initialized")
        return

    # Use main content area if called from there, sidebar otherwise
    progress_container = st if "main_" in str(uploaded_files) else st.sidebar
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()

    logger = logging.getLogger(__name__)
    logger.info(
        f"process_uploaded_files started; last_pdf_ui={st.session_state.get('last_pdf_ui')}"
    )

    try:
        total_files = len(uploaded_files)
        processed_count = {"pdfs": 0, "pptx": 0, "docx": 0, "images": 0}
        chunk_counts = {"pdfs": 0, "pptx": 0, "docx": 0, "images": 0}

        for i, uploaded_file in enumerate(uploaded_files):
            file_extension = Path(uploaded_file.name).suffix.lower()

            progress = (i + 1) / total_files
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}...")

            try:
                # ---------------- PDF ----------------
                if file_extension == ".pdf":
                    uploaded_file.seek(0)
                    pdf_text = services.pdf_processor.extract_text_from_pdf(uploaded_file)

                    if pdf_text and pdf_text.strip():
                        chunks = services.chunker.chunk_text_semantic(pdf_text)
                        chunk_counts["pdfs"] += len(chunks)

                        metadata = {
                            "source": uploaded_file.name,
                            "file_type": "pdf",
                            "upload_timestamp": str(
                                st.session_state.get("generation_timestamp", "")
                            ),
                        }
                        metadatas = [
                            {**metadata, "chunk_id": j}
                            for j in range(len(chunks))
                        ]

                        _store_documents_squad_aware(
                            services,
                            DocumentType.KNOWLEDGE,
                            uploaded_file.name,
                            chunks,
                            metadatas,
                            target_squad,
                        )

                        processed_count["pdfs"] += 1
                        trigger_reseed_for_upload(services, target_squad)

                # ---------------- PPTX ----------------
                elif file_extension in [".pptx", ".ppt"]:
                    uploaded_file.seek(0)
                    pptx_text = services.pdf_processor.extract_text_from_file(uploaded_file)

                    if pptx_text and pptx_text.strip():
                        chunks = services.chunker.chunk_text_semantic(pptx_text)
                        chunk_counts["pptx"] += len(chunks)

                        metadata = {
                            "source": uploaded_file.name,
                            "file_type": "powerpoint",
                            "upload_timestamp": str(
                                st.session_state.get("generation_timestamp", "")
                            ),
                        }
                        metadatas = [
                            {**metadata, "chunk_id": j}
                            for j in range(len(chunks))
                        ]

                        _store_documents_squad_aware(
                            services,
                            DocumentType.KNOWLEDGE,
                            uploaded_file.name,
                            chunks,
                            metadatas,
                            target_squad,
                        )

                        processed_count["pptx"] += 1
                        trigger_reseed_for_upload(services, target_squad)

                # ---------------- DOCX ----------------
                elif file_extension in [".docx", ".doc"]:
                    uploaded_file.seek(0)
                    word_text = services.pdf_processor.extract_text_from_file(uploaded_file)

                    if word_text and word_text.strip():
                        chunks = services.chunker.chunk_text_semantic(word_text)
                        chunk_counts["docx"] += len(chunks)

                        metadata = {
                            "source": uploaded_file.name,
                            "file_type": "word",
                            "upload_timestamp": str(
                                st.session_state.get("generation_timestamp", "")
                            ),
                        }
                        metadatas = [
                            {**metadata, "chunk_id": j}
                            for j in range(len(chunks))
                        ]

                        _store_documents_squad_aware(
                            services,
                            DocumentType.KNOWLEDGE,
                            uploaded_file.name,
                            chunks,
                            metadatas,
                            target_squad,
                        )

                        processed_count["docx"] += 1
                        trigger_reseed_for_upload(services, target_squad)

                # ---------------- IMAGES ----------------
                elif file_extension in [
                    ".png",
                    ".jpeg",
                    ".jpg",
                    ".gif",
                    ".bmp",
                    ".tiff",
                    ".webp",
                ]:
                    image_text = services.pdf_processor.extract_text_from_image(uploaded_file)

                    if image_text and image_text.strip():
                        chunks = services.chunker.chunk_text_semantic(image_text)
                        chunk_counts["images"] += len(chunks)

                        metadata = {
                            "source": uploaded_file.name,
                            "file_type": "image",
                            "extracted_via": "vertex_ai_multimodal",
                            "analysis_method": "gcp_vertex_ai_gemini",
                            "upload_timestamp": str(
                                st.session_state.get("generation_timestamp", "")
                            ),
                        }
                        metadatas = [
                            {**metadata, "chunk_id": j}
                            for j in range(len(chunks))
                        ]

                        _store_documents_squad_aware(
                            services,
                            DocumentType.KNOWLEDGE,
                            uploaded_file.name,
                            chunks,
                            metadatas,
                            target_squad,
                        )

                        processed_count["images"] += 1
                        trigger_reseed_for_upload(services, target_squad)

            except Exception as file_error:
                logger.error(f"Error processing {uploaded_file.name}: {file_error}")
                continue

        # Summary
        summary_lines = []
        if processed_count["pdfs"] > 0:
            summary_lines.append(
                f"PDFs: {processed_count['pdfs']} files, {chunk_counts['pdfs']} chunks"
            )
        if processed_count["pptx"] > 0:
            summary_lines.append(
                f"PowerPoint: {processed_count['pptx']} files, {chunk_counts['pptx']} chunks"
            )
        if processed_count["docx"] > 0:
            summary_lines.append(
                f"Word documents: {processed_count['docx']} files, {chunk_counts['docx']} chunks"
            )
        if processed_count["images"] > 0:
            summary_lines.append(
                f"Images: {processed_count['images']} files, {chunk_counts['images']} chunks"
            )

        if summary_lines:
            st.info("Upload summary:\n" + "\n".join(summary_lines))
        else:
            status_text.warning("No files were successfully processed")

        _clear_kb_cache()

    except Exception as e:
        progress_container.error(f"Error processing files: {str(e)}")

    finally:
        progress_bar.empty()



# ============================================================
# Sidebar PDF / Document Upload UI
# ============================================================

def render_pdf_upload_sidebar():
    logger = logging.getLogger(__name__)

    try:
        st.session_state["last_pdf_ui"] = "sidebar"
    except Exception:
        pass

    logger.info("Rendering PDF upload (sidebar)")

    target_squad = None

    current_tribe_title = st.session_state.get("tribe_selector", "Mortgage")
    current_tribe = current_tribe_title.lower() if current_tribe_title else "mortgage"

    if tribe_supports_squads(current_tribe):
        available_squads = get_available_squads(current_tribe)
        if available_squads:
            st.sidebar.subheader("Squad Selection")
            target_squad = st.sidebar.selectbox(
                "Select target squad:",
                options=available_squads,
                index=0,
                help="Choose squad database for upload",
                key="sidebar_pdf_squad_selection"
            )

    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF, PowerPoint, Word, and Image files:",
        type=["pdf", "pptx", "ppt", "docx", "doc", "png", "jpeg", "jpg", "gif", "bmp", "tiff", "webp"],
        accept_multiple_files=True,
        key="sidebar_pdf_uploader",
        help=(
            "Supported formats:\n"
            "PDF (.pdf)\n"
            "PowerPoint (.pptx, .ppt)\n"
            "Word (.docx, .doc)\n"
            "Images (.png, .jpg, .jpeg, .gif, .bmp, .tiff, .webp)"
        )
    )

    process_disabled = not uploaded_files
    if st.sidebar.button("Process Files", key="sidebar_process_pdfs", disabled=process_disabled):
        process_uploaded_files(uploaded_files, target_squad)


# ============================================================
# Main Content PDF / Document Upload UI
# ============================================================

def render_pdf_upload_main():
    logger = logging.getLogger(__name__)

    try:
        st.session_state["last_pdf_ui"] = "main"
    except Exception:
        pass

    logger.info("Rendering PDF upload (main area)")

    st.info(
        "**Supported File Types:**\n"
        "- **PDFs**: Extract text and content\n"
        "- **PowerPoint**: Extract text from slides\n"
        "- **Word**: Extract text from documents\n"
        "- **Images**: OCR via Vertex AI Gemini\n"
        "- **Screenshots / Diagrams**: Text extraction supported"
    )

    target_squad = render_squad_selection_for_upload("pdf_main")

    uploaded_files = st.file_uploader(
        "Upload PDF, PowerPoint, Word, and Image files:",
        type=["pdf", "pptx", "ppt", "docx", "doc", "png", "jpeg", "jpg", "gif", "bmp", "tiff", "webp"],
        accept_multiple_files=True,
        key="main_pdf_uploader",
        help="Upload documents for knowledge ingestion"
    )

    if uploaded_files:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Process Files", key="main_process_pdfs"):
                process_uploaded_files(uploaded_files, target_squad)

        with col2:
            if st.button("Clear", key="main_clear_pdfs"):
                st.session_state.main_pdf_uploader = None
                st.rerun()


# ============================================================
# Core File Processing Pipeline
# ============================================================

def process_uploaded_files(uploaded_files, target_squad=None):
    services = get_services()

    if not services.is_initialized():
        st.error("Services not initialized")
        return

    progress_container = st if "main" in str(uploaded_files) else st
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()

    logger = logging.getLogger(__name__)
    logger.info(f"process_uploaded_files started; last_pdf_ui={st.session_state.get('last_pdf_ui')}")

    try:
        total_files = len(uploaded_files)
        processed_count = {"pdfs": 0, "pptx": 0, "docx": 0, "images": 0}
        chunk_counts = {"pdfs": 0, "pptx": 0, "docx": 0, "images": 0}

        for i, uploaded_file in enumerate(uploaded_files):
            file_extension = Path(uploaded_file.name).suffix.lower()
            progress_bar.progress((i + 1) / total_files)
            status_text.text(f"Processing {uploaded_file.name}...")

            try:
                if file_extension == ".pdf":
                    _process_text_file(
                        services, uploaded_file, "pdf",
                        processed_count, chunk_counts, target_squad
                    )

                elif file_extension in [".pptx", ".ppt"]:
                    _process_text_file(
                        services, uploaded_file, "powerpoint",
                        processed_count, chunk_counts, target_squad
                    )

                elif file_extension in [".docx", ".doc"]:
                    _process_text_file(
                        services, uploaded_file, "word",
                        processed_count, chunk_counts, target_squad
                    )

                elif file_extension in [".png", ".jpeg", ".jpg", ".gif", ".bmp", ".tiff", ".webp"]:
                    _process_image_file(
                        services, uploaded_file,
                        processed_count, chunk_counts, target_squad
                    )

            except Exception as file_error:
                logger.error(f"Error processing {uploaded_file.name}: {file_error}")
                status_text.error(f"Error processing {uploaded_file.name}")
                continue

        _show_upload_summary(processed_count, chunk_counts)
        _clear_kb_cache()

    except Exception as e:
        progress_container.error(f"Error processing files: {str(e)}")

    finally:
        progress_bar.empty()



# ============================================================
# Helpers (internal)
# ============================================================

def _show_upload_summary(processed_count, chunk_counts):
    summary = []
    if processed_count["pdfs"]:
        summary.append(f"PDFs: {processed_count['pdfs']} files, {chunk_counts['pdfs']} chunks")
    if processed_count["pptx"]:
        summary.append(f"PowerPoint: {processed_count['pptx']} files, {chunk_counts['pptx']} chunks")
    if processed_count["docx"]:
        summary.append(f"Word: {processed_count['docx']} files, {chunk_counts['docx']} chunks")
    if processed_count["images"]:
        summary.append(f"Images: {processed_count['images']} files, {chunk_counts['images']} chunks")

    if summary:
        st.info("Upload summary:\n" + "\n".join(summary))
    else:
        st.warning("No files were successfully processed")





# ---------------------------------------------------------
# Squad selection helper
# ---------------------------------------------------------
def render_squad_selection_for_upload(component_key: str) -> str:
    """
    Render squad selection dropdown for upload components (only for mortgage tribe).
    """

    current_tribe_title = st.session_state.get("tribe_selector", "Mortgage")
    current_tribe = current_tribe_title.lower() if current_tribe_title else "mortgage"

    if not tribe_supports_squads(current_tribe):
        return None

    available_squads = get_available_squads(current_tribe)
    if not available_squads:
        return None

    st.subheader("📂 Squad Selection")

    selected_squad = st.selectbox(
        "Select target squad for this upload:",
        options=available_squads,
        index=0,
        help="Choose which squad database this content should be stored in",
        key=f"squad_selection_{component_key}",
    )

    if selected_squad:
        st.success(
            f"Content will be stored in: **{current_tribe.title()} - {selected_squad}**"
        )

    return selected_squad



# ---------------------------------------------------------
# Trigger reseed after upload
# ---------------------------------------------------------
def trigger_reseed_for_upload(services, target_squad=None):
    """
    Trigger reseed after an upload.
    """

    logger = logging.getLogger(__name__)

    if services is None:
        logger.debug("trigger_reseed_for_upload: services is None; skipping reseed")
        return

    current_tribe_title = st.session_state.get("tribe_selector", "Mortgage")
    current_tribe = current_tribe_title.lower() if current_tribe_title else "mortgage"

    immediate_enabled = (
        os.getenv("IMMEDIATE_RESEED_ON_SQUAD_UPLOAD", "true").lower()
        in ("1", "true", "yes")
    )

    if target_squad and tribe_supports_squads(current_tribe) and immediate_enabled:
        try:
            services.immediate_reseed()
        except Exception as e:
            logger.debug(
                f"Immediate reseed failed, falling back to scheduled reseed: {e}"
            )
            try:
                services.schedule_reseed()
            except Exception as se:
                logger.debug(f"Failed to schedule reseed after fallback: {se}")
    else:
        try:
            services.schedule_reseed()
        except Exception as e:
            logger.debug(f"Failed to schedule reseed: {e}")


# ---------------------------------------------------------
# Process uploaded files (PDF, PPT, DOCX, Images)
# ---------------------------------------------------------
def process_uploaded_files(uploaded_files, target_squad=None):
    """
    Process uploaded PDF, PowerPoint, Word, and Image files
    """

    services = get_services()

    if not services.is_initialized():
        st.error("Services not initialized")
        return

    progress_container = st
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()

    logger = logging.getLogger(__name__)
    logger.info(
        f"process_uploaded_files started; last_pdf_ui={st.session_state.get('last_pdf_ui')}"
    )

    try:
        total_files = len(uploaded_files)
        processed_count = {"pdfs": 0, "pptx": 0, "docx": 0, "images": 0}
        chunk_counts = {"pdfs": 0, "pptx": 0, "docx": 0, "images": 0}

        for i, uploaded_file in enumerate(uploaded_files):
            file_extension = Path(uploaded_file.name).suffix.lower()

            progress = (i + 1) / total_files
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}...")

            try:
                # ---------------- PDF ----------------
                if file_extension == ".pdf":
                    uploaded_file.seek(0)
                    pdf_text = services.pdf_processor.extract_text_from_pdf(uploaded_file)

                    if pdf_text and pdf_text.strip():
                        chunks = services.chunker.chunk_text_semantic(pdf_text)
                        chunk_counts["pdfs"] += len(chunks)

                        metadata = {
                            "source": uploaded_file.name,
                            "file_type": "pdf",
                            "upload_timestamp": str(
                                st.session_state.get("generation_timestamp", "")
                            ),
                        }

                        metadatas = [
                            {**metadata, "chunk_id": j} for j in range(len(chunks))
                        ]

                        _store_documents_squad_aware(
                            services,
                            DocumentType.KNOWLEDGE,
                            uploaded_file.name,
                            chunks,
                            metadatas,
                            target_squad,
                        )

                        processed_count["pdfs"] += 1
                        trigger_reseed_for_upload(services, target_squad)

                # ---------------- PPT ----------------
                elif file_extension in [".pptx", ".ppt"]:
                    uploaded_file.seek(0)
                    pptx_text = services.pdf_processor.extract_text_from_file(
                        uploaded_file
                    )

                    if pptx_text and pptx_text.strip():
                        chunks = services.chunker.chunk_text_semantic(pptx_text)
                        chunk_counts["pptx"] += len(chunks)

                        metadata = {
                            "source": uploaded_file.name,
                            "file_type": "powerpoint",
                            "upload_timestamp": str(
                                st.session_state.get("generation_timestamp", "")
                            ),
                        }

                        metadatas = [
                            {**metadata, "chunk_id": j} for j in range(len(chunks))
                        ]

                        _store_documents_squad_aware(
                            services,
                            DocumentType.KNOWLEDGE,
                            uploaded_file.name,
                            chunks,
                            metadatas,
                            target_squad,
                        )

                        processed_count["pptx"] += 1
                        trigger_reseed_for_upload(services, target_squad)

                # ---------------- WORD ----------------
                elif file_extension in [".docx", ".doc"]:
                    uploaded_file.seek(0)
                    word_text = services.pdf_processor.extract_text_from_file(
                        uploaded_file
                    )

                    if word_text and word_text.strip():
                        chunks = services.chunker.chunk_text_semantic(word_text)
                        chunk_counts["docx"] += len(chunks)

                        metadata = {
                            "source": uploaded_file.name,
                            "file_type": "word",
                            "upload_timestamp": str(
                                st.session_state.get("generation_timestamp", "")
                            ),
                        }

                        metadatas = [
                            {**metadata, "chunk_id": j} for j in range(len(chunks))
                        ]

                        _store_documents_squad_aware(
                            services,
                            DocumentType.KNOWLEDGE,
                            uploaded_file.name,
                            chunks,
                            metadatas,
                            target_squad,
                        )

                        processed_count["docx"] += 1
                        trigger_reseed_for_upload(services, target_squad)

                # ---------------- IMAGES ----------------
                elif file_extension in [
                    ".png",
                    ".jpeg",
                    ".jpg",
                    ".gif",
                    ".bmp",
                    ".tiff",
                    ".webp",
                ]:
                    image_text = services.pdf_processor.extract_text_from_image(
                        uploaded_file
                    )

                    if image_text and image_text.strip():
                        chunks = services.chunker.chunk_text_semantic(image_text)
                        chunk_counts["images"] += len(chunks)

                        metadata = {
                            "source": uploaded_file.name,
                            "file_type": "image",
                            "extracted_via": "vertex_ai_multimodal",
                            "analysis_method": "gcp_vertex_ai_gemini",
                            "upload_timestamp": str(
                                st.session_state.get("generation_timestamp", "")
                            ),
                        }

                        metadatas = [
                            {**metadata, "chunk_id": j} for j in range(len(chunks))
                        ]

                        _store_documents_squad_aware(
                            services,
                            DocumentType.KNOWLEDGE,
                            uploaded_file.name,
                            chunks,
                            metadatas,
                            target_squad,
                        )

                        processed_count["images"] += 1
                        trigger_reseed_for_upload(services, target_squad)

            except Exception as file_error:
                logger.error(
                    f"Error processing {uploaded_file.name}: {file_error}",
                    exc_info=True,
                )
                status_text.error(
                    f"Error processing {uploaded_file.name}: {str(file_error)}"
                )
                continue

        _clear_kb_cache()

    finally:
        progress_bar.empty()


# ---------------------------------------------------------
# Render PDF upload (MAIN area)
# ---------------------------------------------------------
def render_pdf_upload_main():
    logger = logging.getLogger(__name__)

    try:
        st.session_state["last_pdf_ui"] = "main"
    except Exception:
        pass

    logger.info("Rendering PDF upload (main area)")

    st.info(
        """
**Supported File Types:**
- **PDFs**
- **PowerPoint**
- **Word**
- **Images**
"""
    )

    target_squad = render_squad_selection_for_upload("pdf_main")

    uploaded_files = st.file_uploader(
        "Upload PDF, PowerPoint, Word, and Image files:",
        type=[
            "pdf",
            "pptx",
            "ppt",
            "docx",
            "doc",
            "png",
            "jpeg",
            "jpg",
            "gif",
            "bmp",
            "tiff",
            "webp",
        ],
        accept_multiple_files=True,
        key="main_pdf_uploader",
    )

    if uploaded_files:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Process Files", key="main_process_pdfs"):
                process_uploaded_files(uploaded_files, target_squad)
        with col2:
            if st.button("Clear", key="main_clear_pdfs"):
                st.session_state.main_pdf_uploader = None
                st.rerun()


# ---------------------------------------------------------
# Render PDF upload (SIDEBAR)
# ---------------------------------------------------------
def render_pdf_upload_sidebar():
    logger = logging.getLogger(__name__)

    try:
        st.session_state["last_pdf_ui"] = "sidebar"
    except Exception:
        pass

    logger.info("Rendering PDF upload (sidebar)")

    target_squad = None
    current_tribe_title = st.session_state.get("tribe_selector", "Mortgage")
    current_tribe = current_tribe_title.lower() if current_tribe_title else "mortgage"

    if tribe_supports_squads(current_tribe):
        available_squads = get_available_squads(current_tribe)
        if available_squads:
            st.sidebar.subheader("Squad Selection")
            target_squad = st.sidebar.selectbox(
                "Select target squad:",
                options=available_squads,
                key="sidebar_pdf_squad_selection",
            )

    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF, PowerPoint, Word, and Image files:",
        type=[
            "pdf",
            "pptx",
            "ppt",
            "docx",
            "doc",
            "png",
            "jpeg",
            "jpg",
            "gif",
            "bmp",
            "tiff",
            "webp",
        ],
        accept_multiple_files=True,
        key="sidebar_pdf_uploader",
    )

    if st.sidebar.button(
        "Process Files",
        key="sidebar_process_pdfs",
        disabled=not uploaded_files,
    ):
        process_uploaded_files(uploaded_files, target_squad)
