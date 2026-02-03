"""
Spreadsheet and Excel file processing components.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any

from ..app_config import get_services, tribe_supports_squads, get_available_squads

# Reuse centralized reseed helper
try:
    from .confluence import trigger_reseed_for_upload
except Exception:
    try:
        from .pdf import trigger_reseed_for_upload
    except Exception:
        trigger_reseed_for_upload = None

from ..vector_store_interface import DocumentType


# -------------------------------------------------------------------
# Data Processing
# -------------------------------------------------------------------
def process_dataframe(
    df: pd.DataFrame,
    filename: str,
    sheet_name: str,
    include_headers: bool,
    chunk_size: int,
) -> list:
    """Convert DataFrame to text chunks"""

    if df.empty:
        return []

    df = df.dropna(how="all").fillna("")

    for col in df.columns:
        df[col] = df[col].astype(str)

    services = get_services()
    return services.chunker.chunk_spreadsheet_rows(
        df,
        filename=filename,
        sheet_name=sheet_name,
        include_headers=include_headers,
        max_rows_per_chunk=chunk_size,
    )


# -------------------------------------------------------------------
# Squad Selection
# -------------------------------------------------------------------
def _render_squad_selection_spreadsheet(component_key: str) -> Optional[str]:
    """Render squad selection for spreadsheet uploads"""

    current_tribe_title = st.session_state.get("tribe_selector", "Mortgage")
    current_tribe = current_tribe_title.lower() if current_tribe_title else "mortgage"

    if not tribe_supports_squads(current_tribe):
        return None

    available_squads = get_available_squads(current_tribe)
    if not available_squads:
        return None

    st.subheader("📂 Squad Selection")

    selected_squad = st.selectbox(
        "Select target squad for this spreadsheet:",
        options=available_squads,
        index=0,
        help="Choose which squad database this spreadsheet content should be stored in",
        key=f"squad_selection_{component_key}",
    )

    if selected_squad:
        st.success(
            f"Content will be stored in: **{current_tribe.title()} - {selected_squad}**"
        )

    return selected_squad


# -------------------------------------------------------------------
# Sidebar Upload
# -------------------------------------------------------------------
def render_spreadsheet_upload_sidebar():
    """Render spreadsheet upload interface in sidebar"""

    st.sidebar.subheader("Spreadsheet Upload")

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
                help="Choose squad database for spreadsheet upload",
                key="sidebar_spreadsheet_squad_selection",
            )

    uploaded_files = st.sidebar.file_uploader(
        "Upload spreadsheet files",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        help="Upload Excel, XLSX, or CSV files for knowledge ingestion",
    )

    if not uploaded_files:
        return

    st.sidebar.markdown("**Processing Options**")

    treat_as_knowledge = st.sidebar.checkbox(
        "Treat as knowledge documents",
        value=True,
        help="Process spreadsheet content as knowledge for RAG",
    )

    include_headers = st.sidebar.checkbox(
        "Include headers in content",
        value=True,
        help="Include column headers in the processed text",
    )

    max_rows = st.sidebar.number_input(
        "Max rows per sheet",
        min_value=10,
        max_value=10000,
        value=1000,
        help="Limit rows to process per sheet",
    )

    chunk_size = st.sidebar.number_input(
        "Rows per chunk",
        min_value=5,
        max_value=500,
        value=20,
        help="Number of rows to group per text chunk",
    )

    if st.sidebar.button("Process Spreadsheets", key="process_spreadsheets"):
        process_spreadsheet_files(
            uploaded_files,
            treat_as_knowledge,
            include_headers,
            max_rows,
            chunk_size,
            target_squad,
        )


# -------------------------------------------------------------------
# Main Upload
# -------------------------------------------------------------------
def render_spreadsheet_upload():
    """Render spreadsheet upload interface in main content"""

    st.markdown("### Spreadsheet Upload")

    target_squad = _render_squad_selection_spreadsheet("spreadsheet_main")

    uploaded_files = st.file_uploader(
        "Upload spreadsheet files",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        help="Upload Excel, XLSX, or CSV files for knowledge ingestion",
    )

    if not uploaded_files:
        return

    st.markdown("**Processing Options**")

    treat_as_knowledge = st.checkbox(
        "Treat as knowledge documents",
        value=True,
        help="Store processed content as knowledge documents for RAG",
    )

    include_headers = st.checkbox(
        "Include column headers",
        value=True,
        help="Include column headers in processed text",
    )

    max_rows = st.number_input(
        "Maximum rows to process per sheet",
        min_value=10,
        max_value=10000,
        value=1000,
        help="Limit rows to process per sheet",
    )

    chunk_size = st.number_input(
        "Rows per chunk",
        min_value=5,
        max_value=100,
        value=20,
        help="Number of rows to group per text chunk",
    )

    if st.button("Process Files", type="primary"):
        process_spreadsheet_files(
            uploaded_files,
            treat_as_knowledge,
            include_headers,
            max_rows,
            chunk_size,
            target_squad,
        )

    st.markdown("**Preview uploaded file**")
    file_names = [f.name for f in uploaded_files]
    selected_preview = st.selectbox(
        "Select a file to preview:",
        options=file_names,
        key="spreadsheet_preview_select",
    )

    preview_file = next((f for f in uploaded_files if f.name == selected_preview), None)
    if not preview_file:
        return

    try:
        if preview_file.name.endswith(".csv"):
            df_preview = pd.read_csv(preview_file, nrows=10)
            st.dataframe(df_preview)

        elif preview_file.name.endswith((".xlsx", ".xls")):
            excel_file = pd.ExcelFile(preview_file)
            st.write(f"Sheets: {excel_file.sheet_names}")
            sel_sheet = st.selectbox(
                "Select sheet to preview:",
                excel_file.sheet_names,
                key="preview_sheet_select",
            )
            df_preview = pd.read_excel(
                preview_file, sheet_name=sel_sheet, nrows=10
            )
            st.dataframe(df_preview)

    except Exception as e:
        st.warning(f"Could not preview file {preview_file.name}: {e}")


# -------------------------------------------------------------------
# CSV Upload (Alternate Entry)
# -------------------------------------------------------------------
def render_csv_upload():
    """Render CSV / Excel upload interface"""

    st.markdown("### CSV/Excel File Upload")

    target_squad = _render_squad_selection_spreadsheet("csv_main")

    uploaded_files = st.file_uploader(
        "Upload CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Upload CSV or Excel files to extract knowledge",
    )

    if not uploaded_files:
        return

    st.markdown("**Files to process:**")
    for file in uploaded_files:
        st.write(f"📄 {file.name} ({file.size:,} bytes)")

    col1, col2 = st.columns(2)

    with col1:
        treat_as_knowledge = st.checkbox(
            "Process as knowledge documents",
            value=True,
            help="Include spreadsheet content in knowledge base for RAG",
        )

        include_headers = st.checkbox(
            "Include column headers",
            value=True,
            help="Include headers in processed content",
        )

    with col2:
        max_rows_per_sheet = st.number_input(
            "Max rows per sheet",
            min_value=10,
            max_value=5000,
            value=1000,
            help="Limit rows processed per sheet",
        )

        chunk_size = st.number_input(
            "Rows per chunk",
            min_value=5,
            max_value=100,
            value=20,
            help="Number of rows to group per text chunk",
        )

    if st.button("Process Files", type="primary"):
        target_squad = _render_squad_selection_spreadsheet("main_upload")
        process_spreadsheet_files(
            uploaded_files,
            treat_as_knowledge,
            include_headers,
            max_rows_per_sheet,
            chunk_size,
            target_squad,
        )


# ------------------------------------------------------------
# Single Spreadsheet Processing
# ------------------------------------------------------------
def process_single_spreadsheet(
    file,
    include_headers: bool = True,
    max_rows: int = 1000,
    chunk_size: int = 20,
) -> List[str]:
    """Process a single spreadsheet file and return text chunks"""

    logger = logging.getLogger(__name__)
    chunks: List[str] = []

    try:
        logger.info(f"Using row-based chunking for spreadsheet: {file.name}")

        if file.name.endswith(".csv"):
            df = pd.read_csv(file, nrows=max_rows)
            chunks.extend(
                process_dataframe(df, file.name, "Sheet1", include_headers, chunk_size)
            )

        elif file.name.endswith((".xlsx", ".xls")):
            excel_file = pd.ExcelFile(file)

            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(
                        file, sheet_name=sheet_name, nrows=max_rows
                    )
                    sheet_chunks = process_dataframe(
                        df, file.name, sheet_name, include_headers, chunk_size
                    )
                    chunks.extend(sheet_chunks)
                except Exception as e:
                    st.warning(
                        f"Could not process sheet '{sheet_name}' in {file.name}: {e}"
                    )
                    continue

        return chunks

    except Exception as e:
        st.error(f"Error processing {file.name}: {e}")
        return []


# ------------------------------------------------------------
# Sidebar Spreadsheet Processing
# ------------------------------------------------------------
def process_uploaded_spreadsheets(
    files,
    treat_as_knowledge: bool,
    include_headers: bool,
    max_rows: int,
    chunk_size: int = 20,
    target_squad: Optional[str] = None,
):
    """Process uploaded spreadsheet files in sidebar context"""

    services = get_services()

    if not services.is_initialized():
        st.sidebar.error("Services not initialized")
        return

    total_chunks = 0
    processed_files = 0

    try:
        for file in files:
            with st.spinner(f"Processing {file.name}..."):
                chunks = process_single_spreadsheet(
                    file, include_headers, max_rows, chunk_size
                )

            if chunks and treat_as_knowledge:
                metadata = {
                    "source": file.name,
                    "file_type": "spreadsheet",
                    "sheet_count": len(chunks),
                    "processing_options": {
                        "include_headers": include_headers,
                        "max_rows": max_rows,
                    },
                }

                if target_squad:
                    metadata["target_squad"] = target_squad
                    current_tribe_title = st.session_state.get(
                        "tribe_selector", "Mortgage"
                    )
                    current_tribe = (
                        current_tribe_title.lower()
                        if current_tribe_title
                        else "mortgage"
                    )
                    metadata["tribe"] = current_tribe

                services.vector_store.upsert_documents(
                    DocumentType.KNOWLEDGE,
                    f"spreadsheet:{file.name}",
                    chunks,
                    [metadata] * len(chunks),
                )

                total_chunks += len(chunks)
                processed_files += 1

                st.info(f"Processed {file.name}: {len(chunks)} chunks")

        if processed_files > 0:
            st.sidebar.success(
                f"Successfully processed {processed_files} spreadsheet files!"
            )
            st.info(f"Total chunks stored: {total_chunks}")

            _clear_kb_cache()

            if trigger_reseed_for_upload:
                try:
                    trigger_reseed_for_upload(services, target_squad)
                except Exception as e:
                    st.sidebar.warning(f"Reseed trigger failed: {e}")
        else:
            st.sidebar.warning("No files were processed successfully")

    except Exception as e:
        st.sidebar.error(f"Error processing spreadsheets: {str(e)}")


# ------------------------------------------------------------
# Main Content Spreadsheet Processing
# ------------------------------------------------------------
def process_spreadsheet_files(
    files,
    treat_as_knowledge: bool,
    include_headers: bool,
    max_rows: int,
    chunk_size: int,
    target_squad: Optional[str] = None,
):
    """Process spreadsheet files in main content context"""

    services = get_services()

    if not services.is_initialized():
        st.error("Services not initialized")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_chunks = 0
    processed_files = 0

    try:
        for i, file in enumerate(files):
            status_text.text(f"Processing {file.name}...")
            progress_bar.progress(i / len(files))

            chunks = process_single_spreadsheet(
                file, include_headers, max_rows, chunk_size
            )

            if chunks and treat_as_knowledge:
                metadata = {
                    "source": file.name,
                    "file_type": "spreadsheet",
                    "chunk_size": chunk_size,
                    "total_chunks": len(chunks),
                    "processing_options": {
                        "include_headers": include_headers,
                        "max_rows": max_rows,
                    },
                }

                if target_squad:
                    metadata["target_squad"] = target_squad
                    current_tribe_title = st.session_state.get(
                        "tribe_selector", "Mortgage"
                    )
                    current_tribe = (
                        current_tribe_title.lower()
                        if current_tribe_title
                        else "mortgage"
                    )
                    metadata["tribe"] = current_tribe

                services.vector_store.upsert_documents(
                    DocumentType.KNOWLEDGE,
                    f"spreadsheet:{file.name}",
                    chunks,
                    [metadata] * len(chunks),
                )

                total_chunks += len(chunks)
                processed_files += 1

                st.success(
                    f"✓ {file.name}: {len(chunks)} chunks processed"
                )

        progress_bar.progress(1.0)
        status_text.text("Processing complete!")

        if processed_files > 0:
            st.success(
                f"Successfully processed {processed_files} spreadsheet files!"
            )
            st.info(
                f"Total chunks stored in knowledge base: {total_chunks}"
            )

            _clear_kb_cache()

            if trigger_reseed_for_upload:
                try:
                    trigger_reseed_for_upload(services, target_squad)
                except Exception as e:
                    st.warning(f"Reseed trigger failed: {e}")
        else:
            st.warning("No files were processed successfully")

    except Exception as e:
        st.error(f"Error processing spreadsheet files: {str(e)}")



# ============================================================
# Excel analysis UI
# ============================================================
def render_excel_analysis():
    """Render Excel file analysis interface"""

    st.markdown("### Excel File Analysis")

    uploaded_file = st.file_uploader(
        "Upload Excel file for analysis",
        type=["xlsx", "xls"],
        help="Upload an Excel file to analyze its structure and content",
    )

    if not uploaded_file:
        return

    try:
        excel_file = pd.ExcelFile(uploaded_file)

        st.markdown("**File Information:**")
        st.info(f"File: {uploaded_file.name}")
        st.info(f"Sheets: {len(excel_file.sheet_names)}")

        selected_sheet = st.selectbox(
            "Select sheet to preview:",
            excel_file.sheet_names,
        )

        if not selected_sheet:
            return

        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, nrows=100)

        st.markdown(f"**Sheet: {selected_sheet}**")
        st.info(f"Rows: {len(df)}, Columns: {len(df.columns)}")

        st.markdown("**Columns:**")
        col_info = []
        for col in df.columns:
            col_info.append(f"- {col} ({df[col].count()} non-null values)")
        st.write("\n".join(col_info))

        st.markdown("**Data Preview (first 10 rows):**")
        st.dataframe(df.head(10))

        st.markdown("**Processing Options:**")
        col1, col2 = st.columns(2)

        with col1:
            max_rows = st.number_input(
                "Max rows to process",
                min_value=10,
                max_value=10000,
                value=min(1000, len(df)),
            )

        with col2:
            chunk_rows = st.number_input(
                "Rows per chunk",
                min_value=5,
                max_value=100,
                value=20,
            )

        target_squad = _render_squad_selection_spreadsheet("individual_sheet")

        if st.button("Process Sheet", type="primary"):
            process_excel_sheet(
                uploaded_file,
                selected_sheet,
                max_rows,
                chunk_rows,
                target_squad,
            )

    except Exception as e:
        st.error(f"Error analyzing Excel file: {str(e)}")


# ============================================================
# Process individual Excel sheet
# ============================================================
def process_excel_sheet(
    file,
    sheet_name: str,
    max_rows: int,
    chunk_rows: int,
    target_squad: Optional[str] = None,
):
    """Process a specific Excel sheet"""

    services = get_services()
    if not services.is_initialized():
        st.error("Services not initialized")
        return

    try:
        with st.spinner(f"Processing sheet: {sheet_name}..."):
            df = pd.read_excel(file, sheet_name=sheet_name, nrows=max_rows)
            chunks = process_dataframe(
                df, file.name, sheet_name, True, chunk_rows
            )

        if not chunks:
            st.warning("No content could be extracted from the sheet")
            return

        metadata = {
            "source": file.name,
            "sheet_name": sheet_name,
            "file_type": "excel",
            "rows_processed": len(df),
            "chunk_size": chunk_rows,
            "total_chunks": len(chunks),
        }

        if target_squad:
            metadata["target_squad"] = target_squad
            current_tribe_title = st.session_state.get("tribe_selector", "Mortgage")
            metadata["tribe"] = (
                current_tribe_title.lower()
                if current_tribe_title
                else "mortgage"
            )

        services.vector_store.upsert_documents(
            DocumentType.KNOWLEDGE,
            f"excel:{file.name}:{sheet_name}",
            chunks,
            [metadata] * len(chunks),
        )

        st.success(f"Successfully processed sheet '{sheet_name}'")
        st.info(f"Processed {len(df)} rows into {len(chunks)} chunks")

        _clear_kb_cache()

        if trigger_reseed_for_upload:
            try:
                trigger_reseed_for_upload(services, target_squad)
            except Exception as e:
                st.warning(f"Reseed trigger failed: {e}")

    except Exception as e:
        st.error(f"Error processing sheet: {str(e)}")


# ============================================================
# Cache cleanup
# ============================================================
def _clear_kb_cache():
    """Clear knowledge base cache"""

    for key in list(st.session_state.keys()):
        if key.startswith("kb_stats") or key == "kb_stats_cache":
            del st.session_state[key]
