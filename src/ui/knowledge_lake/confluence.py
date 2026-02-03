"""
Confluence integration components for content ingestion.
"""

import streamlit as st
import re
import os
import logging

from ..app_config import (
    get_services,
    tribe_supports_squads,
    get_available_squads,
)
from ..vector_store_interface import DocumentType


# ---------------------------------------------------------
# Reseed Trigger
# ---------------------------------------------------------

def trigger_reseed_for_upload(services, target_squad=None):
    """Trigger reseed after an upload."""

    logger = logging.getLogger(__name__)

    try:
        if services is None:
            logger.debug(
                "trigger_reseed_for_upload: services is None; skipping reseed"
            )
            return

        if not target_squad:
            for k, v in st.session_state.items():
                if isinstance(k, str) and k.startswith("squad_selection_") and v:
                    target_squad = v
                    break

        current_tribe_title = st.session_state.get("tribe_selector", "Mortgage")
        current_tribe = (
            current_tribe_title.lower() if current_tribe_title else "mortgage"
        )

        immediate_enabled = (
            os.getenv("IMMEDIATE_RESEED_ON_SQUAD_UPLOAD", "true")
            .lower()
            .strip()
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
                    logger.debug(
                        f"Failed to schedule reseed after fallback: {se}"
                    )
        else:
            try:
                services.schedule_reseed()
            except Exception as e:
                logger.debug(f"Failed to schedule reseed: {e}")

    except Exception as exc:
        logger.debug(
            f"Unexpected error in trigger_reseed_for_upload: {exc}"
        )


# ---------------------------------------------------------
# Squad Selection (Sidebar)
# ---------------------------------------------------------

def _render_squad_selection_confluence(component_key: str):
    """Render squad selection for confluence uploads"""

    current_tribe_title = st.session_state.get("tribe_selector", "Mortgage")
    current_tribe = (
        current_tribe_title.lower() if current_tribe_title else "mortgage"
    )

    if not tribe_supports_squads(current_tribe):
        return None

    available_squads = get_available_squads(current_tribe)
    if not available_squads:
        return None

    st.sidebar.subheader("📦 Squad Selection")

    selected_squad = st.sidebar.selectbox(
        "Select target squad:",
        options=available_squads,
        index=0,
        help="Choose squad database for confluence content",
        key=f"squad_selection_{component_key}",
    )

    if selected_squad:
        st.sidebar.success(
            f"Target: **{current_tribe.title()} – {selected_squad}**"
        )
        normalized = (
            str(selected_squad)
            .strip()
            .lower()
            .replace(" ", "_")
        )
        st.session_state[
            f"squad_selection_{component_key}_normalized"
        ] = normalized
        st.session_state["selected_squads_normalized"] = [normalized]

    return selected_squad


def _render_squad_selection_confluence_main(component_key: str):
    """Render squad selection in main content"""

    current_tribe_title = st.session_state.get("tribe_selector", "Mortgage")
    current_tribe = (
        current_tribe_title.lower() if current_tribe_title else "mortgage"
    )

    if not tribe_supports_squads(current_tribe):
        return None

    available_squads = get_available_squads(current_tribe)
    if not available_squads:
        return None

    st.subheader("📦 Squad Selection")

    selected_squad = st.selectbox(
        "Select target squad for this confluence content:",
        options=available_squads,
        index=0,
        help="Choose which squad database this confluence content should be stored in",
        key=f"squad_selection_{component_key}",
    )

    if selected_squad:
        st.success(
            f"Content will be stored in: **{current_tribe.title()} – {selected_squad}**"
        )

    return selected_squad


# ---------------------------------------------------------
# Sidebar Upload
# ---------------------------------------------------------

def render_confluence_upload():
    """Render Confluence upload interface for sidebar"""

    confluence_type = st.sidebar.radio(
        "Confluence Upload Type:",
        [
            "Single/Multiple URLs (Recommended)",
            "Bulk Parent + Child Pages",
        ],
        key="confluence_upload_type",
    )

    st.session_state.preferred_confluence_type = confluence_type

    global_confluence_token = st.sidebar.text_input(
        "Confluence API Token (optional)",
        value="",
        type="password",
        help=(
            "Provide an API token to use for Confluence requests during this session; "
            "if blank the configured token will be used"
        ),
        key="confluence_api_token_input_global",
    )

    if confluence_type == "Single/Multiple URLs (Recommended)":
        render_confluence_url_upload()
    elif confluence_type == "Bulk Parent + Child Pages":
        render_confluence_bulk_upload()


# ---------------------------------------------------------
# URL Upload (Sidebar)
# ---------------------------------------------------------

def render_confluence_url_upload():
    """Render interface for single or multiple Confluence URLs"""

    target_squad = _render_squad_selection_confluence("confluence_url")

    confluence_urls = st.sidebar.text_area(
        "URLs",
        placeholder="Enter one or more URLs above to enable the Process Pages button",
        key="confluence_urls",
        help=(
            "Paste each Confluence URL on a separate line. "
            "Multiple URLs are supported."
        ),
    )

    if confluence_urls:
        url_count = len(
            [
                u.strip()
                for u in confluence_urls.strip().split("\n")
                if u.strip()
            ]
        )
        if url_count > 1:
            st.sidebar.success(
                f"{url_count} URLs detected – ready for batch processing"
            )
        elif url_count == 1:
            st.sidebar.info("1 URL detected")
    else:
        st.sidebar.warning("No valid URLs found")

    include_images = st.sidebar.checkbox(
        "Analyze images with AI",
        value=True,
        help="Use Gemini AI to analyze diagrams and images in pages",
    )

    confluence_token_input = st.sidebar.text_input(
        "Confluence API Token (optional)",
        value="",
        type="password",
        key="confluence_api_token_input",
    )

    process_button_disabled = not confluence_urls.strip()

    if st.sidebar.button(
        "Process Pages",
        key="fetch_confluence_urls",
        disabled=process_button_disabled,
    ):
        services = get_services()

        try:
            if confluence_token_input.strip():
                services.confluence.set_api_token(
                    confluence_token_input.strip()
                )
        except Exception:
            pass

        process_confluence_urls(confluence_urls, include_images)


# ---------------------------------------------------------
# Bulk Upload (Sidebar)
# ---------------------------------------------------------

def render_confluence_bulk_upload():
    """Render interface for bulk parent + child page upload"""

    st.sidebar.markdown("**Parent + Child Pages Upload**")

    parent_url = st.sidebar.text_input(
        "Parent Page URL",
        placeholder=(
            "Enter one or more URLs above to enable the Process Pages button"
        ),
        key="parent_confluence_url",
    )

    page_pattern = st.sidebar.text_input(
        "Page Name Pattern",
        value="*",
        key="confluence_pattern",
        help=(
            "Filter pattern (* = all pages, Design* = pages starting with 'Design'). "
            "Applies to both parent and child pages"
        ),
    )

    include_images = st.sidebar.checkbox(
        "Analyze images with AI",
        value=True,
        key="bulk_include_images",
    )

    with st.sidebar.expander("Advanced Options"):
        max_pages = st.number_input(
            "Max pages to process",
            min_value=1,
            max_value=100,
            value=50,
            key="max_pages",
        )

    process_button_disabled = not parent_url.strip()

    if st.sidebar.button(
        "Process Parent + Child Pages",
        key="fetch_confluence_bulk",
        disabled=process_button_disabled,
    ):
        services = get_services()

        try:
            global_token = st.session_state.get(
                "confluence_api_token_input_global"
            )
            if global_token and global_token.strip():
                services.confluence.set_api_token(global_token.strip())
        except Exception:
            pass

        process_confluence_bulk(
            parent_url,
            page_pattern,
            include_images,
            max_pages,
        )


# ---------------------------------------------------------
# URL Processing
# ---------------------------------------------------------

def process_confluence_urls(
    urls_text: str, include_images: bool = True
):
    """Process multiple Confluence URLs from text input."""

    show_progress = True
    enable_dedup = True

    services = get_services()
    if not services.is_initialized():
        st.error("Services not initialized")
        return

    urls = [
        url.strip()
        for url in urls_text.strip().split("\n")
        if url.strip()
    ]

    if not urls:
        st.error("No valid URLs found")
        return

    total_chunks = 0
    processed_pages = 0
    failed_pages = 0
    total_deduplicated = 0
    all_pages_data = []

    try:
        for i, url in enumerate(urls):
            try:
                with st.spinner(
                    f"Processing page {i+1}/{len(urls)}..."
                ):
                    if re.search(
                        r"/spaces/[^/]+/overview",
                        url,
                        re.IGNORECASE,
                    ):
                        raise ValueError(
                            f"Space overview URL detected: {url}"
                        )

                    page_data = (
                        services.confluence.fetch_confluence_simple(
                            url, include_images
                        )
                    )

                existing_count = 0
                if enable_dedup:
                    existing_count = (
                        services.vector_store.delete_documents(
                            doc_type=DocumentType.KNOWLEDGE,
                            source_key=page_data["meta"]["url"],
                        )
                    )

                chunks = services.chunker.chunk_by_type(
                    page_data["text"], content_type="confluence"
                )

                base_metadata = {
                    "source": page_data["meta"]["url"],
                    "title": page_data["meta"]["title"],
                    "page_id": page_data["meta"]["id"],
                    "has_images": page_data["meta"]["has_images"],
                    "include_images": include_images,
                }

                metadatas = [
                    {**base_metadata, "chunk_id": j}
                    for j in range(len(chunks))
                ]

                services.vector_store.upsert_documents(
                    doc_type=DocumentType.KNOWLEDGE,
                    source_key=page_data["meta"]["url"],
                    documents=chunks,
                    metadatas=metadatas,
                )

                try:
                    trigger_reseed_for_upload(services)
                except Exception:
                    pass

                total_chunks += len(chunks)
                processed_pages += 1
                total_deduplicated += existing_count
                all_pages_data.append(page_data)

                if show_progress:
                    st.info(
                        f"Page {i+1}: {len(chunks)} chunks processed"
                    )

            except Exception as e:
                st.error(
                    f"Failed to process URL {i+1}: {str(e)}"
                )
                failed_pages += 1
                continue

        if processed_pages > 0:
            st.success(
                f"Processed {processed_pages} pages, "
                f"{total_chunks} chunks stored"
            )

        if enable_dedup and total_deduplicated > 0:
            st.info(
                f"Removed {total_deduplicated} duplicate chunks"
            )

        if failed_pages > 0:
            st.warning(
                f"{failed_pages} pages failed to process"
            )

        _clear_kb_cache()

    except Exception as e:
        st.error(
            f"Error processing Confluence URLs: {str(e)}"
        )


# ---------------------------------------------------------
# Bulk Processing
# ---------------------------------------------------------

def process_confluence_bulk(
    parent_url: str,
    pattern: str,
    include_images: bool = True,
    max_pages: int = 50,
):
    """Process bulk parent + child pages from a Confluence parent page."""

    show_progress = True
    enable_dedup = True

    services = get_services()
    if not services.is_initialized():
        st.error("Services not initialized")
        return

    try:
        with st.spinner(
            f"Fetching parent + child pages matching '{pattern}' (max {max_pages})..."
        ):
            if re.search(
                r"/spaces/[^/]+/overview",
                parent_url,
                re.IGNORECASE,
            ):
                st.error(
                    "Detected a Confluence Space Overview URL for the parent. "
                    "Please provide a direct parent page URL."
                )
                return

            pages_data = (
                services.confluence.fetch_confluence_bulk(
                    parent_url, pattern, include_images
                )
            )

        if not pages_data:
            st.warning(
                f"No pages found matching pattern '{pattern}'"
            )
            return

        if len(pages_data) > max_pages:
            st.info(
                f"Found {len(pages_data)} pages, processing first {max_pages}"
            )
            pages_data = pages_data[:max_pages]

        total_chunks = 0
        processed_pages = 0
        pages_with_images = 0
        total_deduplicated = 0

        for i, page_data in enumerate(pages_data):
            try:
                with st.spinner(
                    f"Processing page {i+1}/{len(pages_data)}: "
                    f"{page_data['meta']['title']}..."
                ):
                    existing_count = 0
                    if enable_dedup:
                        existing_count = (
                            services.vector_store.delete_documents(
                                doc_type=DocumentType.KNOWLEDGE,
                                source_key=page_data["meta"]["url"],
                            )
                        )

                    chunks = services.chunker.chunk_by_type(
                        page_data["text"],
                        content_type="confluence",
                    )

                    base_metadata = {
                        "source": page_data["meta"]["url"],
                        "title": page_data["meta"]["title"],
                        "page_id": page_data["meta"]["id"],
                        "parent_id": page_data["meta"].get(
                            "parent_id"
                        ),
                        "has_images": page_data["meta"]["has_images"],
                        "include_images": include_images,
                        "pattern": pattern,
                    }

                    metadatas = [
                        {**base_metadata, "chunk_id": j}
                        for j in range(len(chunks))
                    ]

                    services.vector_store.upsert_documents(
                        doc_type=DocumentType.KNOWLEDGE,
                        source_key=page_data["meta"]["url"],
                        documents=chunks,
                        metadatas=metadatas,
                    )

                    try:
                        trigger_reseed_for_upload(services)
                    except Exception:
                        pass

                    total_chunks += len(chunks)
                    processed_pages += 1
                    total_deduplicated += existing_count

                    if page_data["meta"]["has_images"]:
                        pages_with_images += 1

                    if show_progress:
                        st.info(
                            f"Page {i+1}: {len(chunks)} chunks processed - "
                            f"{page_data['meta']['title']}"
                        )

            except Exception as e:
                st.error(
                    f"Failed to process page "
                    f"'{page_data['meta']['title']}': {str(e)}"
                )
                continue

        parent_count = len(
            [
                p
                for p in pages_data
                if p["meta"].get("is_parent", False)
            ]
        )
        child_count = processed_pages - parent_count

        if parent_count > 0 and child_count > 0:
            st.success(
                f"Processed {parent_count} parent + {child_count} child pages = "
                f"{processed_pages} total pages, {total_chunks} chunks stored"
            )
        elif parent_count > 0:
            st.success(
                f"Processed {parent_count} parent page, "
                f"{total_chunks} chunks stored"
            )
        else:
            st.success(
                f"Processed {child_count} child pages, "
                f"{total_chunks} chunks stored"
            )

        if include_images and pages_with_images > 0:
            st.info(
                f"AI image analysis performed on "
                f"{pages_with_images} pages with images"
            )

        if enable_dedup and total_deduplicated > 0:
            st.info(
                f"Removed {total_deduplicated} duplicate chunks"
            )

        _clear_kb_cache()

    except Exception as e:
        st.error(
            f"Error processing bulk Confluence pages: {str(e)}"
        )


# ---------------------------------------------------------
# Cache Clear
# ---------------------------------------------------------

def _clear_kb_cache():
    """Helper function to clear knowledge base cache"""

    for key in list(st.session_state.keys()):
        if key.startswith("kb_stats_") or key == "kb_stats_cache":
            del st.session_state[key]
