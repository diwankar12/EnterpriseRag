"""
Sidebar components for the application.

This module handles tribe selection, file uploads, and
configuration display in the sidebar.
"""

import streamlit as st
import tempfile
import subprocess
import shutil
import pandas as pd
import io
import logging
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any

from ..tribe_config import TRIBE_CONFIG
from ..app_config import get_services
from ..vector_store_interface import DocumentType

logger = logging.getLogger(__name__)

# reuse centralized reseed helper from knowledge_lake modules
try:
    from .knowledge_lake.confluence import trigger_reseed_for_upload
except Exception:
    try:
        from .knowledge_lake.pdf import trigger_reseed_for_upload
    except Exception:
        trigger_reseed_for_upload = None

def render_generation_config_section():
    """Render story generation configuration controls in sidebar"""

    st.sidebar.header("Story Generation Settings")

    with st.sidebar.expander("Search Configuration", expanded=False):
        st.markdown("**Document Retrieval Settings**")

        # Number of knowledge documents to retrieve
        k_docs = st.slider(
            "Knowledge Documents ",
            min_value=1,
            max_value=200,
            value=st.session_state.get("k_docs", 5),
            step=1,
            help="Number of knowledge documents to retrieve for context",
            key="k_docs"
        )

        # Number of code documents to retrieve
        k_code = st.slider(
            "Code Documents ",
            min_value=1,
            max_value=200,
            value=st.session_state.get("k_code", 3),
            step=1,
            help="Number of code documents to retrieve for examples",
            key="k_code"
        )

        # Temperature control
        temperature = st.slider(
            "AI Creativity Level",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("temperature", 0.15),
            step=0.05,
            help="Lower values = more focused, Higher values = more creative",
            key="temperature"
        )

        # Show current configuration
        st.markdown("---")
        st.markdown("**Current Settings:**")
        st.markdown(f"- Knowledge docs: {k_docs}")
        st.markdown(f"- Code docs: {k_code}")
        st.markdown(f"- Temperature: {temperature}")

        # Reset to defaults button
        if st.button("Reset to Defaults", help="Reset all values to defaults"):
            st.session_state.k_docs = 5
            st.session_state.k_code = 3
            st.session_state.temperature = 0.15
            st.rerun()

def render_memory_config_section():
    """Render conversation memory configuration controls in the sidebar."""

    st.sidebar.header("Conversation Memory")

    with st.sidebar.expander("Memory Configuration", expanded=False):
        st.markdown("**Memory Settings**")

        # 1. Number of previous turns
        memory_turns = st.slider(
            "Previous Turns to Include",
            min_value=0,
            max_value=20,
            value=st.session_state.get("memory_turns", 10),
            step=1,
            help="Number of previous conversation turns to include in the prompt. 0 for none.",
            key="memory_turns",
        )

        # 2. Selective history
        memory_include = st.radio(
            "Include in History:",
            ("User and Assistant", "User Only", "Assistant Only"),
            index=st.session_state.get("memory_include_index", 0),
            help="Choose which parts of the conversation to include.",
            key="memory_include",
        )

        # Map radio choice to a more usable value and store index
        if memory_include == "User and Assistant":
            st.session_state.memory_include_index = 0
            st.session_state.memory_include_value = "all"
        elif memory_include == "User Only":
            st.session_state.memory_include_index = 1
            st.session_state.memory_include_value = "user"
        else:
            st.session_state.memory_include_index = 2
            st.session_state.memory_include_value = "assistant"

        # 3. Summarization
        st.markdown("**Summarization**")
        enable_summarization = st.checkbox(
            "Enable History Summarization",
            value=st.session_state.get("enable_summarization", False),
            help="Summarize long conversation histories to save tokens.",
            key="enable_summarization",
        )

        if enable_summarization:
            summarization_threshold = st.slider(
                "Summarization Threshold (tokens)",
                min_value=500,
                max_value=4000,
                value=st.session_state.get("summarization_threshold", 2000),
                step=100,
                help="If the history exceeds this token count, it will be summarized.",
                key="summarization_threshold",
            )

        # 5. Filtering
        st.markdown("**Filtering**")
        enable_filtering = st.checkbox(
            "Filter Repetitive Exchanges",
            value=st.session_state.get("enable_filtering", False),
            help="Filter out repetitive questions or answers from the history.",
            key="enable_filtering",
        )

        # Display current settings
        st.markdown("---")
        st.markdown("**Current Memory Settings:**")
        st.markdown(f"- Previous Turns: {memory_turns}")
        st.markdown(f"- Include: {memory_include}")

        if enable_summarization:
            st.markdown(
                f"- Summarization: Enabled "
                f"(Threshold: {st.session_state.get('summarization_threshold', 2000)} tokens)"
            )
        else:
            st.markdown("- Summarization: Disabled")

        if enable_filtering:
            st.markdown("- Filtering: Enabled")
        else:
            st.markdown("- Filtering: Disabled")

        # Reset to defaults button
        if st.button(
            "Reset Memory Defaults",
            help="Reset memory settings to defaults",
        ):
            st.session_state.memory_turns = 10
            st.session_state.memory_include_index = 0
            st.session_state.memory_include_value = "all"
            st.session_state.memory_include = "User and Assistant"
            st.session_state.enable_summarization = False
            st.session_state.summarization_threshold = 2000
            st.session_state.enable_filtering = False
            st.rerun()

def render_tribe_selector() -> str:
    """
    Render tribe selection dropdown in sidebar and store lowercase key alias.
    Filters tribes based on user permissions from SSO.

    Returns:
        str: Selected tribe name
    """

    # Import authorization service for permission filtering
    try:
        from src.services.authorization_service import AuthorizationService
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
        from services.authorization_service import AuthorizationService

    # Neutral styling – removed Equifax red branding
    st.sidebar.markdown("### Knowledge Corpus Selection")

    # Get user permissions from session
    user_permissions = None
    if 'user-logged-in' in st.session_state:
        auth_result = st.session_state['user-logged-in']
        user_permissions = getattr(auth_result, 'permissions', None)

    # Build options using the human-friendly display_name from TRIBE_CONFIG
    tribe_items = list(TRIBE_CONFIG.items())
    all_tribe_keys = [key for key, cfg in tribe_items]

    # Filter tribes based on permissions
    if user_permissions:
        accessible_tribes = AuthorizationService.get_filtered_tribes(
            user_permissions,
            all_tribe_keys
        )

        # Filter tribe items to only include accessible tribes
        # Note: If user has no accessible tribes, this is now handled at the app level
        # in main_app.py before the sidebar is rendered
        tribe_items = [
            (key, cfg)
            for key, cfg in tribe_items
            if key in accessible_tribes
        ]

    display_options = [
        cfg.get("display_name", key.title())
        for key, cfg in tribe_items
    ]

    # Determine default display (first entry unless session has a preferred key)
    default_key = None
    if 'tribe_selector_key' in st.session_state:
        default_key = st.session_state.get('tribe_selector_key')

        # Ensure the stored key is in accessible tribes
        if user_permissions and default_key not in [key for key, _ in tribe_items]:
            default_key = tribe_items[0][0] if tribe_items else "mortgage"
    else:
        # Fall back to the first configured tribe key
        default_key = tribe_items[0][0] if tribe_items else "mortgage"

    default_display = (
        TRIBE_CONFIG.get(default_key, {}).get("display_name")
        or default_key.title()
    )

    # Use a dedicated session key for the UI display value
    selected_display = st.sidebar.selectbox(
        "Select your Knowledge Corpus:",
        options=display_options,
        index=display_options.index(default_display)
        if default_display in display_options else 0,
        help="This determines which knowledge base and embeddings will be used",
        key="tribe_selector_display",
    )

    # Map selected display name back to canonical tribe key
    selected_tribe = None
    for key, cfg in TRIBE_CONFIG.items():
        if cfg.get("display_name") == selected_display:
            selected_tribe = key
            break

    if not selected_tribe:
        # Last-resort: try title-case match or fall back to default_key
        title_map = {k.title(): k for k in TRIBE_CONFIG.keys()}
        selected_tribe = title_map.get(selected_display, default_key)

    # Preserve legacy short selector and canonical key
    st.session_state["tribe_selector"] = selected_tribe.title()
    st.session_state["tribe_selector_key"] = selected_tribe

    tribe_config = TRIBE_CONFIG[selected_tribe]

    # Show permission badge for selected tribe
    if user_permissions:
        _show_permission_badge(user_permissions, selected_tribe)

    return selected_tribe

def render_file_upload_section():
    """Render file upload section in sidebar"""

    st.sidebar.header("Knowledge Management")

    # Initialize session state for upload type if not exists
    if "preferred_upload_type" not in st.session_state:
        st.session_state.preferred_upload_type = "Confluence Pages"  # Default to Confluence

    # File upload options with simplified session state management
    upload_type = st.sidebar.radio(
        "Upload Type:",
        [
            "Documents & Images",
            "Confluence Pages",
            "Spreadsheets",
            "Clone GitHub repo and ingest",
            "Jira Stories",
        ],
        key="upload_type_selector",
        help="Choose what type of content to upload",
    )

    # Store the selection in session state
    st.session_state.preferred_upload_type = upload_type

    if upload_type == "Documents & Images":
        render_pdf_upload()

    elif upload_type == "Confluence Pages":
        render_confluence_upload()

    elif upload_type == "Spreadsheets":
        from .knowledge_lake.spreadsheets import render_spreadsheet_upload_sidebar
        render_spreadsheet_upload_sidebar()

    elif upload_type == "Clone GitHub repo and ingest":
        render_github_clone()

    elif upload_type == "Jira Stories":
        render_jira_stories_upload_sidebar()

def _show_permission_badge(user_permissions, selected_tribe):
    """
    Display a badge showing the user's permission level for the selected tribe.

    Args:
        user_permissions: UserPermissions object
        selected_tribe: Currently selected tribe name
    """
    pass


def render_jira_stories_upload_sidebar():
    """Render Jira stories upload interface for sidebar"""
    from .knowledge_lake.jira_import import render_jira_stories_import
    render_jira_stories_import()


def render_pdf_upload():
    """Render PDF and Image upload interface with squad support"""
    from .knowledge_lake.pdf import render_pdf_upload_sidebar
    render_pdf_upload_sidebar()


def render_confluence_upload():
    """Render enhanced Confluence upload interface with squad support"""
    from .knowledge_lake.confluence import (
        render_confluence_upload as render_confluence_upload_component
    )
    render_confluence_upload_component()


def render_confluence_url_upload():
    """Render interface for single or multiple Confluence URLs"""

    # Add squad selection for Confluence flows (ensures normalized token is available)
    from .knowledge_lake.confluence import _render_squad_selection_confluence
    target_squad = _render_squad_selection_confluence("confluence_url")

    # Text area for multiple URLs with clearer instructions
    confluence_urls = st.sidebar.text_area(
        "URLs",
        placeholder="Enter one or more URLs above to enable the Process Pages button",
        key="confluence_urls",
        help=(
            "Paste each Confluence URL on a separate line. "
            "Multiple URLs are supported. Use the Process Pages button below to start processing."
        ),
    )

    # URL count display
    if confluence_urls:
        url_count = len(
            [
                url.strip()
                for url in confluence_urls.strip().split("\n")
                if url.strip()
            ]
        )
        if url_count > 1:
            st.sidebar.success(
                f"{url_count} URLs detected - ready for batch processing!"
            )
        elif url_count == 1:
            st.sidebar.info("1 URL detected")
        else:
            st.sidebar.warning("No valid URLs found")

    # Options for image processing
    include_images = st.sidebar.checkbox(
        "Analyze images with AI",
        value=True,
        help=(
            "Use Gemini AI to analyze diagrams and images in pages "
            "for better context understanding"
        ),
    )

    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        chunk_size = st.selectbox(
            "Chunk size:",
            ["Default", "Small (better for code)", "Large (better for docs)"],
            help="Choose chunk size for text processing",
        )

    # Basic URL validation: detect space overview URLs that lack a page id
    import re

    urls_list = [
        u.strip()
        for u in (confluence_urls or "").split("\n")
        if u.strip()
    ]
    has_space_overview = any(
        re.search(r"/spaces/[^/]+/overview", u, re.IGNORECASE)
        for u in urls_list
    )

    # Diagnostic banner for debugging disabled buttons; do not depend on role
    user_info = st.session_state.get("user")
    if user_info:
        with st.sidebar.expander("Diagnostics", expanded=False):
            st.write(f"user: {user_info.get('username', 'unknown')}")
            st.write(
                f"confluence_urls present: "
                f"{bool(confluence_urls and confluence_urls.strip())}"
            )
            st.write(
                "normalized squad (url): "
                f"{st.session_state.get('squad_selection_confluence_url_normalized')}"
            )

    # Process button – Always visible with helpful state feedback
    process_button_disabled = (
        not confluence_urls.strip() or has_space_overview
    )

    if has_space_overview:
        st.sidebar.warning(
            "Detected Confluence Space Overview URL(s). "
            "Please provide a direct page URL or page id "
            "(a URL containing /pages/ or ?pageId=)."
        )

    if st.sidebar.button(
        "Process Pages",
        key="fetch_confluence_urls",
        disabled=process_button_disabled,
    ):
        # Resolve normalized token if available in session state (fallback to target_squad)
        normalized = (
            st.session_state.get(
                "squad_selection_confluence_url_normalized"
            )
            or target_squad
        )
        process_confluence_urls(
            confluence_urls,
            include_images,
            normalized,
        )

def render_confluence_bulk_upload():
    """Render interface for bulk parent + child page upload with selection."""

    st.sidebar.markdown("**Parent + Child Pages Upload**")

    # Initialize session state
    if "bulk_confluence_pages" not in st.session_state:
        st.session_state.bulk_confluence_pages = []
    if "bulk_confluence_selections" not in st.session_state:
        st.session_state.bulk_confluence_selections = {}

    parent_url = st.sidebar.text_input(
        "Parent Page URL:",
        key="parent_confluence_url",
        help="URL of the parent page to start the search from.",
    )

    page_pattern = st.sidebar.text_input(
        "Page Name Pattern:",
        value="*",
        key="confluence_pattern",
        help="Filter pages by title (e.g., 'Design*', '*Spec*').",
    )

    include_images = st.sidebar.checkbox(
        "Analyze images with AI",
        value=True,
        key="bulk_include_images",
        help="Use AI to analyze images in the selected pages.",
    )

    # Detect space overview parent URLs which lack a page id
    has_space_overview_parent = False
    try:
        if parent_url and re.search(
            r"/spaces/[^/]+/overview", parent_url, re.IGNORECASE
        ):
            has_space_overview_parent = True
    except Exception:
        has_space_overview_parent = False

    if has_space_overview_parent:
        st.sidebar.warning(
            "Detected a Confluence Space Overview URL for the parent. "
            "Please provide a direct parent page URL."
        )

    if st.sidebar.button(
        "Preview Pages to Process",
        key="preview_confluence_bulk",
        disabled=not parent_url.strip() or has_space_overview_parent,
    ):
        services = get_services()
        if services.is_initialized():
            try:
                with st.spinner("Collecting descendant pages..."):
                    # Fetch without image analysis for speed
                    page_list = services.confluence.fetch_confluence_bulk(
                        parent_url,
                        page_pattern,
                        include_images=False,
                    )

                st.session_state.bulk_confluence_pages = page_list

                # Initialize selections, defaulting to True
                st.session_state.bulk_confluence_selections = {
                    p["meta"]["id"]: True for p in page_list
                }

                if not page_list:
                    st.sidebar.warning(
                        f"No pages found matching pattern '{page_pattern}'"
                    )

            except Exception as e:
                st.sidebar.error(f"Error collecting pages: {e}")
                st.session_state.bulk_confluence_pages = []
        else:
            st.sidebar.error("Services not initialized")

    if st.session_state.bulk_confluence_pages:
        pages = st.session_state.bulk_confluence_pages
        st.sidebar.success(
            f"Found {len(pages)} pages. Select pages to process."
        )

        # --- Selection Logic ---
        selections = st.session_state.bulk_confluence_selections

        # Select/Deselect All
        all_selected = all(
            selections.get(p["meta"]["id"], False) for p in pages
        )
        select_all_key = "select_all_bulk_confluence"
        select_all_state = st.sidebar.checkbox(
            "Select All",
            value=all_selected,
            key=select_all_key,
        )

        # If 'Select All' is changed, update all other checkboxes
        if select_all_state != all_selected:
            for p in pages:
                selections[p["meta"]["id"]] = select_all_state
            st.rerun()

        # Display checkboxes for each page
        for page in pages:
            page_id = page["meta"]["id"]
            page_title = page["meta"]["title"]
            selections[page_id] = st.sidebar.checkbox(
                page_title,
                value=selections.get(page_id, True),
                key=f"confluence_page_{page_id}",
            )

        selected_pages_data = [
            p for p in pages if selections.get(p["meta"]["id"])
        ]
        selected_urls = [p["meta"]["url"] for p in selected_pages_data]

        st.sidebar.info(
            f"{len(selected_urls)} of {len(pages)} pages selected."
        )

        from .knowledge_lake.confluence import _render_squad_selection_confluence

        # Ensure a normalized squad token exists for bulk flows as well
        bulk_target_squad = st.session_state.get(
            "squad_selection_confluence_bulk_normalized"
        )
        if not bulk_target_squad:
            bulk_target_squad = _render_squad_selection_confluence(
                "confluence_bulk"
            )

        process_button_disabled = not selected_urls
        if st.sidebar.button(
            "Process Selected Pages",
            key="fetch_confluence_bulk",
            disabled=process_button_disabled,
        ):
            urls_text = "\n".join(selected_urls)
            # Use the existing function for processing URLs and pass the target squad
            process_confluence_urls(
                urls_text,
                include_images,
                bulk_target_squad,
            )


def render_multiple_repo_upload():
    """Render interface for multiple repository upload"""

    st.sidebar.markdown("**Multiple Repositories**")

    repo_urls = st.sidebar.text_area(
        "Repository URLs:",
        placeholder=(
            "Paste one or more URLs (one per line):\n"
            "https://github.com/owner/repo1\n"
            "https://github.com/owner/repo2"
        ),
        height=100,
        help="Enter one repository URL per line",
    )

    github_token = st.sidebar.text_input(
        "GitHub Token:",
        type="password",
        help="Enter your GitHub token with repo access for all repositories.",
        key="multi_repo_github_token",
    )

    # Batch processing options
    include_docs = st.sidebar.checkbox(
        "Include documentation",
        value=True,
        key="batch_include_docs",
        help="Process .md, .rst, .txt files as knowledge documents",
    )

    # Advanced options for batch processing
    with st.sidebar.expander("Advanced Options"):
        max_repos = st.number_input(
            "Max repositories:",
            min_value=1,
            max_value=20,
            value=10,
            help="Limit the number of repositories to process",
        )

        max_files_per_repo = st.number_input(
            "Max files per repo:",
            min_value=10,
            max_value=500,
            value=200,
            help="Limit files processed per repository",
        )

    if repo_urls and github_token and st.sidebar.button(
        "Clone & Ingest All Repositories",
        key="clone_multiple_repos",
    ):
        process_multiple_repositories(
            repo_urls,
            github_token,
            include_docs,
            max_repos,
            max_files_per_repo,
        )

def render_single_repo_upload():
    """Render interface for single repository upload"""

    st.sidebar.markdown("**Single Repository**")

    repo_url = st.sidebar.text_input(
        "GitHub Repository URL:",
        placeholder="https://github.com/owner/repo",
        help="Enter the full GitHub repository URL",
    )

    github_token = st.sidebar.text_input(
        "GitHub Token:",
        type="password",
        help="Enter your GitHub token with repo access.",
    )

    branch = st.sidebar.text_input(
        "Branch (optional):",
        placeholder="main",
        help="Leave empty to use default branch",
    )

    # Repository processing options
    include_docs = st.sidebar.checkbox(
        "Include documentation",
        value=True,
        help="Process .md, .rst, .txt files as knowledge documents",
    )

    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        max_files = st.number_input(
            "Max files to process:",
            min_value=10,
            max_value=1000,
            value=500,
            help="Limit the number of files to process",
        )

        exclude_patterns = st.text_area(
            "Exclude patterns:",
            value="test,tests,__tests__,spec,specs,node_modules,.git",
            help="Comma-separated patterns to exclude",
        )

    if repo_url and github_token and st.sidebar.button(
        "Clone & Ingest Repository",
        key="clone_single_repo",
    ):
        process_single_repository(
            repo_url,
            github_token,
            branch,
            include_docs,
            max_files,
            exclude_patterns,
        )

def render_github_clone():
    """Render enhanced GitHub repository cloning interface"""

    st.sidebar.subheader("GitHub Repository Upload")

    # Initialize session state for repo type if not exists
    if "preferred_repo_type" not in st.session_state:
        st.session_state.preferred_repo_type = "Single Repository"

    # Repository type selection with simplified session state management
    repo_type = st.sidebar.radio(
        "Repository Upload Type:",
        ["Single Repository", "Multiple Repositories"],
        key="github_repo_type",
        help="Choose how to clone and ingest repositories",
    )

    # Store the selection in session state
    st.session_state.preferred_repo_type = repo_type

    if repo_type == "Single Repository":
        render_single_repo_upload()
    elif repo_type == "Multiple Repositories":
        render_multiple_repo_upload()
