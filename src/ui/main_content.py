"""
Main content components for story generation and management.

This module handles the primary user interface for creating,
editing, and managing Jira stories.
"""

import streamlit as st
import json
import re
import time
from typing import Dict, Any, Optional
from datetime import datetime
from urllib.parse import urlparse

from src.ui.story.generator import (
    generate_epic_optimized,
    generate_single_story,
)

# Use centralized story generation functions
from src.app_config import get_services, JIRA_PROJECT_CONFIG
from src.agent import StoryDraft


# ============================================================
# Story Generation Section
# ============================================================

def render_story_generation_section():
    """Render the main story generation interface"""

    # Initialize expander state
    if "story_settings_expanded" not in st.session_state:
        st.session_state.story_settings_expanded = False

    col_settings, col_toggle = st.columns([4, 1])

    with col_toggle:
        expanded_state = st.checkbox(
            "Show Settings",
            value=st.session_state.story_settings_expanded,
            key="settings_toggle",
            help="Toggle Story Generation Settings",
        )
        st.session_state.story_settings_expanded = expanded_state

    with col_settings:
        st.markdown("### Story Generation Settings")

        if st.session_state.story_settings_expanded:
            st.markdown("**Document Retrieval & AI Configuration**")

            col1, col2, col3 = st.columns(3)

            with col1:
                k_docs = st.slider(
                    "Knowledge Documents (k)",
                    min_value=1,
                    max_value=600,
                    value=st.session_state.get("k_docs", 200),
                    step=1,
                    help="Number of knowledge documents to retrieve for context",
                    key="k_docs",
                )

            with col2:
                k_code = st.slider(
                    "Code Documents (k)",
                    min_value=1,
                    max_value=600,
                    value=st.session_state.get("k_code", 200),
                    step=1,
                    help="Number of code documents to retrieve for examples",
                    key="k_code",
                )

            with col3:
                temperature = st.slider(
                    "AI Creativity Level",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get("temperature", 0.05),
                    step=0.05,
                    help="Lower = focused, Higher = creative",
                    key="temperature",
                )

            col_info, col_reset = st.columns([3, 1])

            with col_info:
                st.markdown(
                    f"**Current:** {k_docs} knowledge docs • "
                    f"{k_code} code docs • "
                    f"{temperature:.2f} creativity"
                )

            with col_reset:
                if st.button(
                    "Reset to Defaults",
                    help="Reset all values to defaults",
                    key="reset_config",
                ):
                    for key in ["k_docs", "k_code", "temperature"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.session_state.story_settings_expanded = True
                    st.session_state.preserve_tab = True
                    st.rerun()

        else:
            k_docs = st.session_state.get("k_docs", 200)
            k_code = st.session_state.get("k_code", 200)
            temperature = st.session_state.get("temperature", 0.15)

    # --------------------------------------------------------
    # Story Input
    # --------------------------------------------------------

    with st.container():
        st.markdown("### Create Your Story")

        col1, col2 = st.columns([3, 1])

        with col1:
            user_story = st.text_area(
                "Describe your user story or feature request:",
                placeholder=(
                    "e.g., As a user, I want to be able to reset my password "
                    "so that I can regain access to my account"
                ),
                height=100,
                key="user_story_input",
            )

        with col2:
            include_code = st.checkbox(
                "Include code examples",
                value=False,
                help="Include relevant code snippets in the generated story",
            )

            code_language = st.selectbox(
                "Code language:",
                ["Java", "JavaScript", "Python", "C#", "Go", "Rust"],
                disabled=not include_code,
                key="code_language",
            )

    # --------------------------------------------------------
    # Squad validation
    # --------------------------------------------------------

    from src.app_config import tribe_supports_squads

    current_tribe = st.session_state.get("tribe_selector", "mortgage")
    selected_squads = st.session_state.get("selected_squads", [])

    squad_required = tribe_supports_squads(current_tribe)
    squad_missing = squad_required and not selected_squads

    if squad_missing:
        st.warning(
            "⚠️ **Squad Selection Required:** "
            "Please select squads in the sidebar before generating stories."
        )

    button_disabled = lambda: not user_story.strip() or squad_missing

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "Generate Single Story",
            type="primary",
            disabled=button_disabled(),
            use_container_width=True,
        ):
            generate_single_story(
                user_story,
                include_code,
                code_language.lower() if include_code else None,
                temperature,
                k_docs,
                k_code,
            )

    with col2:
        if st.button(
            "Generate Epic",
            type="secondary",
            disabled=button_disabled(),
            use_container_width=True,
        ):
            generate_epic_optimized(
                user_story,
                include_code,
                code_language.lower() if include_code else None,
                temperature,
                k_docs,
                k_code,
            )

    render_story_display_section()


# ============================================================
# Story Display Section
# ============================================================

def render_story_display_section():
    """Render the generated story display and editing interface"""

    if "draft" not in st.session_state:
        return

    current_tribe = st.session_state.get("tribe_selector", "mortgage")
    stored_tribe = st.session_state.get("content_tribe")

    if stored_tribe and stored_tribe != current_tribe:
        st.warning(
            f"⚠️ Content was generated for {stored_tribe.title()} tribe but "
            f"you're now in {current_tribe.title()} tribe. Clearing content."
        )
        for key in [
            "draft",
            "context",
            "generation_timestamp",
            "substories",
            "content_tribe",
        ]:
            st.session_state.pop(key, None)
        st.rerun()
        return

    draft = st.session_state.draft
    is_epic = st.session_state.get("is_epic", False)

    st.header("Generated Epic" if is_epic else "Generated Story")

    if is_epic:
        st.success("**Epic Created** – High-level epic with multiple stories")
    else:
        st.info("**Story Created** – Single user story")

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    st.subheader("Title")
    new_title = st.text_input(
        "Title",
        value=draft.title,
        key="edit_title",
        help="Edit the story title",
    )

    # --------------------------------------------------------
    # Description
    # --------------------------------------------------------

    st.subheader("Description")
    clean_description, code_blocks = extract_code_blocks(draft.description)

    new_description = st.text_area(
        "Description",
        value=clean_description,
        height=200,
        key="edit_description",
        help="Edit the story description",
    )

    if code_blocks:
        st.subheader("Code Examples")
        with st.expander("Code Examples (Click to view)", expanded=True):
            for i, block in enumerate(code_blocks):
                st.markdown(f"**Example {i + 1} ({block['language']}):**")
                st.code(block["code"], language=block["language"])

    # --------------------------------------------------------
    # RAG Context
    # --------------------------------------------------------

    context = st.session_state.get("context")
    if context:
        services = get_services()
        context_text = services.agentic_rag.context_to_text(context)

        with st.expander("Show RAG References / Source Context", expanded=False):
            st.markdown(
                f"**The following context was retrieved from your "
                f"{current_tribe.title()} knowledge base:**"
            )
            st.markdown(
                f"<div style='font-size:13px; white-space:pre-wrap;'>"
                f"{context_text}</div>",
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # Acceptance Criteria
    # --------------------------------------------------------

    st.subheader("Acceptance Criteria")

    if draft.acceptance_criteria:
        for i, criterion in enumerate(draft.acceptance_criteria):
            st.markdown(f"**{i + 1}.** {criterion}")

    new_criterion = st.text_input(
        "Acceptance Criteria:",
        placeholder="Given [context]... When [action]... Then [expected outcome]...",
        key="new_criterion",
    )

    if st.button("Add Criterion", disabled=not new_criterion.strip()):
        add_acceptance_criterion(new_criterion)
        st.rerun()

    # --------------------------------------------------------
    # Update Draft
    # --------------------------------------------------------

    if new_title != draft.title or new_description != clean_description:
        updated_description = reconstruct_text_with_code_blocks(
            new_description, code_blocks
        )
        update_story_draft(new_title, updated_description)

    render_substories_section()
    render_feedback_section()

    if st.session_state.get("story_approved", False):
        render_jira_integration_section()


# ============================================================
# Substories Section
# ============================================================

def render_substories_section():
    """Render substories as compact buttons"""

    if "substories" not in st.session_state or not st.session_state.substories:
        return

    is_epic = st.session_state.get("is_epic", False)

    if is_epic:
        st.header("Epic Stories")
        st.info(
            f"This Epic contains {len(st.session_state.substories)} stories "
            "that work together to deliver functionality."
        )
    else:
        st.header("Related Sub-Stories")
        st.info(
            f"Found {len(st.session_state.substories)} related sub-stories."
        )

    st.markdown("**Click to view full details:**")

    for index, substory in enumerate(st.session_state.substories):
        title = substory.get("title", f"Untitled Story {index + 1}")
        label = (
            f"Story {index + 1}: {title}"
            if is_epic
            else f"Sub-Story {index + 1}: {title}"
        )

        if st.button(
            label,
            key=f"substory_button_{index}",
            use_container_width=False,
        ):
            st.session_state["modal_substory_index"] = index
            render_substory_modal()

    reference_type = "story" if is_epic else "sub story"
    st.info(
        f"**For AI Feedback:** You can reference these as "
        f"'{reference_type} 1', '{reference_type} 2', etc."
    )
# ============================================================
# Sub-Story Modal (Dialog)
# ============================================================

@st.dialog("Story Details")
def render_substory_modal():
    """Render modal popup for substory details using st.dialog"""

    modal_index = st.session_state.get("modal_substory_index", 0)

    if modal_index >= len(st.session_state.substories):
        st.error("Invalid story index")
        return

    substory = st.session_state.substories[modal_index]
    is_epic = st.session_state.get("is_epic", False)

    story_type = "Story" if is_epic else "Sub-Story"

    st.markdown(
        f"### {story_type} {modal_index + 1}: "
        f"{substory.get('title', 'Untitled')}"
    )

    # Description
    st.markdown("**Description:**")
    st.markdown(substory.get("description", "No description provided"))

    # Acceptance Criteria
    st.markdown("**Acceptance Criteria:**")
    criteria = substory.get("acceptance_criteria", [])

    if criteria:
        for i, criterion in enumerate(criteria, 1):
            st.markdown(f"{i}. {criterion}")
    else:
        st.markdown("*No acceptance criteria provided*")

    # Code Examples
    if substory.get("code_examples"):
        st.markdown("**Code Examples:**")

        for i, code_example in enumerate(substory["code_examples"]):
            if isinstance(code_example, dict):
                language = code_example.get("language", "text")
                code_content = code_example.get("code", "")
            else:
                language = "text"
                code_content = str(code_example) if code_example else ""

            with st.expander(f"Example {i + 1} ({language})", expanded=False):
                st.code(code_content, language=language)

    # Copyable text
    st.markdown(f"**Copy {story_type} Details:**")

    substory_text = f"""
**{story_type} {modal_index + 1}: {substory.get('title', 'Untitled')}**

**Description:**
{substory.get('description', 'No description provided')}

**Acceptance Criteria:**
{chr(10).join([f"{i+1}. {c}" for i, c in enumerate(criteria)])}
"""

    if substory.get("code_examples"):
        substory_text += "\n**Code Examples:**\n"
        for i, code_example in enumerate(substory["code_examples"]):
            substory_text += (
                f"\nExample {i + 1} "
                f"({code_example.get('language', 'text')}):\n"
                f"{code_example.get('code', '')}\n"
            )

    st.text_area(
        "Copy this text:",
        value=substory_text,
        height=200,
        key=f"modal_copy_text_{modal_index}",
    )

    reference_type = "story" if is_epic else "sub story"
    st.info(
        f"**For AI Assistant:** You can refer to this as "
        f"'{reference_type} {modal_index + 1}' when giving feedback."
    )


# ============================================================
# Feedback Section
# ============================================================

def render_feedback_section():
    """Render the feedback and improvement interface"""

    if "draft" not in st.session_state:
        return

    st.header("Story Feedback & Improvement")

    st.info(
        """
**How to improve your story:**

**AI Assistant**
- AI decides whether to approve, edit, regenerate, or reject
- Best for general feedback like “add more details”

**Apply Edit**
- Makes targeted edits based on your exact instructions
- Best for specific requests like “change title”

**Force Regenerate**
- Completely regenerates the story
- Best for major changes
"""
    )

    services = get_services()

    feedback = st.text_area(
        "Tell us what you think about the story:",
        placeholder=(
            "Examples: 'Add more technical details', "
            "'Acceptance criteria are too vague', "
            "'This looks perfect!'"
        ),
        height=100,
        key="feedback_input",
    )

    # AI decision transparency
    if feedback.strip() and services.feedback_agent:
        if st.checkbox("Show AI Decision Process"):
            with st.expander("AI Analysis Details", expanded=True):
                st.markdown("**Feedback Agent Configuration:**")
                st.json(
                    {
                        "model": services.feedback_agent.model,
                        "max_retries": services.feedback_agent.max_retries,
                        "timeout": services.feedback_agent.timeout,
                    }
                )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button(
            "Approve Story",
            type="primary",
            help="Approve this story and proceed to Jira integration",
        ):
            st.session_state["story_approved"] = True
            st.success("Story approved! Ready for Jira ticket creation")

    with col2:
        if st.button(
            "AI Assistant",
            disabled=not feedback.strip(),
            help="Let AI decide the best action",
        ):
            process_smart_feedback(feedback)

    with col3:
        if st.button(
            "Apply Edit",
            disabled=not feedback.strip(),
            help="Apply specific edits",
        ):
            process_edit_feedback(feedback)

    with col4:
        if st.button(
            "Force Regenerate",
            disabled=not feedback.strip(),
            help="Always regenerate the story",
        ):
            process_regenerate_feedback(feedback)


# ============================================================
# Jira Integration Section
# ============================================================

def render_jira_integration_section():
    """
    Render Jira integration section with a fully dynamic UI
    that discovers required fields from Jira API.
    """

    if "draft" not in st.session_state:
        return

    st.header("Jira Integration")

    services = get_services()

    is_epic = st.session_state.get("is_epic", False)
    issue_type_name = "Epic" if is_epic else "Story"

    st.subheader(f"Create Jira {issue_type_name}")

    st.markdown("**Project Selection**")

    project_keys = list(JIRA_PROJECT_CONFIG.keys())
    selected_project_key = st.selectbox(
        "Project Key",
        options=project_keys,
        help="Select the Jira project",
    )

    if not selected_project_key:
        return

    st.info(f"Tickets will be created in project: **{selected_project_key}**")

    # --------------------------------------------------------
    # Board Selection (multi-board projects)
    # --------------------------------------------------------

    selected_board_id = None
    selected_board_info = None

    project_config = JIRA_PROJECT_CONFIG[selected_project_key]

    if "boards" in project_config:
        st.markdown("**Board Selection**")
        st.info(
            f"Project **{selected_project_key}** has multiple boards. "
            "Select the target board:"
        )

        board_options = []
        board_mapping = {}

        for board_id, board_info in project_config["boards"].items():
            display = f"{board_info.get('name', 'Board')} (ID: {board_id})"
            board_options.append(display)
            board_mapping[display] = board_info | {"id": board_id}

        selected_display = st.selectbox("Select Board", board_options)

        if selected_display:
            selected_board_info = board_mapping[selected_display]
            selected_board_id = selected_board_info["id"]

            st.session_state["jira_selected_board_id"] = selected_board_id
            st.session_state["jira_selected_board_info"] = selected_board_info

            with st.expander("Board Details", expanded=False):
                st.markdown(f"**Name:** {selected_board_info.get('name')}")
                st.markdown(f"**Description:** {selected_board_info.get('description')}")
                st.markdown(
                    f"**URL:** [{selected_board_info.get('url')}]"
                    f"({selected_board_info.get('url')})"
                )

    # --------------------------------------------------------
    # Dynamic Jira Fields
    # --------------------------------------------------------

    all_jira_fields = services.jira_client.get_project_fields(
        selected_project_key, issue_type_name
    )

    if not all_jira_fields:
        st.error(
            f"Could not load fields for {issue_type_name} "
            f"in project {selected_project_key}"
        )
        return

    st.divider()
    st.markdown(f"**{issue_type_name} Configuration**")

    field_values = {}
    fields_to_render = all_jira_fields.copy()

    # Remove duplicates handled elsewhere
    for f in ["summary", "description", "project", "issuetype"]:
        fields_to_render.pop(f, None)

    for field_id, details in fields_to_render.items():
        if details.get("required"):
            label = f"{details['name']} (Required)"
            schema = details.get("schema", {})
            schema_type = schema.get("type")

            if schema_type == "number":
                field_values[field_id] = st.number_input(label, step=1.0)
            elif "array" in schema.get("type", "") and details.get("allowedValues"):
                options = [
                    v.get("value", v.get("name"))
                    for v in details["allowedValues"]
                    if v.get("value") or v.get("name")
                ]
                field_values[field_id] = st.multiselect(label, options)
            elif details.get("allowedValues"):
                options = [
                    v.get("value", v.get("name"))
                    for v in details["allowedValues"]
                    if v.get("value") or v.get("name")
                ]
                field_values[field_id] = st.selectbox(label, options)
            else:
                field_values[field_id] = st.text_input(label)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    missing_fields = []
    if not st.session_state.draft.title:
        missing_fields.append("Title")

    for fid, val in field_values.items():
        if not val:
            missing_fields.append(fields_to_render[fid]["name"])

    if missing_fields:
        st.error(
            f"Please fill in all required fields: {', '.join(missing_fields)}"
        )
        st.button(
            f"Create {issue_type_name} in Jira",
            type="primary",
            disabled=True,
        )
    else:
        if st.button(f"Create {issue_type_name} in Jira", type="primary"):
            create_jira_ticket(
                project_key=selected_project_key,
                issue_type=issue_type_name,
                draft=st.session_state.draft,
                field_values=field_values,
                board_id=selected_board_id,
            )
