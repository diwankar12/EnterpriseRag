"""
Feedback component for story improvement and AI-powered editing.
"""

import streamlit as st
import inspect
import os
import re
from typing import Tuple, List, Dict

from ..app_config import get_services
from ..agent import StoryDraft


# ---------------------------------------------------------
# Story Data Helpers
# ---------------------------------------------------------

def get_selected_story_data(selected_story_key: str):
    """Get the data for the selected story"""
    if selected_story_key == "main":
        return st.session_state.draft.model_dump()

    elif selected_story_key.startswith("substory_"):
        index = int(selected_story_key.split("_")[1])
        if "substories" in st.session_state and index < len(st.session_state.substories):
            return st.session_state.substories[index]
        else:
            raise ValueError(f"Invalid substory index: {index}")

    else:
        raise ValueError(f"Invalid story key: {selected_story_key}")


def update_selected_story_data(selected_story_key: str, updated_data: dict):
    """Update the data for the selected story"""
    if selected_story_key == "main":
        st.session_state["draft"] = StoryDraft(**updated_data)

    elif selected_story_key.startswith("substory_"):
        index = int(selected_story_key.split("_")[1])
        if "substories" in st.session_state and index < len(st.session_state.substories):
            updated_data = fix_acceptance_criteria_format(updated_data)
            st.session_state.substories[index] = updated_data
        else:
            raise ValueError(f"Invalid substory index: {index}")

    else:
        raise ValueError(f"Invalid story key: {selected_story_key}")


# ---------------------------------------------------------
# UI Rendering
# ---------------------------------------------------------

def render_feedback_section():
    """Render the feedback and improvement interface"""
    if "draft" not in st.session_state:
        return

    st.header("Story Feedback & Improvement")

    story_options = []
    selected_story_key = None

    story_options.append(("Main Story", "main"))

    if "substories" in st.session_state and st.session_state.substories:
        is_epic = st.session_state.get("is_epic", False)
        story_type = "Story" if is_epic else "Sub-Story"

        for i, substory in enumerate(st.session_state.substories):
            title = substory.get("title", f"Untitled {story_type} {i+1}")
            story_options.append(
                (f"{story_type} {i+1}: {title}", f"substory_{i}")
            )

    if len(story_options) > 1:
        st.markdown("### Select Story to Provide Feedback On")
        selected_option = st.selectbox(
            "Choose which story you want to improve:",
            options=[opt[0] for opt in story_options],
            key="story_selector",
        )

        for option_label, option_key in story_options:
            if option_label == selected_option:
                selected_story_key = option_key
                break

        st.markdown("---")
    else:
        selected_story_key = "main"

    st.info(
        """
**How to improve your story:**

**AI Assistant** – Let AI automatically decide what to do with your feedback  
**Apply Edit** – Make specific targeted changes  
**Force Regenerate** – Start completely fresh
"""
    )

    services = get_services()

    feedback = st.text_area(
        "Tell us what you think about the story:",
        placeholder=(
            "Examples:\n"
            "• Add more technical details\n"
            "• Acceptance criteria are too vague\n"
            "• Include security considerations"
        ),
        height=100,
        key="feedback_input",
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Approve Story", type="primary"):
            st.session_state["story_approved"] = True
            story_name = "Main Story" if selected_story_key == "main" else f"Sub-Story {selected_story_key.split('_')[1]}"
            st.success(f"{story_name} approved! Ready for Jira ticket creation")

    with col2:
        if st.button("AI Assistant", disabled=not feedback.strip()):
            process_smart_feedback(feedback, selected_story_key)

    with col3:
        if st.button("Apply Edit", disabled=not feedback.strip()):
            process_edit_feedback(feedback, selected_story_key)

    with col4:
        if st.button("Force Regenerate", disabled=not feedback.strip()):
            process_regenerate_feedback(feedback, selected_story_key)


# ---------------------------------------------------------
# Smart Feedback
# ---------------------------------------------------------

def process_smart_feedback(feedback: str, selected_story_key: str = "main"):
    """Process feedback using the smart feedback agent"""
    services = get_services()

    story_prefix_match = re.match(
        r"^\s*Story\s*(\d+)\s*:\s*(.+)$", feedback, re.IGNORECASE
    )
    if story_prefix_match:
        story_num = int(story_prefix_match.group(1))
        feedback = story_prefix_match.group(2).strip()
        selected_story_key = f"substory_{story_num - 1}"

    if not services.feedback_agent:
        st.error("Feedback agent not available")
        return

    with st.spinner("Feedback Agent is analyzing your feedback..."):
        try:
            current_draft = get_selected_story_data(selected_story_key)

            clean_description, code_blocks = extract_code_blocks(
                current_draft.get("description", "")
            )

            if code_blocks:
                feedback += (
                    f"\n\nNote: This story includes {len(code_blocks)} code example(s). "
                    "Please preserve code formatting and structure when making changes."
                )

            result = services.feedback_agent.process_feedback(
                draft=current_draft,
                feedback=feedback,
            )

            action = result.get("action")
            processing_time = result.get("processing_time", "N/A")

            if action == "approve":
                st.session_state["story_approved"] = True
                st.success(f"Story approved (took {processing_time})")

            elif action == "edit":
                draft_data = result.get("draft")
                if draft_data:
                    draft_data = fix_acceptance_criteria_format(draft_data)
                    update_selected_story_data(selected_story_key, draft_data)
                    st.success(f"Story updated with edits (took {processing_time})")
                    st.rerun()
                else:
                    st.error("No draft data returned from edit action")

            elif action == "regenerate":
                draft_data = result.get("draft")
                if draft_data:
                    draft_data = fix_acceptance_criteria_format(draft_data)
                    update_selected_story_data(selected_story_key, draft_data)
                    st.success(f"Story regenerated (took {processing_time})")
                    st.rerun()
                else:
                    st.error("No draft data returned from regenerate action")

            elif action == "reject":
                st.error("Story rejected – please revise requirements")

            elif action == "error":
                st.error(
                    f"FeedbackAgent encountered an error: {result.get('changes', 'Unknown error')}"
                )

            else:
                st.warning(f"Unknown action returned: {action}")

        except Exception as e:
            st.error(f"Error processing feedback: {str(e)}")


# ---------------------------------------------------------
# Edit Feedback
# ---------------------------------------------------------

def process_edit_feedback(feedback: str, selected_story_key: str = "main"):
    """Process feedback as edit instructions"""
    services = get_services()

    with st.spinner("Applying edits..."):
        try:
            current_draft = get_selected_story_data(selected_story_key)
            new_draft = services.agentic_rag.apply_feedback(
                current_draft,
                feedback,
            )

            new_draft = fix_acceptance_criteria_format(new_draft)
            update_selected_story_data(selected_story_key, new_draft)

            story_name = "Main Story" if selected_story_key == "main" else f"Sub-Story {selected_story_key.split('_')[1]}"
            st.success(f"{story_name} updated with your feedback")
            st.rerun()

        except Exception as e:
            st.error(f"Error applying feedback: {str(e)}")


# ---------------------------------------------------------
# Regenerate Feedback
# ---------------------------------------------------------

def process_regenerate_feedback(feedback: str, selected_story_key: str = "main"):
    """Process feedback by regenerating the story"""
    services = get_services()

    with st.spinner("Regenerating story..."):
        try:
            if services.feedback_agent:
                new_draft = services.feedback_agent.regenerate(feedback)
            else:
                current_draft = get_selected_story_data(selected_story_key)
                new_draft = services.agentic_rag.apply_feedback(
                    current_draft,
                    feedback,
                )

            new_draft = fix_acceptance_criteria_format(new_draft)
            update_selected_story_data(selected_story_key, new_draft)

            story_name = "Main Story" if selected_story_key == "main" else f"Sub-Story {selected_story_key.split('_')[1]}"
            st.success(f"{story_name} regenerated successfully")
            st.rerun()

        except Exception as e:
            st.error(f"Error regenerating story: {str(e)}")


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def fix_acceptance_criteria_format(draft_data: dict) -> dict:
    """Fix acceptance_criteria format if it contains dictionaries instead of strings"""
    if "acceptance_criteria" in draft_data and draft_data["acceptance_criteria"]:
        criteria = draft_data["acceptance_criteria"]

        if isinstance(criteria, list):
            fixed_criteria = []
            for criterion in criteria:
                if isinstance(criterion, dict) and "criterion" in criterion:
                    fixed_criteria.append(criterion["criterion"])
                elif isinstance(criterion, str):
                    fixed_criteria.append(criterion)
                else:
                    fixed_criteria.append(str(criterion))

            draft_data["acceptance_criteria"] = fixed_criteria

    return draft_data


def extract_code_blocks(text: str) -> Tuple[str, List[dict]]:
    """Extract code blocks from text and return cleaned text with code blocks separately"""
    code_blocks = []

    code_pattern = r"```(\w*)\n(.*?)\n```"
    matches = re.findall(code_pattern, text, re.DOTALL)

    if matches:
        for language, code in matches:
            code_blocks.append(
                {
                    "language": language or "text",
                    "code": code.strip(),
                }
            )

        clean_text = re.sub(
            code_pattern,
            "\n[Code example shown below]\n",
            text,
            flags=re.DOTALL,
        )
        clean_text = re.sub(r"\n+", "\n", clean_text).strip()

        return clean_text, code_blocks

    return text, []
