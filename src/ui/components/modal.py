"""
Modal components for substory display and interaction.
"""

import streamlit as st
from .feedback import (
    process_smart_feedback,
    process_edit_feedback,
    process_regenerate_feedback,
)


class SubstoryComponents:

    @st.dialog("Story Details")
    def render_substory_modal(self):
        """Render modal popup for substory details using st.dialog"""
        modal_index = st.session_state.get("modal_substory_index", 0)

        if modal_index >= len(st.session_state.substories):
            st.error("Invalid story index")
            return

        substory = st.session_state.substories[modal_index]
        is_epic = st.session_state.get("is_epic", False)

        story_type = "Story" if is_epic else "Sub-Story"
        st.markdown(
            f"### {story_type} {modal_index + 1}: {substory.get('title', 'Untitled')}"
        )

        # Description
        st.markdown("**Description:**")
        st.markdown(substory.get("description", "No description provided"))

        # Acceptance criteria
        st.markdown("**Acceptance Criteria:**")
        criteria = substory.get("acceptance_criteria", [])
        if criteria:
            for i, criterion in enumerate(criteria, 1):
                st.markdown(f"{i}. {criterion}")
        else:
            st.markdown("*No acceptance criteria provided*")

        # Code examples
        if substory.get("code_examples"):
            st.markdown("**Code Examples:**")
            for i, code_example in enumerate(substory["code_examples"]):
                with st.expander(
                    f"Example {i + 1} ({code_example.get('language', 'java')})",
                    expanded=False,
                ):
                    st.code(
                        code_example.get("code", ""),
                        language=code_example.get("language", "java"),
                    )

        # Copy text
        st.markdown(f"### Copy {story_type} Details:")
        substory_text = f"""
**{story_type} {modal_index + 1}: {substory.get('title', 'Untitled')}**

**Description:**
{substory.get('description', 'No description provided')}

**Acceptance Criteria:**
{chr(10).join([f"{i+1}. {c}" for i, c in enumerate(substory.get('acceptance_criteria', []))])}
"""

        if substory.get("code_examples"):
            substory_text += "\n\n**Code Examples:**\n"
            for i, code_example in enumerate(substory.get("code_examples", [])):
                substory_text += (
                    f"\nExample {i+1} ({code_example.get('language','java')}):\n"
                    f"```{code_example.get('language','java')}\n"
                    f"{code_example.get('code','')}\n```"
                )

        st.text_area(
            "Copy this text:",
            value=substory_text,
            height=200,
            key=f"modal_copy_text_{modal_index}",
        )

        reference_type = "story" if is_epic else "sub story"
        st.info(
            f"**For AI Assistant:** You can refer to this as '{reference_type} {modal_index + 1}' "
            "when providing feedback or making changes."
        )

        st.markdown("---")
        st.markdown(f"## Improve This {story_type}")

        feedback = st.text_area(
            f"Provide feedback for {story_type} {modal_index + 1}:",
            placeholder=(
                f"Examples:\n"
                f"Add more technical details to {story_type} {modal_index + 1}\n"
                "The acceptance criteria are too vague\n"
                "This looks perfect!"
            ),
            height=80,
            key=f"modal_feedback_{modal_index}",
        )

        if feedback.strip():
            col1, col2, col3 = st.columns(3)
            story_key = f"substory_{modal_index}"

            with col1:
                if st.button(
                    "AI Assistant",
                    key=f"modal_ai_{modal_index}",
                    help="Let AI decide the best action",
                ):
                    process_smart_feedback(feedback, story_key)
                    st.rerun()

            with col2:
                if st.button(
                    "Apply Edit",
                    key=f"modal_edit_{modal_index}",
                    help="Apply specific edits",
                ):
                    process_edit_feedback(feedback, story_key)
                    st.rerun()

            with col3:
                if st.button(
                    "Regenerate",
                    key=f"modal_regen_{modal_index}",
                    help="Regenerate completely",
                ):
                    process_regenerate_feedback(feedback, story_key)
                    st.rerun()

    def render_substories_section(self):
        """Render substories as compact buttons that open modal popups"""
        if "substories" not in st.session_state or not st.session_state.substories:
            return

        is_epic = st.session_state.get("is_epic", False)

        if is_epic:
            st.header("Epic Stories")
        else:
            st.header("Related Sub-Stories")
            st.info(
                f"Found {len(st.session_state.substories)} related sub-stories "
                "that can be developed independently."
            )

        st.markdown("**Click to view full details:**")

        for index, substory in enumerate(st.session_state.substories):
            button_label = (
                f"Story {index + 1}: {substory.get('title', f'Untitled Story {index + 1}')}"
                if is_epic
                else f"Sub-Story {index + 1}: {substory.get('title', f'Untitled Story {index + 1}')}"
            )

            if st.button(
                button_label,
                key=f"substory_button_{index}",
                help="Click to view full details",
                use_container_width=False,
            ):
                st.session_state["modal_substory_index"] = index
                self.render_substory_modal()

        reference_type = "story" if is_epic else "sub story"
        st.info(
            f"**For AI Feedback:** You can reference these as "
            f"'{reference_type} 1', '{reference_type} 2', etc. when giving feedback or requesting changes."
        )
