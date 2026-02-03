"""
Jira integration components for ticket creation and management.
"""

import streamlit as st
import re
from urllib.parse import urlparse
from typing import Dict, Any

from ...app_config import get_services
from ...app_config import JIRA_PROJECT_CONFIG


class JiraIntegration:

    def render_jira_integration_section(self):
        """Render Jira integration and ticket creation"""
        if "draft" not in st.session_state:
            return

        st.header("Jira Integration")
        services = get_services()

        st.subheader("Create Jira Ticket")
        st.markdown("**Project Selection**")

        selected_project_key = st.text_input(
            "Project Key:",
            value="AVNGR",
            placeholder="e.g. AVNGR, INN, PROP, SPECFIN",
            help="Enter the Jira project key where the story will be created",
        ).upper()

        selected_board_id = None
        selected_board_info = None

        if selected_project_key and selected_project_key in JIRA_PROJECT_CONFIG:
            project_config = JIRA_PROJECT_CONFIG[selected_project_key]

            if "boards" in project_config:
                st.markdown("**Board Selection**")
                st.info(
                    f"Project **{selected_project_key}** has multiple boards. "
                    "Please select the target board:"
                )

                board_options = []
                board_mapping = {}

                for board_id, board_info in project_config["boards"].items():
                    display_name = f"{board_info.get('name', 'Board')} (ID: {board_id})"
                    board_options.append((display_name, board_id))
                    board_mapping[str(board_id)] = board_info

                selected_board_index = st.selectbox(
                    "Select Board:",
                    options=[opt[0] for opt in board_options],
                    help="Choose the specific board where the ticket should be created",
                )

                if selected_board_index:
                    selected_tuple = next(
                        (t for t in board_options if t[0] == selected_board_index),
                        None,
                    )
                    if selected_tuple:
                        selected_board_id = selected_tuple[1]
                        selected_board_info = board_mapping.get(str(selected_board_id))

                        st.session_state["jira_selected_board_id"] = selected_board_id
                        st.session_state["jira_selected_board_info"] = selected_board_info

                        with st.expander("Board Details", expanded=False):
                            st.markdown(f"**Name:** {selected_board_info.get('name')}")
                            st.markdown(
                                f"**Description:** {selected_board_info.get('description')}"
                            )
                            st.markdown(f"**Board ID:** {selected_board_id}")
                            st.markdown(
                                f"**URL:** {selected_board_info.get('url')}"
                            )
            else:
                st.info(
                    f"Tickets will be created in project: **{selected_project_key}**"
                )

        elif selected_project_key:
            st.warning(
                f"Project **{selected_project_key}** is not in the predefined configuration. "
                "Using default settings."
            )
            st.info(
                f"Tickets will be created in project: **{selected_project_key}**"
            )

        st.divider()
        st.markdown("**Configuration**")

        col1, col2 = st.columns(2)

        with col1:
            app_in_scope_options = [
                "EMG",
                "STS",
                "OPSVIEW",
                "UDM",
                "STANDALONE",
                "PROPERTY",
                "TWN",
            ]
            app_in_scope = st.multiselect(
                "Application In Scope",
                options=app_in_scope_options,
                default=["PROPERTY"],
                help="Select applications that are in scope for this story",
            )

            planned_change = st.selectbox(
                "MAPSF-Planned Change",
                ["No", "Yes"],
                help="Indicates if this is a planned change",
            )

        with col2:
            story_points = st.number_input(
                "Story Point Estimate (for Epic)",
                min_value=0.0,
                value=5.0,
                step=1.0,
                help="Estimate the effort required for the main epic/story",
            )

            work_type = st.selectbox(
                "Work Type",
                [
                    "COGS-SECURITY",
                    "COGS-AUDITS",
                    "TRANSFORMATION",
                    "NPI",
                    "COGS-PROD-DEFECTS",
                ],
                help="Select the type of work this story represents",
            )

            sprint_id = st.number_input(
                "Sprint ID (optional)",
                min_value=0,
                value=0,
                step=1,
                help=(
                    "Enter the numeric Sprint ID from Jira "
                    "(leave as 0 to skip sprint assignment)"
                ),
            )

        jira_token_input = st.text_input(
            "Jira API Token (optional)",
            value="",
            type="password",
            help=(
                "Provide an API token to use for this session; "
                "if left blank the app will use the configured or environment token"
            ),
            key="jira_api_token_input",
        )

        st.markdown("**Ticket Details**")

        story_title = st.text_input(
            "Title:",
            value=st.session_state.draft.title
            if hasattr(st.session_state.draft, "title")
            else "",
            help="Enter the title for the Jira story or epic",
        )

        story_description = st.text_area(
            "Description:",
            value=st.session_state.draft.description
            if hasattr(st.session_state.draft, "description")
            else "",
            height=200,
            help="Edit the description as needed",
        )

        has_code_examples = (
            hasattr(st.session_state.draft, "code_examples")
            and st.session_state.draft.code_examples
            and len(st.session_state.draft.code_examples) > 0
        )

        if has_code_examples:
            st.info(
                "This story includes code examples that will be attached as files to the Jira ticket."
            )

        create_button_disabled = not selected_project_key.strip()

        if (
            selected_project_key
            and selected_project_key in JIRA_PROJECT_CONFIG
            and "boards" in JIRA_PROJECT_CONFIG[selected_project_key]
            and not selected_board_id
        ):
            create_button_disabled = True
            st.warning("Please select a board for this multi-board project.")

        if st.button(
            "Create in Jira",
            type="primary",
            disabled=create_button_disabled,
        ):
            board_id_to_use = st.session_state.get(
                "jira_selected_board_id", selected_board_id
            )
            board_info_to_use = st.session_state.get(
                "jira_selected_board_info", selected_board_info
            )

            if jira_token_input and jira_token_input.strip():
                try:
                    services.jira_client.set_api_token(jira_token_input.strip())
                except Exception:
                    pass

            self.create_jira_ticket_enhanced(
                selected_project_key,
                story_title,
                story_description,
                app_in_scope,
                story_points,
                planned_change,
                work_type,
                sprint_id,
                services,
                has_code_examples,
                board_id_to_use,
                board_info_to_use,
            )

    def create_jira_ticket_enhanced(
        self,
        project_key: str,
        story_title: str,
        story_description: str,
        app_in_scope: list,
        story_points: float,
        planned_change: str,
        work_type: str,
        sprint_id: int,
        services,
        has_code_examples: bool = False,
        board_id: str = None,
        board_info: Dict[str, Any] = None,
    ):
        """Create a Jira ticket, handling both single stories and epics with sub-stories."""

        if not services.jira_client:
            st.error("Jira client not configured")
            return

        if not project_key.strip():
            st.error("Project key is required")
            return

        is_epic_creation = st.session_state.get("is_epic", False)
        main_issue_type = "Epic" if is_epic_creation else "Story"

        try:
            with st.spinner(f"Creating Jira {main_issue_type}..."):
                config = services.config

                if board_id and services.jira_client:
                    services.jira_client.set_board_context(board_id, board_info)

                formatted_description = story_description
                code_pattern = r"```(\w*)\n(.*?)```"

                def replace_code_block(match):
                    language = match.group(1) or "text"
                    code = match.group(2)
                    return f"{{code:{language}}}\n{code}\n{{code}}"

                formatted_description = re.sub(
                    code_pattern,
                    replace_code_block,
                    formatted_description,
                    flags=re.DOTALL,
                )

                main_ticket_fields = {
                    "project": {"key": project_key},
                    "summary": story_title,
                    "description": formatted_description,
                    "issuetype": {"name": main_issue_type},
                    "labels": ["ideasflow"],
                    "customfield_21102": story_points,
                    "customfield_26989": [{"value": app} for app in app_in_scope],
                    "customfield_26865": {"value": planned_change},
                    "customfield_25163": {"value": work_type},
                }

                if board_id:
                    board_label = f"board-{board_id}"
                    existing_labels = main_ticket_fields.get("labels", []) or []
                    if board_label not in existing_labels:
                        existing_labels.append(board_label)
                    main_ticket_fields["labels"] = existing_labels

                if is_epic_creation:
                    main_ticket_fields["customfield_10011"] = story_title

                if sprint_id > 0:
                    main_ticket_fields["customfield_10300"] = sprint_id

                if (
                    hasattr(st.session_state.draft, "acceptance_criteria")
                    and st.session_state.draft.acceptance_criteria
                ):
                    acceptance_text = "\n\nAcceptance Criteria:\n"
                    for i, criterion in enumerate(
                        st.session_state.draft.acceptance_criteria, 1
                    ):
                        acceptance_text += f"{i}. {criterion}\n"
                    main_ticket_fields["description"] += acceptance_text

                if is_epic_creation:
                    ticket_key = services.jira_client.create_epic(
                        main_ticket_fields
                    )
                else:
                    ticket_key = services.jira_client.create_story(
                        main_ticket_fields
                    )

                if not ticket_key:
                    st.error(
                        f"Failed to create Jira {main_issue_type}. "
                        "Check the application logs for details."
                    )
                    return

                parsed_url = urlparse(config.JIRA_API_URL)
                base_jira_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                jira_browse_url = f"{base_jira_url}/browse/{ticket_key}"

                success_message = (
                    f"Jira {main_issue_type} created in project **{project_key}**"
                )
                if board_info:
                    success_message += f" on board **{board_info.get('name')}**"

                st.success(success_message)
                st.markdown(
                    f"**[View Ticket: {ticket_key}]({jira_browse_url})**"
                )

                if board_info and "url" in board_info:
                    st.markdown(
                        f"**[View on Board: {board_info.get('name')}]({board_info.get('url')})**"
                    )

                with st.expander("Ticket Details", expanded=False):
                    st.markdown(f"**Project:** {project_key}")
                    if board_info:
                        st.markdown(
                            f"**Board:** {board_info.get('name')} "
                            f"(ID: {board_id})"
                        )
                        st.markdown(
                            f"**Board Description:** {board_info.get('description')}"
                        )
                    st.markdown(f"**Ticket Key:** {ticket_key}")
                    st.markdown(f"**Direct Link:** {jira_browse_url}")

                if has_code_examples and hasattr(
                    st.session_state.draft, "code_examples"
                ):
                    self.attach_code_examples_to_jira(
                        ticket_key,
                        st.session_state.draft.code_examples,
                        services,
                    )

                if (
                    is_epic_creation
                    and "substories" in st.session_state
                    and st.session_state["substories"]
                ):
                    st.info(
                        f"Creating {len(st.session_state['substories'])} "
                        f"sub-stories under Epic {ticket_key}..."
                    )

                    substories = st.session_state["substories"]
                    progress_bar = st.progress(
                        0, text="Creating sub-stories..."
                    )

                    for i, substory in enumerate(substories):
                        progress_text = (
                            f"Creating sub-story {i+1}/{len(substories)}: "
                            f"{substory['title']}"
                        )
                        progress_bar.progress(
                            (i + 1) / len(substories), text=progress_text
                        )

                        substory_description = substory.get("description", "")
                        if substory.get("acceptance_criteria"):
                            substory_description += (
                                "\n\nAcceptance Criteria:\n"
                            )
                            for ac in substory["acceptance_criteria"]:
                                substory_description += f"- {ac}\n"

                        substory_fields = {
                            "project": {"key": project_key},
                            "summary": substory["title"],
                            "description": substory_description,
                            "issuetype": {"name": "Story"},
                            "labels": ["ideasflow"],
                        }

                        if board_id:
                            board_label = f"board-{board_id}"
                            if board_label not in substory_fields.get(
                                "labels", []
                            ):
                                substory_fields.setdefault(
                                    "labels", []
                                ).append(board_label)

                        story_key = (
                            services.jira_client.create_story_with_epic_link(
                                substory_fields,
                                ticket_key,
                            )
                        )

                        if story_key:
                            story_url = (
                                f"{base_jira_url}/browse/{story_key}"
                            )
                            st.write(
                                f"✅ Created story "
                                f"[{story_key}]({story_url}) "
                                f"and linked to Epic {ticket_key}"
                            )
                        else:
                            st.warning(
                                f"⚠️ Failed to create sub-story: "
                                f"{substory['title']}"
                            )

                    progress_bar.empty()
                    st.success("All sub-stories have been processed.")

                if services.jira_client:
                    services.jira_client.clear_board_context()

                if st.button("Create Another Story"):
                    for key in [
                        "draft",
                        "context",
                        "generation_timestamp",
                        "story_approved",
                        "substories",
                        "is_epic",
                    ]:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

        except Exception as e:
            if services.jira_client:
                services.jira_client.clear_board_context()
            st.error(
                f"An unexpected error occurred while creating the Jira ticket: {str(e)}"
            )

    def attach_code_examples_to_jira(
        self, ticket_key: str, code_examples: list, services
    ):
        """Attach code examples to a Jira ticket as text files."""
        try:
            with st.spinner(
                f"Attaching code examples to {ticket_key}..."
            ):
                for i, code_example in enumerate(code_examples):
                    language = code_example.get("language", "txt")
                    filename = f"code_example_{i+1}.{language}"
                    content = code_example.get("code", "")

                    header = (
                        f"# Code Example {i+1}\n"
                        f"# Language: {language}\n"
                        f"# Generated by AI Story Generator\n\n"
                    )
                    full_content = header + content

                    success = (
                        services.jira_client.add_attachment_from_content(
                            ticket_key,
                            filename,
                            full_content.encode("utf-8"),
                            "text/plain",
                        )
                    )

                    if success:
                        st.write(
                            f"📎 Attached `{filename}` to {ticket_key}."
                        )
                    else:
                        st.warning(
                            f"⚠️ Failed to attach `{filename}`."
                        )

        except Exception as e:
            st.warning(
                f"An error occurred while attaching code examples: {str(e)}"
            )
