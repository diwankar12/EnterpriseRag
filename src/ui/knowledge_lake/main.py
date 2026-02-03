import streamlit as st

from .confluence import render_confluence_upload_main
from .github import render_github_clone_main
from .jira_import import render_jira_import_main
from .pdf import render_pdf_upload_main
from .spreadsheets import render_csv_upload


def render_knowledge_management_main():
    """Render the main knowledge management interface with tabs"""

    st.info(
        "Welcome to the Knowledge Lake. Here you can manage all the data sources that power the AI assistant."
    )

    # Tabbed interface for better organization
    tabs = st.tabs(
        [
            "📄 Confluence",
            "📁 GitHub",
            "📋 Jira",
            "📊 Spreadsheets",
            "📎 Documents & Images",
        ]
    )

    with tabs[0]:
        render_confluence_upload_main()

    with tabs[1]:
        render_github_clone_main()

    with tabs[2]:
        render_jira_import_main()

    with tabs[3]:
        render_csv_upload()

    with tabs[4]:
        render_pdf_upload_main()
