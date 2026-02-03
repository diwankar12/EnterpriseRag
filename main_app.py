"""
Agile Flow: AI-Powered Story Generation

Production-ready Streamlit application for generating high-quality
Jira stories using AI and Retrieval-Augmented Generation (RAG).
"""

from venv import logger
import os
import datetime
import streamlit as st

# ===================== UI Imports =====================
from src.ui.main_content import (
    render_story_generation_section,
    render_jira_integration_section,
)

from src.ui.main_content_refactored import render_idea_flow_chatbot

from src.ui.sidebar import (
    render_tribe_selector,
    render_memory_config_section,
    render_system_status,
    render_knowledge_management_main,
)

# ===================== Config & Services =====================
from src.app_config import (
    setup_streamlit_config,
    configure_logging,
    get_services,
)

from src.services.authorization_service import AuthorizationService
from src.tribe_config import TRIBE_CONFIG

# ===================== Global Config =====================
CHATBOT_INTERFACE = os.environ.get("CHATBOT_INTERFACE", "streamlit").lower()

# ============================================================
# Main Application
# ============================================================

def main_app():

    # ---------- Session Handling ----------
    now = datetime.datetime.now()
    st.session_state["last_activity"] = now

    # ---------- Streamlit Setup ----------
    setup_streamlit_config()
    configure_logging()

    # ---------- Header ----------
    col_logo, col_title, col_user = st.columns([1, 4, 2])

    with col_logo:
        st.image("static/logo.png", width=110)

    with col_title:
        st.markdown(
            """
            <div style="display:flex;align-items:center;justify-content:center;height:110px;">
              <span style="
                font-size:3.2rem;
                font-weight:900;
                letter-spacing:0.25em;
                text-transform:uppercase;
                background:linear-gradient(135deg,#b61c2b,#d63447,#b61c2b);
                -webkit-background-clip:text;
                -webkit-text-fill-color:transparent;
              ">
                IDEAS&nbsp;FLOW
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_user:
        st.write("")

    # =====================================================
    # Authorization
    # =====================================================
    user_permissions = None
    if "user-logged-in" in st.session_state:
        auth_result = st.session_state["user-logged-in"]
        user_permissions = getattr(auth_result, "permissions", None)

    if user_permissions:
        all_tribes = list(TRIBE_CONFIG.keys())
        accessible_tribes = AuthorizationService.get_filtered_tribes(
            user_permissions, all_tribes
        )

        if not accessible_tribes:
            st.stop()

    # =====================================================
    # Sidebar Navigation
    # =====================================================
    with st.sidebar:

        selected_tribe = render_tribe_selector()
        tribe_norm = selected_tribe.lower()
        st.session_state["tribe_selector_normalized"] = tribe_norm

        render_system_status()
        render_memory_config_section()
        services = get_services(selected_tribe)

        nav_sections = {
            "data_ingestion": [],
            "retrieval": [],
            "management": [],
            "help": [],
        }

        # ---------- Data Ingestion ----------
        nav_sections["data_ingestion"].extend([
            "Knowledge Lake",
            "Knowledge Lake Metrics",
            "Knowledge Lake Sync",
        ])

        # ---------- Retrieval ----------
        nav_sections["retrieval"].extend([
            "Story Generation",
            "Idea Flow Chatbot",
            "Dashboard",
        ])

        # ---------- Management ----------
        nav_sections["management"].append("Settings")

        # ---------- Help ----------
        nav_sections["help"].append("FAQ")

        all_tabs = (
            nav_sections["data_ingestion"]
            + nav_sections["retrieval"]
            + nav_sections["management"]
            + nav_sections["help"]
        )

        if "active_tab" not in st.session_state:
            st.session_state.active_tab = all_tabs[0]

        st.divider()
        st.markdown("## Navigation")

        for section, tabs in nav_sections.items():
            if not tabs:
                continue

            with st.expander(section.replace("_", " ").upper(), expanded=True):
                for tab in tabs:
                    btn_type = "primary" if st.session_state.active_tab == tab else "secondary"
                    if st.button(tab, type=btn_type, use_container_width=True):
                        st.session_state.active_tab = tab
                        st.rerun()

    # =====================================================
    # Main Content Area
    # =====================================================
    tab = st.session_state.active_tab

    if tab == "Knowledge Lake":
        st.title("Knowledge Lake")
        render_knowledge_management_main()

    elif tab == "Story Generation":
        st.title("Story Generation")
        render_story_generation_section()

    elif tab == "Idea Flow Chatbot":
        st.title("Idea Flow Chatbot")
        render_idea_flow_chatbot(services)

    elif tab == "Dashboard":
        from src.ui.dashboard import render_usage_dashboard
        render_usage_dashboard()

    elif tab == "Knowledge Lake Metrics":
        from src.ui.knowledge_lake_metrics import render_knowledge_lake_metrics
        render_knowledge_lake_metrics()

    elif tab == "Knowledge Lake Sync":
        st.title("Knowledge Lake Sync")
        st.markdown("### Synchronize Knowledge Sources")
        st.info("Manage synchronization of knowledge sources from external platforms.")

    elif tab == "Settings":
        st.title("Settings")
        st.markdown("### Application Configuration")

        with st.expander("🎨 Theme Settings", expanded=True):
            st.color_picker("Primary Color", "#b61c2b")
            st.color_picker("Secondary Color", "#2c3e50")

        with st.expander("🔔 Notification Settings"):
            st.checkbox("Email Notifications", value=True)
            st.checkbox("Slack Notifications", value=False)
            st.text_input("Email Address", placeholder="user@example.com")

        with st.expander("🤖 AI Model Settings"):
            st.selectbox("Default Model", ["GPT-4", "GPT-3.5", "Claude-3", "Gemini"])
            st.slider("Temperature", 0.0, 1.0, 0.15)
            st.slider("Max Tokens", 100, 4000, 2000)

        with st.expander("🔐 Security Settings"):
            st.checkbox("Two-Factor Authentication", value=False)
            st.checkbox("Session Timeout", value=True)
            st.number_input("Timeout (minutes)", 15, 120, 30)

        st.divider()
        st.button("Save Settings", type="primary", use_container_width=True)

    elif tab == "FAQ":
        st.title("FAQ")
        with st.expander("How do I add a new knowledge source?"):
            st.markdown("""
            1. Go to **Knowledge Lake**
            2. Click **Add Source**
            3. Select source type (GitHub, Confluence, Jira)
            4. Provide credentials
            5. Click **Save**
            """)

    else:
        st.warning("Unknown tab selected.")

# ============================================================
# SSO Handling
# ============================================================

def enable_sso() -> bool:
    sso_flag = os.getenv("ENABLE_SSO", "True")
    logger.info("Enable SSO flag: %s", sso_flag)
    return sso_flag.lower() == "true"

def main():
    if "user" not in st.session_state:
        st.session_state["user"] = {}

    main_app()

    from oauth import sso
    if enable_sso():
        logger.info("SSO enabled")
        sso.doSSO()
    else:
        logger.info("SSO disabled")

if __name__ == "__main__":
    logger.info("Starting main app")
    main()
