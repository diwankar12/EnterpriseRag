"""
Story settings component for configuration and input controls.
"""

import streamlit as st


def render_story_settings():
    """Render the story generation settings and configuration interface"""

    # Initialize expander state for persistence
    if "story_settings_expanded" not in st.session_state:
        st.session_state.story_settings_expanded = False

    # Story Generation Settings with checkbox to control expander state
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

    # Show settings content when expanded
    if st.session_state.story_settings_expanded:
        st.markdown("**Document Retrieval & AI Configuration**")

        col1, col2, col3 = st.columns(3)

        with col1:
            k_docs = st.slider(
                "Knowledge Documents",
                min_value=1,
                max_value=600,
                value=st.session_state.get("k_docs", 200),
                step=1,
                help="Number of knowledge documents to retrieve for context",
                key="k_docs",
            )

        with col2:
            k_code = st.slider(
                "Code Documents",
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
                value=st.session_state.get("temperature", 0.15),
                step=0.05,
                help="Lower values = more focused, Higher values = more creative",
                key="temperature",
            )

        col_info, col_reset = st.columns([3, 1])

        with col_info:
            st.markdown(
                f"**Current:** {k_docs} knowledge docs • {k_code} code docs • {temperature:.2f} creativity"
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

    return k_docs, k_code, temperature


def render_story_input():
    """Render story input form and generation controls"""

    with st.container():
        col1, col2 = st.columns([3, 1])

        with col1:
            user_story = st.text_area(
                "Describe your user story or feature request:",
                placeholder=(
                    "e.g. As a user, I want to be able to reset my password "
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

    return user_story, include_code, code_language


def render_chatbot_settings():
    """Render the story generation settings for the chatbot - always visible"""

    st.markdown("**Document Retrieval & AI Configuration**")

    col1, col2, col3 = st.columns(3)

    with col1:
        k_docs = st.slider(
            "Knowledge Documents",
            min_value=1,
            max_value=600,
            value=st.session_state.get("chatbot_k_docs", 10),
            step=1,
            help="Number of knowledge documents to retrieve for context",
            key="chatbot_k_docs",
        )

    with col2:
        k_code = st.slider(
            "Code Documents",
            min_value=1,
            max_value=100,
            value=st.session_state.get("chatbot_k_code", 10),
            step=1,
            help="Number of code documents to retrieve for examples",
            key="chatbot_k_code",
        )

    with col3:
        temperature = st.slider(
            "AI Creativity Level",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("chatbot_temperature", 0.15),
            step=0.05,
            help="Lower values = more focused, Higher values = more creative",
            key="chatbot_temperature",
        )

    col_info, col_reset = st.columns([3, 1])

    with col_info:
        st.markdown(
            f"**Current:** {k_docs} knowledge docs • {k_code} code docs • {temperature:.2f} creativity"
        )

    with col_reset:
        if st.button(
            "Reset to Defaults",
            help="Reset all values to defaults",
            key="reset_chatbot_config",
        ):
            for key in [
                "chatbot_k_docs",
                "chatbot_k_code",
                "chatbot_temperature",
            ]:
                if key in st.session_state:
                    del st.session_state[key]

            st.rerun()

    return k_docs, k_code, temperature
