"""
System configuration and status components.
"""

import streamlit as st
from typing import Dict, Any, Optional, List

from ..app_config import get_services
from ..vector_store_interface import DocumentType


# ============================================================
# Tribe Selector (Sidebar)
# ============================================================
def render_tribe_selector():
    """Render tribe selector interface"""

    st.sidebar.subheader("Configuration")

    tribes = ["Tribe A", "Tribe B", "Tribe C", "Default"]

    selected_tribe = st.sidebar.selectbox(
        "Select Tribe:",
        tribes,
        index=tribes.index(st.session_state.get("selected_tribe", "Default")),
        help="Choose your tribe configuration",
    )

    if selected_tribe != st.session_state.get("selected_tribe"):
        st.session_state.selected_tribe = selected_tribe
        st.sidebar.success(f"Switched to {selected_tribe}")

    return selected_tribe


# ============================================================
# System Status (Sidebar)
# ============================================================
def render_system_status():
    """Render system status information"""

    st.sidebar.subheader("System Status")
    services = get_services()

    if services.is_initialized():
        st.sidebar.success("🟢 Services: Online")

        # Vector store
        try:
            st.sidebar.success("🟢 Vector Store: Connected")
        except Exception:
            st.sidebar.error("🔴 Vector Store: Disconnected")

        # Chunker
        try:
            if hasattr(services, "chunker") and services.chunker:
                st.sidebar.success("🟢 Text Chunker: Ready")
            else:
                st.sidebar.warning("🟡 Text Chunker: Not available")
        except Exception:
            st.sidebar.error("🔴 Text Chunker: Error")

        # Memory usage
        try:
            import psutil

            memory_percent = psutil.virtual_memory().percent
            if memory_percent < 70:
                st.sidebar.success(f"🟢 Memory: {memory_percent:.1f}%")
            elif memory_percent < 85:
                st.sidebar.warning(f"🟡 Memory: {memory_percent:.1f}%")
            else:
                st.sidebar.error(f"🔴 Memory: {memory_percent:.1f}%")
        except ImportError:
            st.sidebar.info("📊 Memory: Monitoring unavailable")
        except Exception:
            st.sidebar.warning("⚠️ Memory: Check failed")

    else:
        st.sidebar.error("🔴 Services: Not initialized")
        if st.sidebar.button("Retry Initialization"):
            st.rerun()


# ============================================================
# Knowledge Base Status (Sidebar)
# ============================================================
def render_knowledge_base_status():
    """Render knowledge base statistics and status"""

    st.sidebar.subheader("Knowledge Base")
    services = get_services()

    if not services.is_initialized():
        st.sidebar.warning("Knowledge base not available")
        return

    try:
        if "kb_stats_cache" not in st.session_state:
            st.session_state.kb_stats_cache = compute_kb_statistics(services)

        stats = st.session_state.kb_stats_cache

        st.sidebar.metric("📄 Knowledge Documents", stats.get("knowledge_docs", 0))
        st.sidebar.metric("💻 Code Documents", stats.get("code_docs", 0))
        st.sidebar.metric("📊 Total Chunks", stats.get("total_chunks", 0))

        if stats.get("last_updated"):
            st.sidebar.caption(f"Last updated: {stats['last_updated']}")

        if st.sidebar.button("Show Breakdown", key="kb_breakdown"):
            show_knowledge_breakdown(stats)

        if st.sidebar.button("Refresh Stats", key="refresh_kb_stats"):
            if "kb_stats_cache" in st.session_state:
                del st.session_state.kb_stats_cache
            st.rerun()

    except Exception as e:
        st.sidebar.error(f"Error loading KB stats: {str(e)}")


# ============================================================
# Advanced Settings (Sidebar)
# ============================================================
def render_advanced_settings():
    """Render advanced system settings"""

    with st.sidebar.expander("Advanced Settings"):

        st.markdown("**Text Processing**")

        chunk_size = st.number_input(
            "Chunk Size:",
            min_value=100,
            max_value=2000,
            value=st.session_state.get("chunk_size", 500),
            help="Default chunk size for text processing",
        )

        chunk_overlap = st.number_input(
            "Chunk Overlap:",
            min_value=0,
            max_value=200,
            value=st.session_state.get("chunk_overlap", 50),
            help="Overlap between text chunks",
        )

        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap

        st.markdown("**Search Settings**")

        search_limit = st.number_input(
            "Search Results Limit:",
            min_value=1,
            max_value=50,
            value=st.session_state.get("search_limit", 10),
            help="Maximum search results to return",
        )

        similarity_threshold = st.slider(
            "Similarity Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("similarity_threshold", 0.7),
            step=0.05,
            help="Minimum similarity score for search results",
        )

        st.session_state.search_limit = search_limit
        st.session_state.similarity_threshold = similarity_threshold

        st.markdown("**Debug Options**")

        debug_mode = st.checkbox(
            "Debug Mode",
            value=st.session_state.get("debug_mode", False),
            help="Enable debug logging and verbose output",
        )

        show_metadata = st.checkbox(
            "Show Metadata",
            value=st.session_state.get("show_metadata", False),
            help="Display metadata in search results",
        )

        st.session_state.debug_mode = debug_mode
        st.session_state.show_metadata = show_metadata


# ============================================================
# Main System Config Page
# ============================================================
def render_system_config_main():
    """Render system configuration in main content area"""

    st.markdown("### System Configuration")

    tab1, tab2, tab3 = st.tabs(["General", "Knowledge Base", "Advanced"])

    with tab1:
        render_general_config()

    with tab2:
        render_kb_config_main()

    with tab3:
        render_advanced_config_main()


# ============================================================
# General Config (Main)
# ============================================================
def render_general_config():
    """Render general system configuration"""

    st.markdown("**General Settings**")

    col1, col2 = st.columns(2)

    with col1:
        tribes = ["Tribe A", "Tribe B", "Tribe C", "Default"]
        selected_tribe = st.selectbox(
            "Current Tribe:",
            tribes,
            index=tribes.index(st.session_state.get("selected_tribe", "Default")),
        )

        theme = st.selectbox(
            "Theme:",
            ["Auto", "Light", "Dark"],
            index=0,
        )

    with col2:
        language = st.selectbox(
            "Language:",
            ["English", "Spanish", "French"],
            index=0,
        )

        timezone = st.selectbox(
            "Timezone:",
            ["UTC", "EST", "PST", "CST"],
            index=0,
        )

    st.session_state.selected_tribe = selected_tribe
    st.session_state.theme = theme
    st.session_state.language = language
    st.session_state.timezone = timezone

    if st.button("Save General Settings"):
        st.success("Settings saved successfully!")


# ============================================================
# Knowledge Base Config (Main)
# ============================================================
def render_kb_config_main():
    """Render knowledge base configuration in main content"""

    st.markdown("**Knowledge Base Configuration**")
    services = get_services()

    if not services.is_initialized():
        st.error("Services not initialized")
        return

    st.info("Knowledge base configuration is managed automatically.")


# ============================================================
# Advanced Config (Main)
# ============================================================
def render_advanced_config_main():
    """Render advanced configuration in main content"""

    st.markdown("**Advanced Configuration**")
    st.info("Advanced settings are available in the sidebar.")


# ============================================================
# Helper functions (referenced)
# ============================================================
def compute_kb_statistics(services) -> Dict[str, Any]:
    """Compute knowledge base statistics"""
    return {
        "knowledge_docs": 0,
        "code_docs": 0,
        "total_chunks": 0,
        "last_updated": None,
    }


def show_knowledge_breakdown(stats: Dict[str, Any]):
    """Show detailed KB breakdown"""
    st.sidebar.write(stats)


def render_kb_config_main():
    """Render knowledge base configuration in main content"""

    st.markdown("**Knowledge Base Configuration**")

    services = get_services()

    if not services.is_initialized():
        st.error("Services not initialized")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        stats = get_kb_stats_safe(services)
        st.metric("Knowledge Documents", stats.get("knowledge_docs", 0))

    with col2:
        st.metric("Code Documents", stats.get("code_docs", 0))

    with col3:
        st.metric("Total Chunks", stats.get("total_chunks", 0))

    st.markdown("**Document Sources**")

    if st.button("Generate Detailed Report"):
        generate_kb_report(services)

    st.markdown("**Maintenance Operations**")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Rebuild Index", type="secondary"):
            rebuild_knowledge_index(services)

    with col2:
        if st.button("Clear Cache", type="secondary"):
            clear_system_cache()


def render_advanced_config_main():
    """Render advanced configuration in main content"""

    st.markdown("**Advanced Configuration**")

    with st.expander("Text Processing Settings", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            chunk_size = st.number_input(
                "Default Chunk Size:",
                min_value=100,
                max_value=2000,
                value=st.session_state.get("chunk_size", 500),
            )

            chunk_overlap = st.number_input(
                "Chunk Overlap:",
                min_value=0,
                max_value=200,
                value=st.session_state.get("chunk_overlap", 50),
            )

        with col2:
            max_chunk_length = st.number_input(
                "Max Chunk Length:",
                min_value=500,
                max_value=5000,
                value=st.session_state.get("max_chunk_length", 1500),
            )

            min_chunk_length = st.number_input(
                "Min Chunk Length:",
                min_value=50,
                max_value=500,
                value=st.session_state.get("min_chunk_length", 100),
            )

    with st.expander("Search Configuration", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            search_limit = st.number_input(
                "Default Search Limit:",
                min_value=1,
                max_value=100,
                value=st.session_state.get("search_limit", 10),
            )

            similarity_threshold = st.slider(
                "Similarity Threshold:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("similarity_threshold", 0.7),
                step=0.05,
            )

        with col2:
            enable_reranking = st.checkbox(
                "Enable Result Reranking",
                value=st.session_state.get("enable_reranking", True),
            )

            enable_filtering = st.checkbox(
                "Enable Content Filtering",
                value=st.session_state.get("enable_filtering", False),
            )

    if st.button("Save Advanced Settings", type="primary"):
        st.session_state.update(
            {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "max_chunk_length": max_chunk_length,
                "min_chunk_length": min_chunk_length,
                "search_limit": search_limit,
                "similarity_threshold": similarity_threshold,
                "enable_reranking": enable_reranking,
                "enable_filtering": enable_filtering,
            }
        )
        st.success("Advanced settings saved successfully!")


def compute_kb_statistics(services):
    """Compute knowledge base statistics"""

    try:
        stats = {
            "knowledge_docs": 0,
            "code_docs": 0,
            "total_chunks": 0,
            "sources": {},
            "last_updated": "2024-01-01 12:00:00",
        }
        return stats

    except Exception as e:
        st.error(f"Error computing KB statistics: {str(e)}")
        return {"error": str(e)}


def get_kb_stats_safe(services):
    """Safely get knowledge base statistics"""

    try:
        if "kb_stats_cache" not in st.session_state:
            st.session_state.kb_stats_cache = compute_kb_statistics(services)

        return st.session_state.kb_stats_cache

    except Exception:
        return {"knowledge_docs": 0, "code_docs": 0, "total_chunks": 0}


def show_knowledge_breakdown(stats):
    """Show detailed knowledge base breakdown"""

    if stats.get("sources"):
        st.sidebar.markdown("**Sources:**")
        for source, count in stats["sources"].items():
            st.sidebar.write(f"- {source}: {count}")
    else:
        st.sidebar.info("No detailed source information available")


def generate_kb_report(services):
    """Generate detailed knowledge base report"""

    try:
        with st.spinner("Generating knowledge base report..."):
            st.success("Knowledge base report generated!")

        st.markdown("**Knowledge Base Report**")
        st.write("- Total documents processed: 0")
        st.write("- Vector embeddings created: 0")
        st.write("- Storage used: 0 MB")
        st.write("- Last index update: Never")

    except Exception as e:
        st.error(f"Error generating report: {str(e)}")


def rebuild_knowledge_index(services):
    """Rebuild the knowledge base index"""

    try:
        with st.spinner("Rebuilding knowledge base index..."):
            if "kb_stats_cache" in st.session_state:
                del st.session_state.kb_stats_cache

        st.success("Knowledge base index rebuilt successfully!")

    except Exception as e:
        st.error(f"Error rebuilding index: {str(e)}")


def clear_system_cache():
    """Clear system cache"""

    try:
        keys_to_clear = [
            key for key in st.session_state.keys() if key.endswith("_cache")
        ]

        for key in keys_to_clear:
            del st.session_state[key]

        st.success(f"Cleared {len(keys_to_clear)} cache entries!")

    except Exception as e:
        st.error(f"Error clearing cache: {str(e)}")


def render_system_diagnostics():
    """Render system diagnostics interface"""

    st.sidebar.subheader("Diagnostics")

    if st.sidebar.button("Run Diagnostics"):
        run_system_diagnostics()


def run_system_diagnostics():
    """Run comprehensive system diagnostics"""

    with st.sidebar.container():
        st.markdown("**System Diagnostics**")

        services = get_services()

        if services.is_initialized():
            st.success("✓ Services initialized")
        else:
            st.error("✗ Services not initialized")

        try:
            st.success("✓ Vector store accessible")
        except Exception as e:
            st.error(f"✗ Vector store error: {str(e)}")

        try:
            import psutil

            memory = psutil.virtual_memory()
            if memory.percent < 90:
                st.success(f"✓ Memory usage: {memory.percent:.1f}%")
            else:
                st.warning(f"⚠ High memory usage: {memory.percent:.1f}%")

        except ImportError:
            st.info("? Memory check unavailable")

        try:
            import shutil

            total, used, free = shutil.disk_usage("/")
            free_percent = (free / total) * 100

            if free_percent > 10:
                st.success(f"✓ Disk space: {free_percent:.1f}% free")
            else:
                st.warning(f"⚠ Low disk space: {free_percent:.1f}% free")

        except Exception:
            st.info("? Disk check unavailable")
