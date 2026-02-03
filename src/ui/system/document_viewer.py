class DocumentViewer:
    """Class to handle document viewing and refresh functionality"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def render_document_refresh_interface(self):
        """Render the document refresh interface"""

        st.header("📚 Vector Store Document Viewer")

        services = get_services()
        if not services.is_initialized():
            st.error("Services not initialized. Please select a tribe first.")
            return

        doc_type_options = {
            "Knowledge Documents": DocumentType.KNOWLEDGE,
            "Code Documents": DocumentType.CODE,
        }

        selected_type = st.selectbox(
            "Select Document Type:",
            options=list(doc_type_options.keys()),
            key="doc_type_selector",
        )

        doc_type = doc_type_options[selected_type]

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            limit = st.number_input(
                "Documents to Display:",
                min_value=1,
                max_value=100,
                value=10,
                key="doc_limit",
            )

        with col2:
            offset = st.number_input(
                "Skip Documents:",
                min_value=0,
                value=0,
                key="doc_offset",
            )

        with col3:
            if st.button("🔄 Refresh Documents", key="refresh_docs"):
                st.cache_data.clear()
                st.rerun()

        self._display_documents(services, doc_type, limit, offset)

    @st.cache_data(ttl=60)
    def _get_documents_cached(
        self,
        tribe_name: str,
        doc_type_value: str,
        limit: int,
        offset: int,
    ) -> List[Dict[str, Any]]:
        """Get documents with caching"""

        services = get_services()
        if not services.vector_store:
            return []

        doc_type = (
            DocumentType.KNOWLEDGE
            if doc_type_value == "knowledge_docs"
            else DocumentType.CODE
        )

        try:
            documents = services.vector_store.get_documents(
                doc_type=doc_type,
                limit=limit,
                offset=offset,
            )

            return [
                {
                    "id": doc.id,
                    "text": doc.text[:500] + "..." if len(doc.text) > 500 else doc.text,
                    "metadata": doc.metadata,
                    "full_text_length": len(doc.text),
                }
                for doc in documents
            ]
        except Exception as e:
            st.error(f"Error retrieving documents: {e}")
            return []

    def _display_documents(
        self,
        services,
        doc_type: DocumentType,
        limit: int,
        offset: int,
    ):
        """Display the documents"""

        if not services.vector_store:
            st.error("Vector store not available")
            return

        with st.expander("ℹ️ Current Configuration", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.info(f"**Tribe:** {services.current_tribe}")
                st.info(f"**Document Type:** {doc_type.value}")

            with col2:
                health = services.vector_store.health_check()
                st.info(f"**Vector Store:** {health.get('vector_store_type', 'Unknown')}")
                st.info(f"**Status:** {health.get('status', 'Unknown')}")

        with st.spinner("Loading statistics..."):
            stats = services.vector_store.get_collection_stats(doc_type)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📄 Total Documents", stats.get("document_count", 0))

        with col2:
            st.metric(
                "📏 Avg Length",
                f"{stats.get('average_doc_length', 0):.0f} chars",
            )

        with col3:
            st.metric("🔗 Unique Sources", stats.get("unique_sources", 0))

        with col4:
            if "table_size_mb" in stats:
                st.metric("💾 Storage", f"{stats.get('table_size_mb', 0)} MB")
            elif "collection_name" in stats:
                st.metric("📦 Collection", stats.get("document_count", 0))

        if stats.get("earliest_document") and stats.get("latest_document"):
            st.info(
                f"**Document Timeline:** "
                f"{stats['earliest_document']} → {stats['latest_document']}"
            )

        with st.spinner(f"Loading {limit} documents..."):
            documents = self._get_documents_cached(
                services.current_tribe,
                doc_type.value,
                limit,
                offset,
            )

        if not documents:
            st.warning("No documents found in this collection.")
            return

        st.success(
            f"📄 Found {len(documents)} documents "
            f"(showing {offset + 1}–{offset + len(documents)})"
        )

        for i, doc in enumerate(documents):
            with st.expander(
                f"📄 Document {offset + i + 1}: {doc['id']}",
                expanded=False,
            ):
                st.markdown("**Content:**")
                st.text_area(
                    "Document Text",
                    value=doc["text"],
                    height=150,
                    key=f"doc_text_{offset + i}",
                    label_visibility="collapsed",
                )

                st.markdown("**Metadata:**")
                metadata = doc["metadata"]

                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Document ID:** {doc['id']}")
                    st.write(
                        f"**Full Text Length:** "
                        f"{doc['full_text_length']} characters"
                    )
                    st.write(
                        f"**Source Key:** "
                        f"{metadata.get('source_key', 'Unknown')}"
                    )

                with col2:
                    st.write(
                        f"**Document Type:** "
                        f"{metadata.get('doc_type', 'Unknown')}"
                    )
                    st.write(f"**Tribe:** {metadata.get('tribe', 'Unknown')}")

                if metadata.get("created_at"):
                    st.write(f"**Created:** {metadata['created_at']}")

                if metadata.get("updated_at"):
                    st.write(f"**Updated:** {metadata['updated_at']}")

                if st.checkbox(
                    "Show Full Metadata",
                    key=f"show_metadata_{offset + i}",
                ):
                    st.json(metadata)

        if len(documents) == limit:
            st.info(
                "ℹ️ There may be more documents. "
                "Increase the offset to see additional documents."
            )

        if st.button("📤 Export Document List", key="export_docs"):
            self._export_documents(documents, doc_type)

    def _export_documents(
        self,
        documents: List[Dict[str, Any]],
        doc_type: DocumentType,
    ):
        """Export documents to JSON"""

        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "document_type": doc_type.value,
                "document_count": len(documents),
                "documents": documents,
            }

            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)

            st.download_button(
                label="⬇️ Download JSON",
                data=json_str,
                file_name=f"documents_{doc_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
        except Exception as e:
            st.error(f"Export failed: {e}")


def render_document_refresh_sidebar():
    """Render a compact version for sidebar"""

    st.sidebar.markdown("### 📊 Quick Document View")

    services = get_services()
    if not services.is_initialized():
        st.sidebar.warning("Select a tribe first")
        return

    doc_type = st.sidebar.selectbox(
        "Document Type:",
        ["Knowledge", "Code"],
        key="sidebar_doc_type"
    )

    if st.sidebar.button("🔍 View Documents", key="sidebar_view_docs"):
        st.session_state["show_document_viewer"] = True
        st.rerun()

    # Show quick stats
    try:
        doc_type_enum = (
            DocumentType.KNOWLEDGE
            if doc_type == "Knowledge"
            else DocumentType.CODE
        )
        stats = services.vector_store.get_collection_stats(doc_type_enum)
        st.sidebar.metric("📄 Documents", stats.get("document_count", 0))
    except Exception as e:
        st.sidebar.error(f"Stats error: {e}")


if __name__ == "__main__":
    # For testing
    render_document_refresh_page()
