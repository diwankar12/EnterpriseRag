"""
GitHub repository cloning and processing components.
"""

import streamlit as st
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from ..app_config import get_services, tribe_supports_squads, get_available_squads
from .confluence import _clear_kb_cache, trigger_reseed_for_upload
from ..vector_store_interface import DocumentType


# ---------------------------------------------------------
# Squad selection (shared for GitHub uploads)
# ---------------------------------------------------------

def _render_squad_selection_github(component_key: str) -> Optional[str]:
    """Render squad selection for GitHub uploads"""

    current_tribe_title = st.session_state.get("tribe_selector", "Mortgage")
    current_tribe = current_tribe_title.lower() if current_tribe_title else "mortgage"

    if not tribe_supports_squads(current_tribe):
        return None

    available_squads = get_available_squads(current_tribe)
    if not available_squads:
        return None

    st.sidebar.subheader("📂 Squad Selection")

    selected_squad = st.sidebar.selectbox(
        "Select target squad:",
        options=available_squads,
        index=0,
        help="Choose squad database for GitHub content",
        key=f"squad_selection_{component_key}",
    )

    if selected_squad:
        st.sidebar.success(
            f"Target: **{current_tribe.title()} - {selected_squad}**"
        )

    return selected_squad


# ---------------------------------------------------------
# Main GitHub / GitLab Upload Entry
# ---------------------------------------------------------

def render_github_clone():
    st.sidebar.subheader("GitHub And GitLab Repository Upload")

    if "preferred_repo_type" not in st.session_state:
        st.session_state.preferred_repo_type = "Single Repository"

    repo_type = st.sidebar.radio(
        "Repository Upload Type:",
        ["Single Repository", "Multiple Repositories"],
        key="github_repo_type",
        help="Choose how to clone and ingest repositories",
    )

    st.session_state.preferred_repo_type = repo_type

    if repo_type == "Single Repository":
        render_single_repo_upload()
    elif repo_type == "Multiple Repositories":
        render_multiple_repo_upload()


# ---------------------------------------------------------
# Single Repository Upload
# ---------------------------------------------------------

def render_single_repo_upload():
    """Render interface for single repository upload"""

    st.sidebar.markdown("**Single Repository**")

    repo_url = st.sidebar.text_input(
        "GitHub Repository URL:",
        placeholder="https://github.com/owner/repo",
        help="Enter the full GitHub repository URL",
    )

    branch = st.sidebar.text_input(
        "Branch (optional):",
        placeholder="main",
        help="Leave empty to use default branch",
    )

    github_token = st.sidebar.text_input(
        "GitHub Token (optional):",
        type="password",
        help="Personal access token for private repositories",
        key="github_token_single",
    )

    include_docs = st.sidebar.checkbox(
        "Include documentation",
        value=True,
        help="Process .md, .rst, .txt files as knowledge documents",
    )

    with st.sidebar.expander("Advanced Options"):
        max_files = st.number_input(
            "Max files to process:",
            min_value=1,
            max_value=1000,
            value=500,
            key="single_max_files",
            help="Limit the number of files to process",
        )

        exclude_patterns = st.text_input(
            "Exclude patterns:",
            value="test,tests,node_modules,.git",
            key="single_exclude_patterns",
            help="Comma-separated patterns to exclude (e.g., test,docs,build)",
        )

    target_squad = _render_squad_selection_github("sidebar_single")

    if repo_url and st.sidebar.button(
        "Clone & Ingest Repository", key="clone_single_repo"
    ):
        process_single_repository(
            repo_url=repo_url,
            branch=branch,
            include_docs=include_docs,
            max_files=max_files,
            exclude_patterns=exclude_patterns,
            target_squad=target_squad,
        )


# ---------------------------------------------------------
# Multiple Repository Upload
# ---------------------------------------------------------

def render_multiple_repo_upload():
    """Render interface for multiple repository upload"""

    st.sidebar.markdown("**Multiple Repositories**")

    repo_urls = st.sidebar.text_area(
        "Repository URLs:",
        placeholder=(
            "Paste one or more URLs (one per line)\n"
            "https://github.com/owner/repo1\n"
            "https://github.com/owner/repo2"
        ),
        height=100,
        help="Enter one repository URL per line",
    )

    github_token = st.sidebar.text_input(
        "GitHub Token (optional):",
        type="password",
        help="Personal access token for private repositories",
        key="github_token_multi",
    )

    include_docs = st.sidebar.checkbox(
        "Include documentation",
        value=True,
        help="Process .md, .rst, .txt files as knowledge documents",
        key="multi_include_docs",
    )

    with st.sidebar.expander("Advanced Options"):
        max_files = st.number_input(
            "Max files per repo:",
            min_value=1,
            max_value=1000,
            value=500,
            key="multi_max_files",
        )

        exclude_patterns = st.text_input(
            "Exclude patterns:",
            value="test,tests,node_modules,.git",
            key="multi_exclude_patterns",
        )

    target_squad = _render_squad_selection_github("sidebar_multi")

    if repo_urls and st.sidebar.button(
        "Clone & Ingest Repositories", key="clone_multi_repo"
    ):
        for repo_url in repo_urls.splitlines():
            repo_url = repo_url.strip()
            if not repo_url:
                continue

            process_single_repository(
                repo_url=repo_url,
                branch="",
                include_docs=include_docs,
                max_files=max_files,
                exclude_patterns=exclude_patterns,
                target_squad=target_squad,
            )


# ---------------------------------------------------------
# Repository Processing
# ---------------------------------------------------------

"""
GitHub / GitLab repository cloning and ingestion (merged).
"""

import streamlit as st
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from ..app_config import get_services, tribe_supports_squads, get_available_squads
from .confluence import trigger_reseed_for_upload
from ..vector_store_interface import DocumentType


# =========================================================
# SQUAD SELECTION (DEDUPED)
# =========================================================

def _render_squad_selection(component_key: str) -> Optional[str]:
    current_tribe_title = st.session_state.get("tribe_selector", "Mortgage")
    current_tribe = current_tribe_title.lower() if current_tribe_title else "mortgage"

    if not tribe_supports_squads(current_tribe):
        return None

    squads = get_available_squads(current_tribe)
    if not squads:
        return None

    st.subheader("📂 Squad Selection")

    target_squad = st.selectbox(
        "Select target squad:",
        options=squads,
        index=0,
        key=f"squad_select_{component_key}",
        help="Choose squad database for repository content",
    )

    if target_squad:
        st.success(f"Target: **{current_tribe.title()} - {target_squad}**")

    return target_squad


# =========================================================
# MAIN UI
# =========================================================

def render_github_clone_main():
    """Render GitHub / GitLab clone UI (main content area)"""

    st.markdown("## Clone GitHub And GitLab Repository")

    repo_provider = st.selectbox(
        "Select Provider:",
        ["GitHub", "GitLab"],
        index=0,
        help="Choose whether to clone from GitHub or GitLab",
    )

    if repo_provider == "GitHub":
        url_placeholder = "https://github.com/owner/repo.git"
        token_help = "Personal Access Token (repo scope required)"
    else:
        url_placeholder = "https://gitlab.com/group/repo.git"
        token_help = "Personal Access Token (read_repository scope required)"

    col1, col2 = st.columns([2, 1])

    with col1:
        repo_url = st.text_input(
            f"{repo_provider} Repository URL",
            placeholder=url_placeholder,
            help="Enter full HTTPS repository URL",
        )

    with col2:
        branch = st.text_input(
            "Branch (optional)",
            placeholder="main",
            value="main",
            help="Leave empty to use main branch",
        )

    repo_token = st.text_input(
        f"{repo_provider} Token (optional, for private repos)",
        type="password",
        help=token_help,
        key=f"{repo_provider.lower()}_token_main",
    )

    include_docs = st.checkbox(
        "Include documentation files",
        value=True,
        help="Include .md, .rst, .txt, .docx files",
    )

    max_files = st.number_input(
        "Max files to process",
        min_value=1,
        max_value=1000,
        value=500,
    )

    exclude_patterns = st.text_input(
        "Exclude patterns",
        value="test,tests,node_modules,.git,__pycache__",
        help="Comma-separated patterns",
    )

    target_squad = _render_squad_selection("github_main")

    if st.button(
        "Clone & Ingest Repository",
        type="primary",
        disabled=not repo_url.strip(),
    ):
        process_single_repository(
            provider=repo_provider,
            repo_url=repo_url,
            branch=branch,
            token=repo_token,
            include_docs=include_docs,
            max_files=max_files,
            exclude_patterns=exclude_patterns,
            target_squad=target_squad,
        )


# =========================================================
# FILE WALK + CHUNK + UPSERT
# =========================================================

def _process_repository_files(
    services,
    repo_url: str,
    repo_name: str,
    path: str,
    branch: str,
    include_docs: bool,
    max_files: int,
    exclude_patterns: str,
    target_squad: Optional[str],
):
    exclude_set = {".git"} | {p.strip().lower() for p in exclude_patterns.split(",")}

    code_ext = {
        ".py", ".java", ".js", ".ts", ".go", ".cpp", ".c", ".rb", ".cs",
        ".yml", ".yaml", ".json", ".xml", ".html", ".css", ".scss",
        ".php", ".sql", ".sh", ".bat"
    }

    doc_ext = {".md", ".rst", ".txt", ".doc", ".docx"}

    files = [p for p in Path(path).rglob("*") if p.is_file()]
    total = min(len(files), max_files)
    progress = st.progress(0)
    processed = 0

    for idx, p in enumerate(files):
        if processed >= max_files:
            break

        parts = {part.lower() for part in p.parts}
        if any(e in parts for e in exclude_set):
            continue

        ext = p.suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".exe"}:
            continue

        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue
        except Exception:
            continue

        # Documents
        if ext in doc_ext and include_docs:
            chunks = services.chunker.chunk_text_semantic(text)
            doc_type = DocumentType.KNOWLEDGE

        # Code
        elif ext in code_ext:
            lang = services.chunker.get_file_language(str(p))
            chunks = services.chunker.chunk_code_hybrid(
                text=text,
                file_path=str(p.relative_to(path)),
                language=lang,
            )
            doc_type = DocumentType.CODE
        else:
            continue

        metadata = {
            "repo": repo_url,
            "repo_name": repo_name,
            "path": str(p.relative_to(path)),
            "file_type": ext,
            "branch": branch or "main",
        }

        if target_squad:
            metadata["target_squad"] = target_squad
            metadata["tribe"] = st.session_state.get(
                "tribe_selector", "mortgage"
            ).lower()

        services.vector_store.upsert_documents(
            doc_type,
            f"repo:{repo_name}",
            chunks,
            [metadata] * len(chunks),
        )

        processed += 1
        progress.progress(min(processed / total, 1.0))


# =========================================================
# CLONE LOGIC
# =========================================================

def clone_repo(
    provider: str,
    repo_url: str,
    token: Optional[str] = None,
    branch: Optional[str] = None,
) -> str:
    """Clone a GitHub or GitLab repository to a temporary directory"""

    temp_dir = tempfile.mkdtemp()
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    clone_path = Path(temp_dir) / repo_name

    auth_url = repo_url
    if token:
        auth_url = repo_url.replace("https://", f"https://oauth2:{token}@")

    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd.extend(["-b", branch])
    cmd.extend([auth_url, str(clone_path)])

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return str(clone_path)

# =========================================================
# MAIN PROCESSOR
# =========================================================


def process_single_repository(
    repo_url: str,
    branch: str,
    include_docs: bool,
    max_files: int,
    exclude_patterns: str,
    target_squad: str = None,
):
    """
    Process a single GitHub repository with enhanced features.
    (Merged from overlapping implementations)
    """

    show_progress = True
    enable_dedup = True

    services = get_services()

    if not services.is_initialized():
        st.sidebar.error("Services not initialized")
        return

    if not repo_url.strip():
        st.sidebar.error("Please enter a GitHub repository URL")
        return

    try:
        repo_name = repo_url.split("/")[-1].replace(".git", "")

        # --------------------------------------------------
        # Deduplication
        # --------------------------------------------------
        if enable_dedup:
            with st.spinner("Checking for existing repository content..."):
                existing_count = services.vector_store.delete_documents(
                    doc_type=DocumentType.CODE,
                    source_key=f"repo:{repo_name}",
                )

                if existing_count > 0:
                    st.sidebar.info(
                        f"Removed {existing_count} existing chunks for repository: {repo_name}"
                    )

        # --------------------------------------------------
        # Clone repository
        # --------------------------------------------------
        with st.spinner(f"Cloning repository: {repo_name}..."):
            path = clone_repo(repo_url, branch or None)

        # --------------------------------------------------
        # Parse exclude patterns
        # --------------------------------------------------
        base_exclude = {".git"}
        user_exclude = {
            p.strip().lower()
            for p in exclude_patterns.split(",")
            if p.strip()
        }
        exclude_set = base_exclude.union(user_exclude)

        # --------------------------------------------------
        # Process repository files
        # --------------------------------------------------
        with st.spinner("Processing repository files..."):
            code_cnt = 0
            doc_cnt = 0
            processed_files = 0
            skipped_files = 0

            total_files = sum(
                1 for p in Path(path).rglob("*") if p.is_file()
            )

            if show_progress:
                st.sidebar.info(f"Found {total_files} files in repository")

            for p in Path(path).rglob("*"):
                if not p.is_file() or processed_files >= max_files:
                    continue

                if p.suffix.lower() == ".pack":
                    skipped_files += 1
                    continue

                path_parts = [part.lower() for part in p.parts]
                if any(part in exclude_set for part in path_parts):
                    skipped_files += 1
                    continue

                ext = p.suffix.lower()

                if ext in {
                    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".exe",
                    ".class", ".zip", ".bin", ".ico", ".woff",
                    ".woff2", ".ttf"
                }:
                    skipped_files += 1
                    continue

                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                    if not txt.strip():
                        skipped_files += 1
                        continue
                except Exception:
                    skipped_files += 1
                    continue

                # ------------------------------------------
                # Determine document type
                # ------------------------------------------
                doc_extensions = {".md", ".rst", ".txt", ".doc", ".docx"}
                code_extensions = {
                    ".py", ".java", ".js", ".ts", ".go", ".cpp", ".c",
                    ".rb", ".cs", ".yml", ".yaml", ".json", ".xml",
                    ".html", ".css", ".scss", ".php", ".sql", ".sh",
                    ".bat", ".xsl", ".xslt"
                }

                if ext in doc_extensions and include_docs:
                    chunks = services.chunker.chunk_text_semantic(txt)
                    doc_type = DocumentType.KNOWLEDGE
                    doc_cnt += len(chunks)

                    metadata = {
                        "source": str(p),
                        "repo": repo_url,
                        "repo_name": repo_name,
                        "path": str(p.relative_to(path)),
                        "file_type": ext,
                        "branch": branch or "main",
                        "doc_type": doc_type.value,
                    }

                    if target_squad:
                        metadata["target_squad"] = target_squad
                        current_tribe_title = st.session_state.get(
                            "tribe_selector", "Mortgage"
                        )
                        metadata["tribe"] = (
                            current_tribe_title.lower()
                            if current_tribe_title
                            else "mortgage"
                        )

                    services.vector_store.upsert_documents(
                        doc_type,
                        f"repo:{repo_name}",
                        chunks,
                        [metadata] * len(chunks),
                    )

                elif ext in code_extensions:
                    file_language = services.chunker.get_file_language(str(p))
                    chunks_with_metadata = services.chunker.chunk_code_hybrid(
                        txt,
                        file_path=str(p.relative_to(path)),
                        language=file_language,
                    )

                    doc_type = DocumentType.CODE
                    code_cnt += len(chunks_with_metadata)

                    for chunk_text, chunk_metadata in chunks_with_metadata:
                        metadata = {
                            "source": str(p),
                            "repo": repo_url,
                            "repo_name": repo_name,
                            "path": str(p.relative_to(path)),
                            "file_type": ext,
                            "branch": branch or "main",
                            "doc_type": doc_type.value,
                            **chunk_metadata,
                        }

                        if target_squad:
                            metadata["target_squad"] = target_squad
                            current_tribe_title = st.session_state.get(
                                "tribe_selector", "Mortgage"
                            )
                            metadata["tribe"] = (
                                current_tribe_title.lower()
                                if current_tribe_title
                                else "mortgage"
                            )

                        services.vector_store.upsert_documents(
                            doc_type,
                            f"repo:{repo_name}",
                            [chunk_text],
                            [metadata],
                        )

                else:
                    skipped_files += 1
                    continue

                processed_files += 1

                if show_progress and processed_files % 10 == 0:
                    st.sidebar.info(
                        f"Processed {processed_files} files..."
                    )

        # --------------------------------------------------
        # Cleanup
        # --------------------------------------------------
        try:
            shutil.rmtree(path)
            if show_progress:
                st.sidebar.info("Cleaned up temporary repository files")
        except Exception as cleanup_error:
            st.sidebar.warning(
                f"Could not clean up temporary files: {cleanup_error}"
            )

        # --------------------------------------------------
        # Success summary + reseed
        # --------------------------------------------------
        total_chunks = code_cnt + doc_cnt
        st.sidebar.success(
            f"Successfully processed repository: {repo_name}"
        )
        st.sidebar.info(
            f"Processed {processed_files} files, stored {total_chunks} chunks"
        )

        if include_docs and doc_cnt > 0:
            st.sidebar.info(f"Documentation: {doc_cnt} chunks")

        if code_cnt > 0:
            st.sidebar.info(f"Code: {code_cnt} chunks")

        if skipped_files > 0:
            st.sidebar.info(f"Skipped {skipped_files} files")

        try:
            trigger_reseed_for_upload(services, target_squad)
        except Exception:
            pass

        _clear_kb_cache()

    except Exception as e:
        try:
            if "path" in locals() and Path(path).exists():
                shutil.rmtree(path)
        except Exception:
            pass

        st.sidebar.error(f"Error processing repository: {str(e)}")


def process_multiple_repositories(
    repos_text: str,
    include_docs: bool,
    max_repos: int,
    max_files_per_repo: int,
    exclude_patterns: str,
    target_squad: str = None,
):
    """
    Unified GitHub/GitLab repository ingestion pipeline.
    - Supports single & multiple repos
    - Handles clone, dedup, chunking, metadata, vector upsert
    - Always cleans up temp directories
    """

    show_progress = True
    enable_dedup = True

    services = get_services()

    if not services.is_initialized():
        st.sidebar.error("Services not initialized")
        return

    # Parse repository URLs
    repo_urls = [u.strip() for u in repos_text.strip().split("\n") if u.strip()]
    if not repo_urls:
        st.sidebar.error("No valid repository URLs found")
        return

    if len(repo_urls) > max_repos:
        st.sidebar.info(
            f"Found {len(repo_urls)} repositories, processing first {max_repos}"
        )
        repo_urls = repo_urls[:max_repos]

    total_processed = 0
    total_failed = 0
    total_chunks = 0
    total_deduplicated = 0

    for idx, repo_url in enumerate(repo_urls):
        path = None
        try:
            repo_name = repo_url.split("/")[-1].replace(".git", "")

            with st.spinner(f"Processing repository {idx+1}/{len(repo_urls)}: {repo_name}"):

                # Deduplication
                if enable_dedup:
                    existing = services.vector_store.delete_documents(
                        doc_type=DocumentType.CODE,
                        source_key=f"repo:{repo_name}",
                    )
                    total_deduplicated += existing or 0
                    if existing and show_progress:
                        st.sidebar.info(
                            f"Removed {existing} existing chunks for {repo_name}"
                        )

                # Clone repository
                path = clone_repo(repo_url, None)

                base_exclude = {".git"}
                user_exclude = {
                    p.strip().lower()
                    for p in exclude_patterns.split(",")
                    if p.strip()
                }
                exclude_set = base_exclude.union(user_exclude)

                code_cnt = 0
                doc_cnt = 0
                processed_files = 0
                skipped_files = 0

                total_files = sum(
                    1 for p in Path(path).rglob("*") if p.is_file()
                )
                if show_progress:
                    st.sidebar.info(f"Found {total_files} files in {repo_name}")

                for p in Path(path).rglob("*"):
                    if not p.is_file() or processed_files >= max_files_per_repo:
                        continue

                    if p.suffix.lower() == ".pack":
                        skipped_files += 1
                        continue

                    path_parts = [part.lower() for part in p.parts]
                    if any(part in exclude_set for part in path_parts):
                        skipped_files += 1
                        continue

                    ext = p.suffix.lower()
                    if ext in {
                        ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".exe",
                        ".class", ".zip", ".bin", ".ico", ".woff", ".woff2", ".ttf"
                    }:
                        skipped_files += 1
                        continue

                    try:
                        txt = p.read_text(encoding="utf-8", errors="ignore")
                        if not txt.strip():
                            skipped_files += 1
                            continue
                    except Exception:
                        skipped_files += 1
                        continue

                    doc_exts = {".md", ".rst", ".txt", ".doc", ".docx"}
                    code_exts = {
                        ".py", ".java", ".js", ".ts", ".go", ".cpp", ".c",
                        ".rb", ".cs", ".yml", ".yaml", ".json", ".xml",
                        ".html", ".css", ".scss", ".php", ".sql", ".sh",
                        ".bat", ".xsl", ".xslt"
                    }

                    if ext in doc_exts and include_docs:
                        chunks = services.chunker.chunk_text_semantic(txt)
                        doc_type = DocumentType.KNOWLEDGE
                        doc_cnt += len(chunks)
                        metadatas = []
                        for _ in chunks:
                            metadata = {
                                "source": str(p),
                                "repo": repo_url,
                                "repo_name": repo_name,
                                "path": str(p.relative_to(path)),
                                "file_type": ext,
                                "branch": "main",
                                "doc_type": doc_type.value,
                            }
                            if target_squad:
                                metadata["target_squad"] = target_squad
                                tribe = (
                                    st.session_state.get("tribe_selector", "Mortgage")
                                    .lower()
                                )
                                metadata["tribe"] = tribe
                            metadatas.append(metadata)

                        services.vector_store.upsert_documents(
                            doc_type,
                            f"repo:{repo_name}",
                            chunks,
                            metadatas,
                        )

                    elif ext in code_exts:
                        language = services.chunker.get_file_language(str(p))
                        chunks_with_meta = services.chunker.chunk_code_hybrid(
                            txt,
                            file_path=str(p.relative_to(path)),
                            language=language,
                        )
                        doc_type = DocumentType.CODE
                        code_cnt += len(chunks_with_meta)

                        for chunk_text, chunk_meta in chunks_with_meta:
                            metadata = {
                                "source": str(p),
                                "repo": repo_url,
                                "repo_name": repo_name,
                                "path": str(p.relative_to(path)),
                                "file_type": ext,
                                "branch": "main",
                                "doc_type": doc_type.value,
                                **chunk_meta,
                            }
                            if target_squad:
                                metadata["target_squad"] = target_squad
                                tribe = (
                                    st.session_state.get("tribe_selector", "Mortgage")
                                    .lower()
                                )
                                metadata["tribe"] = tribe

                            services.vector_store.upsert_documents(
                                doc_type,
                                f"repo:{repo_name}",
                                [chunk_text],
                                [metadata],
                            )

                    else:
                        skipped_files += 1
                        continue

                    processed_files += 1

                total_chunks += code_cnt + doc_cnt
                total_processed += 1

                if show_progress:
                    st.sidebar.success(
                        f"{repo_name}: {processed_files} files, {code_cnt + doc_cnt} chunks"
                    )

        except Exception as e:
            total_failed += 1
            st.sidebar.error(f"Failed to process {repo_url}: {str(e)}")

        finally:
            try:
                if path and Path(path).exists():
                    shutil.rmtree(path)
            except Exception:
                pass

    if total_processed > 0:
        st.sidebar.success(
            f"Processed {total_processed} repositories, {total_chunks} chunks stored"
        )
    if enable_dedup and total_deduplicated > 0:
        st.sidebar.info(f"Removed {total_deduplicated} duplicate chunks")
    if total_failed > 0:
        st.sidebar.warning(f"{total_failed} repositories failed")

    _clear_kb_cache()

    try:
        trigger_reseed_for_upload(services, target_squad)
    except Exception:
        pass


def process_repository_files(
    path,
    repo_url,
    repo_name,
    include_docs,
    max_files,
    services,
    target_squad=None,
):
    """
    Processes repository files, chunks documents/code, stores in vector DB,
    and clears KB cache — fully merged into a single method.
    """

    code_cnt = 0
    doc_cnt = 0
    processed_files = 0

    exclude_patterns = {
        "test", "tests", "__tests__", "spec", "specs",
        "node_modules", ".git", "__pycache__"
    }

    for p in Path(path).rglob("*"):
        if not p.is_file() or processed_files >= max_files:
            continue

        if p.suffix.lower() == ".pack":
            continue

        path_parts = [part.lower() for part in p.parts]
        if any(part in exclude_patterns for part in path_parts):
            continue

        ext = p.suffix.lower()

        # Skip binary files
        if ext in {
            ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".exe",
            ".class", ".zip", ".bin", ".ico", ".woff", ".woff2", ".ttf"
        }:
            continue

        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if not txt.strip():
                continue
        except Exception:
            continue

        doc_extensions = {".md", ".rst", ".txt"}
        code_extensions = {
            ".py", ".java", ".js", ".ts", ".go", ".cpp", ".c",
            ".rb", ".cs", ".yml", ".yaml", ".json", ".xml",
            ".html", ".css", ".scss", ".php", ".sql", ".sh",
            ".bat", ".xsl", ".xslt"
        }

        # ---------------- DOCUMENTS ----------------
        if ext in doc_extensions and include_docs:
            chunks = services.chunker.chunk_text_semantic(txt)
            doc_type = DocumentType.KNOWLEDGE
            doc_cnt += len(chunks)

            metadata_list = []
            for _ in chunks:
                metadata = {
                    "source": str(p),
                    "repo": repo_url,
                    "repo_name": repo_name,
                    "path": str(p.relative_to(path)),
                    "file_type": ext,
                    "branch": "main",
                    "doc_type": doc_type.value,
                }

                if target_squad:
                    metadata["target_squad"] = target_squad
                    tribe_title = st.session_state.get("tribe_selector", "Mortgage")
                    metadata["tribe"] = tribe_title.lower() if tribe_title else "mortgage"

                metadata_list.append(metadata)

            services.vector_store.upsert_documents(
                doc_type,
                f"repo:{repo_name}",
                chunks,
                metadata_list,
            )

        # ---------------- CODE ----------------
        elif ext in code_extensions:
            file_language = services.chunker.get_file_language(str(p))
            chunks_with_metadata = services.chunker.chunk_code_hybrid(
                txt,
                file_path=str(p.relative_to(path)),
                language=file_language,
            )

            doc_type = DocumentType.CODE
            code_cnt += len(chunks_with_metadata)

            for chunk_text, chunk_metadata in chunks_with_metadata:
                metadata = {
                    "source": str(p),
                    "repo": repo_url,
                    "repo_name": repo_name,
                    "path": str(p.relative_to(path)),
                    "file_type": ext,
                    "branch": "main",
                    "doc_type": doc_type.value,
                    **chunk_metadata,
                }

                if target_squad:
                    metadata["target_squad"] = target_squad
                    tribe_title = st.session_state.get("tribe_selector", "Mortgage")
                    metadata["tribe"] = tribe_title.lower() if tribe_title else "mortgage"

                services.vector_store.upsert_documents(
                    doc_type,
                    f"repo:{repo_name}",
                    [chunk_text],
                    [metadata],
                )

        else:
            continue

        processed_files += 1

    # ---------------- CLEAR KB CACHE (INLINED) ----------------
    for key in list(st.session_state.keys()):
        if key.startswith("kb_stats_") or key == "kb_stats_cache":
            del st.session_state[key]

    return code_cnt, doc_cnt, processed_files


def _clear_kb_cache():
    """Helper function to clear knowledge base cache"""
    for key in list(st.session_state.keys()):
        if key.startswith("kb_stats_") or key == "kb_stats_cache":
            del st.session_state[key]
