# config.py

import os
import json
import ssl
from dotenv import load_dotenv

import vertexai
from google.cloud import aiplatform

# ---- Workaround for SSL verification issues in corporate environments ----
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# ---- Import Secret Manager only when needed (for Kubernetes mode) ----
try:
    from google.cloud import secretmanager
    SECRETMANAGER_AVAILABLE = True
except ImportError:
    SECRETMANAGER_AVAILABLE = False


class AppConfig:
    def __init__(self):
        # Check if running in a Kubernetes environment
        is_in_kubernetes = "KUBERNETES_SERVICE_HOST" in os.environ

        if is_in_kubernetes:
            print(
                "Running in Kubernetes mode. Fetching all config from a single secret "
                "in Google Secret Manager."
            )
            self._load_all_from_secret_manager()
        else:
            print("Running in local mode. Loading configuration from .env file.")
            load_dotenv()
            self._load_from_env()

        # ---- Common Initialization for GCP ----
        # ---- Set SSL Bundle for requests ----
        if hasattr(self, "SSL_CERT_PATH") and self.SSL_CERT_PATH:
            if os.path.exists(self.SSL_CERT_PATH):
                print(
                    f"Found SSL cert bundle. Setting REQUESTS_CA_BUNDLE to: "
                    f"{self.SSL_CERT_PATH}"
                )
                os.environ["REQUESTS_CA_BUNDLE"] = self.SSL_CERT_PATH
            else:
                print(
                    f"Warning: SSL_CERT_PATH '{self.SSL_CERT_PATH}' not found. "
                    "SSL verification may fail."
                )

        # Initialize Vertex AI
        vertexai.init(project=self.PROJECT_ID, location=self.REGION)
        aiplatform.init(project=self.PROJECT_ID, location=self.REGION)

    # ------------------------------------------------------------------
    # Load configuration from .env
    # ------------------------------------------------------------------
    def _load_from_env(self):
        """Loads all configuration from a local .env file."""

        # ---- Vector Store ----
        self.VECTOR_STORE_PROVIDER = os.getenv("VECTOR_STORE_PROVIDER", "chromadb")
        self.CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db_store")
        self.CHROMA_COLLECTION_NAME = os.getenv(
            "CHROMA_COLLECTION_NAME", "jira_rag_collection"
        )

        # ---- GCP / Project ----
        self.PROJECT_ID = os.getenv("PROJECT_ID")
        self.REGION = os.getenv("REGION")

        # ---- Jira ----
        self.JIRA_API_URL = os.getenv("JIRA_API_URL")
        self.JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")
        self.JIRA_EMAIL = os.getenv("JIRA_EMAIL")
        self.JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

        # ---- Vector Search ----
        self.EXPECTED_DIM = 768
        self.INDEX_ID = os.getenv("INDEX_ID")
        self.ENDPOINT_ID = os.getenv("ENDPOINT_ID")
        self.DEPLOYED_INDEX_ID = os.getenv("DEPLOYED_INDEX_ID")

        # ---- SSL ----
        self.SSL_CERT_PATH = os.getenv("SSL_CERT_PATH")

        # ---- AlloyDB ----
        self.ALLOYDB_HOST = os.getenv("ALLOYDB_HOST")
        self.ALLOYDB_PORT = os.getenv("ALLOYDB_PORT", 5432)
        self.ALLOYDB_DB = os.getenv("ALLOYDB_DB")
        self.ALLOYDB_USER = os.getenv("ALLOYDB_USER")
        self.ALLOYDB_PASS = os.getenv("ALLOYDB_PASS")

        # ---- Gemini ----
        self.GEMINI_MODEL_NAME = os.getenv(
            "GEMINI_MODEL_NAME", "gemini-2.5-flash"
        )
        self.GEMINI_EMBED_MODEL_NAME = os.getenv(
            "GEMINI_EMBED_MODEL_NAME", "text-embedding-004"
        )
        self.CROSS_ENCODER_MODEL_PATH = os.getenv(
            "CROSS_ENCODER_MODEL_PATH", None
        )

        # ---- Feedback Agent ----
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
        self.TIMEOUT = int(os.getenv("TIMEOUT", "30"))

        # ---- Terminology ----
        try:
            self.TERMINOLOGY = json.loads(os.getenv("TERMINOLOGY", "{}"))
        except json.JSONDecodeError:
            self.TERMINOLOGY = {}

        # ---- SSO ----
        self.SSO_AUTHCODE_URL = os.getenv(
            "SSO_AUTHCODE_URL",
            "https://fedssoqa.equifax.com/as/authorization.oauth2",
        )
        self.SSO_TOKEN_URL = os.getenv(
            "SSO_TOKEN_URL",
            "https://fedssoqa.equifax.com/as/token.oauth2",
        )
        self.BASE_URL = os.getenv(
            "BASE_URL",
            "https://spcfn-api.dev.usel.npe.usis.gcp.efx/jira-story-generator/",
        )

    # ------------------------------------------------------------------
    # Load configuration from Google Secret Manager (Kubernetes)
    # ------------------------------------------------------------------
    def _load_all_from_secret_manager(self):
        """Loads all configuration from a single JSON secret in Google Secret Manager."""

        if not SECRETMANAGER_AVAILABLE:
            raise RuntimeError(
                "Secret Manager not available. Install google-cloud-secret-manager package."
            )

        self.VECTOR_STORE_PROVIDER = os.getenv("VECTOR_STORE_PROVIDER", "chromadb")
        self.PROJECT_ID = os.getenv("PROJECT_ID")

        if not self.PROJECT_ID:
            raise ValueError("PROJECT_ID environment variable must be set in deployment.")

        self.SECRET_ID = os.getenv("SECRET_ID")
        if not self.SECRET_ID:
            raise ValueError("SECRET_ID environment variable must be set in deployment.")

        self.BASE_URL = os.getenv("BASE_URL")
        self.SSO_TOKEN_URL = os.getenv("SSO_TOKEN_URL")
        self.SSO_AUTHCODE_URL = os.getenv("SSO_AUTHCODE_URL")

        client = secretmanager.SecretManagerServiceClient()
        secret_name = (
            f"projects/{self.PROJECT_ID}/secrets/{self.SECRET_ID}/versions/latest"
        )

        response = client.access_secret_version(request={"name": secret_name})
        config_data = json.loads(response.payload.data.decode("UTF-8"))

        # ---- Vector Store ----
        self.VECTOR_STORE_PROVIDER = config_data.get(
            "VECTOR_STORE_PROVIDER", "chromadb"
        )
        self.CHROMA_DB_PATH = config_data.get(
            "CHROMA_DB_PATH", "./chroma_db_store"
        )
        self.CHROMA_COLLECTION_NAME = config_data.get(
            "CHROMA_COLLECTION_NAME", "jira_rag_collection"
        )

        # ---- GCP ----
        self.REGION = config_data.get("REGION")

        # ---- Jira ----
        self.JIRA_API_URL = config_data.get("JIRA_API_URL")
        self.JIRA_PROJECT_KEY = config_data.get("JIRA_PROJECT_KEY")
        self.JIRA_EMAIL = config_data.get("JIRA_EMAIL")
        self.JIRA_API_TOKEN = config_data.get("JIRA_API_TOKEN")

        # ---- Vector Search ----
        self.EXPECTED_DIM = config_data.get("EXPECTED_DIM", 768)
        self.INDEX_ID = config_data.get("INDEX_ID")
        self.ENDPOINT_ID = config_data.get("ENDPOINT_ID")
        self.DEPLOYED_INDEX_ID = config_data.get("DEPLOYED_INDEX_ID")

        # ---- SSL ----
        self.SSL_CERT_PATH = config_data.get("SSL_CERT_PATH")

        # ---- AlloyDB ----
        self.ALLOYDB_HOST = config_data.get("ALLOYDB_HOST")
        self.ALLOYDB_PORT = config_data.get("ALLOYDB_PORT", 5432)
        self.ALLOYDB_DB = config_data.get("ALLOYDB_DB")
        self.ALLOYDB_USER = config_data.get("ALLOYDB_USER")
        self.ALLOYDB_PASS = config_data.get("ALLOYDB_PASS")

        # ---- Gemini ----
        self.GEMINI_MODEL_NAME = config_data.get(
            "GEMINI_MODEL_NAME", "gemini-2.5-flash"
        )
        self.GEMINI_EMBED_MODEL_NAME = config_data.get(
            "GEMINI_EMBED_MODEL_NAME", "text-embedding-004"
        )
        self.CROSS_ENCODER_MODEL_PATH = config_data.get(
            "CROSS_ENCODER_MODEL_PATH", None
        )

        # ---- Feedback Agent ----
        self.MAX_RETRIES = int(config_data.get("MAX_RETRIES", "3"))
        self.TIMEOUT = int(config_data.get("TIMEOUT", "30"))

        # ---- Terminology ----
        self.TERMINOLOGY = config_data.get("TERMINOLOGY", {})
