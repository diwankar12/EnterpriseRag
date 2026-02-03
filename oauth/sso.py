import logging
import json
import time
import os
import requests
import jwt
import streamlit as st
import googleapiclient.discovery
from google.cloud import secretmanager

# Optional Vault support
try:
    import hvac
    HVAC_AVAILABLE = True
except Exception:
    hvac = None
    HVAC_AVAILABLE = False

# Authorization service
try:
    from src.services.authorization_service import AuthorizationService, UserPermissions
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from services.authorization_service import AuthorizationService, UserPermissions

# App config
try:
    from .. import config as global_config
except Exception:
    import config as global_config


# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------

def get_iam_client():
    """Build and return Google IAM client."""
    return googleapiclient.discovery.build("iam", "v1")


def _reportUnexpectedError(message="An unexpected error happened, please contact ideaFlow support"):
    logging.error(message)
    st.warning(message)
    st.stop()


# ------------------------------------------------------------------
# SSO Credentials Holder
# ------------------------------------------------------------------

class SSOCred:
    client_id = None
    secret = None
    sso_authcode_url = None
    ideaflow_redirecturl = None
    token_url = None


class SSOAuthResult:
    def __init__(self, valid_user=None, user_id=None, user_name=None, email=None, permissions=None):
        self.valid_user = valid_user
        self.user_id = user_id
        self.user_name = user_name
        self.email = email
        self.permissions = permissions  # UserPermissions object


# ------------------------------------------------------------------
# Secret loading
# ------------------------------------------------------------------

def _readClientIdAndSecret():
    """Reads client ID and secret from Vault or Google Secret Manager."""

    try:
        app_cfg = global_config.AppConfig()
        SSOCred.sso_authcode_url = getattr(app_cfg, "SSO_AUTHCODE_URL", None) or os.getenv("SSO_AUTHCODE_URL")
        base_url = getattr(app_cfg, "BASE_URL", None) or os.getenv("BASE_URL")
        if base_url:
            SSOCred.ideaflow_redirecturl = base_url
            SSOCred.token_url = getattr(app_cfg, "SSO_TOKEN_URL", None) or os.getenv("SSO_TOKEN_URL")
    except Exception:
        SSOCred.sso_authcode_url = os.getenv("SSO_AUTHCODE_URL")
        base_url = os.getenv("BASE_URL")
        if base_url:
            SSOCred.ideaflow_redirecturl = base_url
            SSOCred.token_url = os.getenv("SSO_TOKEN_URL")

    # Vault
    use_vault = os.getenv("VAULT_ENABLED", "false").lower() == "true"
    if use_vault:
        _readFromVault()

    # Google Secret Manager
    use_secret_manager = os.getenv("SECRET_MANAGER_ENABLED", "true").lower() == "true"
    if use_secret_manager:
        _readFromSecretManager()

    if not SSOCred.client_id or not SSOCred.secret:
        logging.error("No secret backend enabled for SSO credentials")
        _reportUnexpectedError(
            "Failed to retrieve SSO credentials. Check Vault or Secret Manager configuration."
        )


def _readFromVault():
    """Read SSO credentials from Vault."""
    if not HVAC_AVAILABLE:
        logging.error("hvac not installed but Vault enabled")
        return

    iam_client = get_iam_client()

    iam_role = os.getenv("VAULT_IAM_ROLE")
    service_account = os.getenv("VAULT_SERVICE_ACCOUNT")
    namespace = os.getenv("VAULT_NAMESPACE")
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    vault_path = os.getenv("VAULT_PATH")
    vault_url = os.getenv("VAULT_URL")

    client_id_key = os.getenv("VAULT_CLIENT_ID_KEY", "ideaflow-clientid")
    secret_key = os.getenv("VAULT_SECRET_KEY", "ideaflow-secret")

    now = int(time.time())
    payload = {
        "iat": now,
        "exp": now + 120,
        "sub": service_account,
        "aud": f"vault/{iam_role}"
    }

    name = f"projects/{project}/serviceAccounts/{service_account}"
    body = {"payload": json.dumps(payload)}

    request = iam_client.projects().serviceAccounts().signJwt(name=name, body=body)
    resp = request.execute()

    client = hvac.Client(url=vault_url, verify=False, namespace=namespace)
    client.auth.gcp.login(role=iam_role, jwt=resp["signedJwt"])

    secret_resp = client.secrets.kv.v2.read_secret(mount_point="kv", path=vault_path)
    SSOCred.client_id = secret_resp["data"]["data"][client_id_key]
    SSOCred.secret = secret_resp["data"]["data"][secret_key]


def _readFromSecretManager():
    """Read SSO credentials from Google Secret Manager."""
    project = os.getenv("PROJECT_ID")
    secret_id = os.getenv("SECRET_ID", "jira-app-secret-dev")
    version = os.getenv("SECRET_VERSION", "latest")

    client = secretmanager.SecretManagerServiceClient()
    secret_name = f"projects/{project}/secrets/{secret_id}/versions/{version}"
    response = client.access_secret_version(request={"name": secret_name})

    payload = response.payload.data.decode("utf-8")
    secret_json = json.loads(payload)

    SSOCred.client_id = secret_json.get("ideaflow-clientid")
    SSOCred.secret = secret_json.get("ideaflow-secret")


# ------------------------------------------------------------------
# SSO Flow
# ------------------------------------------------------------------

def _redirectToAuthentication():
    """Redirects user to SSO authorization endpoint."""
    auth_url = (
        f"{SSOCred.sso_authcode_url}"
        f"?client_id={SSOCred.client_id}"
        f"&redirect_uri={SSOCred.ideaflow_redirecturl}"
        f"&response_type=code"
        f"&scope=openid"
    )

    logging.info("Redirecting user to SSO provider")
    st.markdown(
        f'<meta http-equiv="refresh" content="0; url={auth_url}">',
        unsafe_allow_html=True,
    )
    st.stop()


def doSso():
    """Executes full SSO login flow."""
    logging.info("Starting SSO process")

    if "user-logged-in" not in st.session_state:
        _readClientIdAndSecret()

        if "code" not in st.query_params:
            _redirectToAuthentication()
            return SSOAuthResult(False)

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "authorization_code",
            "client_id": SSOCred.client_id,
            "client_secret": SSOCred.secret,
            "code": st.query_params["code"],
            "redirect_uri": SSOCred.ideaflow_redirecturl,
        }

        try:
            response = requests.post(SSOCred.token_url, headers=headers, data=data)
            if response.status_code != 200:
                logging.error("Token error: %s", response.text)
                _redirectToAuthentication()

            id_token = response.json()["id_token"]
            decoded = jwt.decode(id_token, options={"verify_signature": False})

            user_id = decoded.get("sub")
            groups = decoded.get("groups", [])
            groups_str = ",".join(groups)

            permissions = AuthorizationService.parse_user_permissions(
                user_id=user_id,
                groups_string=groups_str
            )

            auth_result = SSOAuthResult(
                valid_user=True,
                user_id=user_id,
                user_name=f"{decoded.get('lastName')} {decoded.get('firstName')}",
                email=decoded.get("mail"),
                permissions=permissions,
            )

            st.session_state["user-logged-in"] = auth_result
            return auth_result

        except Exception as ex:
            logging.error("SSO failed: %s", ex)
            _reportUnexpectedError("SSO authentication failed.")

    return st.session_state["user-logged-in"]


# ------------------------------------------------------------------
# Module entry (for local testing)
# ------------------------------------------------------------------

if __name__ == "__main__":
    doSso()
