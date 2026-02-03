"""
Authorization Service for IdeasFlow.

Handles entitlement parsing from SSO groups and permission checking.
"""

import re
from typing import List, Set, Optional
from dataclasses import dataclass


# -------------------------------------------------------------------
# User Permissions Model
# -------------------------------------------------------------------

@dataclass
class UserPermissions:
    """
    User permission structure extracted from SSO groups.
    """

    user_id: str
    is_superuser: bool
    accessible_tribes: Set[str]          # All tribes user can access
    tribe_admin_access: Set[str]          # Tribes where user has admin access
    tribe_basic_access: Set[str]          # Tribes where user has basic access

    def can_access_tribe(self, tribe_name: str) -> bool:
        """Check if user can access a specific tribe."""
        return self.is_superuser or tribe_name in self.accessible_tribes

    def has_knowledge_lake_access(self, tribe_name: str) -> bool:
        """Check if user has Knowledge Lake (chunking) access."""
        return self.is_superuser or tribe_name in self.tribe_admin_access

    def has_story_generation_access(self, tribe_name: str) -> bool:
        """Check if user has story generation access."""
        return (
            self.is_superuser
            or tribe_name in self.tribe_admin_access
            or tribe_name in self.tribe_basic_access
        )

    def has_chatbot_access(self, tribe_name: str) -> bool:
        """Check if user has chatbot access."""
        return (
            self.is_superuser
            or tribe_name in self.tribe_admin_access
            or tribe_name in self.tribe_basic_access
        )

    def has_dashboard_access(self) -> bool:
        """Dashboard access (superuser only)."""
        return self.is_superuser

    def get_permission_level(self, tribe_name: str) -> Optional[str]:
        """
        Get permission level for a tribe:
        'superuser', 'admin', 'basic', or None.
        """
        if self.is_superuser:
            return "superuser"
        if tribe_name in self.tribe_admin_access:
            return "admin"
        if tribe_name in self.tribe_basic_access:
            return "basic"
        return None


# -------------------------------------------------------------------
# Authorization Service
# -------------------------------------------------------------------

class AuthorizationService:
    """
    Service for parsing SSO groups and checking user permissions.
    """

    # Group name patterns
    SUPERUSER_PATTERN = r"CN=ideasflow_superuser_admin"
    TRIBE_ADMIN_PATTERN = r"CN=ideasflow_([a-zA-Z0-9_]+)_admin"
    TRIBE_BASIC_PATTERN = r"CN=ideasflow_([a-zA-Z0-9_]+)_basic"

    # Mapping from entitlement names to internal tribe names
    ENTITLEMENT_TO_TRIBE = {
        "mortgage": "mortgage",
        "altfi": "altfi",
        "mbh": "mortgage_batch_products",
        "rd": "risk_decisioning",
        "datax": "datax",
        "property": "property_gateway",
        "da": "data_analytics",
        "com": "commercial",
    }

    # Valid tribes
    VALID_TRIBES = set(ENTITLEMENT_TO_TRIBE.values())

    # ---------------------------------------------------------------
    # Group Parsing
    # ---------------------------------------------------------------

    @staticmethod
    def parse_cn_groups(groups_string: str) -> List[str]:
        """
        Extract CN (Common Name) values from LDAP distinguished name string.

        Example:
            Input:  "CN=ideasflow_superuser_admin,OU=USIS_Ideas_ideasFlow,..."
            Output: ["ideasflow_superuser_admin"]
        """
        if not groups_string:
            return []

        cn_pattern = r"CN=([^,]+)"
        return re.findall(cn_pattern, groups_string, re.IGNORECASE)

    # ---------------------------------------------------------------
    # Permission Parsing
    # ---------------------------------------------------------------

    @staticmethod
    def parse_user_permissions(
        user_id: str,
        groups_string: str
    ) -> UserPermissions:
        """
        Parse SSO groups string and extract user permissions.

        Priority hierarchy:
        1. Superuser takes precedence over everything
        2. Admin takes precedence over Basic for the same tribe
        3. If user has both admin and basic for same tribe, admin wins
        """

        cn_groups = AuthorizationService.parse_cn_groups(groups_string)

        is_superuser = False
        tribe_admin_access: Set[str] = set()
        tribe_basic_access: Set[str] = set()

        for group in cn_groups:
            # Superuser check (highest priority)
            if re.match(
                AuthorizationService.SUPERUSER_PATTERN,
                f"CN={group}"
            ):
                is_superuser = True
                continue

            # Tribe admin check
            admin_match = re.match(
                AuthorizationService.TRIBE_ADMIN_PATTERN,
                f"CN={group}"
            )
            if admin_match:
                entitlement_name = admin_match.group(1)
                tribe_name = AuthorizationService.ENTITLEMENT_TO_TRIBE.get(entitlement_name)
                if tribe_name:
                    tribe_admin_access.add(tribe_name)
                continue

            # Tribe basic check
            basic_match = re.match(
                AuthorizationService.TRIBE_BASIC_PATTERN,
                f"CN={group}"
            )
            if basic_match:
                entitlement_name = basic_match.group(1)
                tribe_name = AuthorizationService.ENTITLEMENT_TO_TRIBE.get(entitlement_name)
                if tribe_name:
                    tribe_basic_access.add(tribe_name)

        # Admin overrides basic
        tribe_basic_access = tribe_basic_access - tribe_admin_access

        # Combine all accessible tribes
        accessible_tribes = tribe_admin_access.union(tribe_basic_access)

        return UserPermissions(
            user_id=user_id,
            is_superuser=is_superuser,
            accessible_tribes=accessible_tribes,
            tribe_admin_access=tribe_admin_access,
            tribe_basic_access=tribe_basic_access,
        )

    # ---------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------

    @staticmethod
    def get_filtered_tribes(
        permissions: UserPermissions,
        all_tribes: List[str]
    ) -> List[str]:
        """
        Filter list of tribes based on user permissions.
        """
        if permissions.is_superuser:
            return all_tribes

        return [
            tribe
            for tribe in all_tribes
            if tribe in permissions.accessible_tribes
        ]
