from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class StoryService:
    """
    Service class responsible for story generation, feedback processing,
    validation, export, and statistics.
    """

    def __init__(self, agentic_rag, feedback_agent=None):
        self.agentic_rag = agentic_rag
        self.feedback_agent = feedback_agent

    # ------------------------------------------------------------------
    # Story Generation
    # ------------------------------------------------------------------
    def generate_story(
        self,
        user_story: str,
        tribe_name: str,
        include_code: bool = False,
        code_language: str = "python",
        temperature: float = 0.7,
        k_docs: int = 5,
        k_code: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate a new story based on user input.
        """
        try:
            logger.info(
                f"Generating story for tribe '{tribe_name}': {user_story[:100]}..."
            )

            tribe_suffix = f"_{tribe_name}"

            result = self.agentic_rag.generate_draft(
                one_liner=user_story,
                include_code=include_code,
                temperature=temperature,
                code_lang=code_language,
                tribe_suffix=tribe_suffix,
                k_docs=k_docs,
                k_code=k_code,
            )

            # Add metadata
            result["metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "tribe": tribe_name,
                "include_code": include_code,
                "code_language": code_language,
                "temperature": temperature,
                "k_docs": k_docs,
                "k_code": k_code,
                "user_input": user_story,
            }

            logger.info("Story generated successfully")
            return result

        except Exception as e:
            logger.error(f"Error generating story: {e}")
            raise

    # ------------------------------------------------------------------
    # Feedback Processing
    # ------------------------------------------------------------------
    def process_feedback(
        self,
        current_draft: Dict[str, Any],
        feedback: str,
        action_type: str = "smart",
    ) -> Dict[str, Any]:
        """
        Process user feedback on a story draft.
        """
        try:
            logger.info(
                f"Processing {action_type} feedback: {feedback[:100]}..."
            )

            if action_type == "smart" and self.feedback_agent:
                return self._process_smart_feedback(current_draft, feedback)

            elif action_type == "edit":
                return self._process_edit_feedback(current_draft, feedback)

            elif action_type == "regenerate":
                return self._process_regenerate_feedback(feedback)

            else:
                raise ValueError(f"Unknown action type: {action_type}")

        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            raise

    def _process_smart_feedback(
        self, current_draft: Dict[str, Any], feedback: str
    ) -> Dict[str, Any]:
        """Process feedback using intelligent feedback agent"""
        result = self.feedback_agent.process_feedback(
            draft=current_draft,
            feedback=feedback,
        )

        result["metadata"] = {
            "processed_at": datetime.now().isoformat(),
            "processing_type": "smart",
            "feedback": feedback,
        }

        return result

    def _process_edit_feedback(
        self, current_draft: Dict[str, Any], feedback: str
    ) -> Dict[str, Any]:
        """Process feedback as direct edit instructions"""
        new_draft = self.agentic_rag.apply_feedback(current_draft, feedback)

        return {
            "action": "edit",
            "draft": new_draft,
            "changes": "Direct edits applied",
            "metadata": {
                "processed_at": datetime.now().isoformat(),
                "processing_type": "edit",
                "feedback": feedback,
            },
        }

    def _process_regenerate_feedback(self, feedback: str) -> Dict[str, Any]:
        """Process feedback by regenerating the story"""
        if not self.feedback_agent:
            raise NotImplementedError(
                "Regeneration without feedback agent not implemented"
            )

        new_draft = self.feedback_agent.regenerate(feedback)

        return {
            "action": "regenerate",
            "draft": new_draft,
            "changes": "Story regenerated from scratch",
            "metadata": {
                "processed_at": datetime.now().isoformat(),
                "processing_type": "regenerate",
                "feedback": feedback,
            },
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_story(self, draft: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix a story draft.
        """
        try:
            logger.info("Validating story draft...")
            validated_draft = self.agentic_rag.validate_and_fix(draft)

            return {
                "draft": validated_draft,
                "metadata": {
                    "validated_at": datetime.now().isoformat(),
                },
            }

        except Exception as e:
            logger.error(f"Error validating story: {e}")
            raise

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_story_json(self, draft) -> str:
        """Export story as JSON string"""
        return draft.model_dump_json(indent=2)

    def export_story_markdown(self, draft) -> str:
        """Export story as Markdown"""
        markdown = f"""# {draft.title}

## Description
{draft.description}

## Acceptance Criteria
"""
        for i, criterion in enumerate(draft.acceptance_criteria, 1):
            markdown += f"{i}. {criterion}\n"

        markdown += f"""
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return markdown

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def get_story_statistics(self, draft) -> Dict[str, Any]:
        """Get statistics about a story draft"""
        return {
            "title_length": len(draft.title),
            "description_length": len(draft.description),
            "description_words": len(draft.description.split()),
            "acceptance_criteria_count": len(draft.acceptance_criteria),
            "total_criteria_length": sum(
                len(c) for c in draft.acceptance_criteria
            ),
            "estimated_story_points": self._estimate_story_points(draft),
        }

    def _estimate_story_points(self, draft) -> int:
        """Simple heuristic to estimate story points"""
        points = 1

        # Description complexity
        if len(draft.description.split()) > 100:
            points += 2
        elif len(draft.description.split()) > 50:
            points += 1

        # Acceptance criteria complexity
        if len(draft.acceptance_criteria) > 5:
            points += 2
        elif len(draft.acceptance_criteria) > 3:
            points += 1

        # Keyword-based complexity
        complexity_keywords = [
            "integration",
            "security",
            "performance",
            "database",
            "api",
            "authentication",
        ]

        text = f"{draft.title} {draft.description}".lower()
        for keyword in complexity_keywords:
            if keyword in text:
                points += 1

        return min(points, 13)  # Cap at 13 (Fibonacci max)
