"""Specialized prompts for Kubernetes DevOps queries"""

from enum import Enum
from typing import Optional

_FORMAT_RULES = """
FORMATTING RULES (mandatory):
- Use markdown formatting in your response.
- Wrap ALL code, YAML, shell commands, and config snippets in fenced code blocks with the correct language tag, e.g. ```yaml, ```bash, ```json.
- Never output raw code outside a fenced block.
- Use headers (##) and bullet lists where they improve readability.
"""


class QueryType(Enum):
    """Types of DevOps queries"""

    CONFIGURATION = "configuration"
    TROUBLESHOOTING = "troubleshooting"
    BEST_PRACTICES = "best_practices"
    MIGRATION = "migration"
    SECURITY = "security"
    NETWORKING = "networking"
    EXAM = "exam"
    GENERAL = "general"


def classify_query(query: str) -> QueryType:
    """Classify the query type based on keywords"""
    query_lower = query.lower()

    # Configuration generation
    if any(
        word in query_lower
        for word in [
            "create",
            "configure",
            "setup",
            "deploy",
            "install",
            "yaml",
            "manifest",
        ]
    ):
        return QueryType.CONFIGURATION

    # Troubleshooting
    if any(
        word in query_lower
        for word in [
            "error",
            "fail",
            "not working",
            "issue",
            "problem",
            "debug",
            "why",
            "troubleshoot",
        ]
    ):
        return QueryType.TROUBLESHOOTING

    # Best practices
    if any(
        word in query_lower
        for word in [
            "best practice",
            "recommendation",
            "should",
            "production",
            "optimize",
        ]
    ):
        return QueryType.BEST_PRACTICES

    # Migration/upgrade
    if any(
        word in query_lower for word in ["migrate", "upgrade", "update", "version", "transition"]
    ):
        return QueryType.MIGRATION

    # Security
    if any(
        word in query_lower
        for word in ["security", "rbac", "policy", "secure", "permission", "auth"]
    ):
        return QueryType.SECURITY

    # Certification / exam prep
    if any(
        word in query_lower
        for word in [
            "certif",
            "exam",
            "cka",
            "ckad",
            "scenario",
            "compare",
            "difference between",
            "what is the",
        ]
    ):
        return QueryType.EXAM

    # Networking
    if any(
        word in query_lower for word in ["network", "service", "ingress", "cni", "connectivity"]
    ):
        return QueryType.NETWORKING

    return QueryType.GENERAL


class DevOpsPromptBuilder:
    """Build specialized prompts for DevOps queries"""

    @staticmethod
    def build_configuration_prompt(query: str, context: list[str]) -> str:
        """Prompt for configuration generation"""
        context_text = "\n\n".join(context)

        return f"""You are a Kubernetes expert. Based ONLY on the documentation provided below, answer the user's question accurately.
{_FORMAT_RULES}
CONTENT RULES:
- Use ONLY information from the provided documentation.
- Keep YAML examples simple and accurate.
- Do NOT add assumptions or extra features not mentioned in the docs.
- If information is not in the docs, say so.

Documentation:
{context_text}

Question: {query}

Answer:"""

    @staticmethod
    def build_troubleshooting_prompt(query: str, context: list[str]) -> str:
        """Prompt for troubleshooting assistance"""
        context_text = "\n\n".join(context)

        return f"""You are a Kubernetes troubleshooting expert.
{_FORMAT_RULES}
Based on the following documentation, help diagnose and resolve the issue.

Documentation Context:
{context_text}

Issue: {query}

Provide:
1. Most likely root cause(s)
2. Step-by-step debugging commands (in ```bash blocks)
3. Solution or workaround
4. How to verify the fix

Response:"""

    @staticmethod
    def build_best_practices_prompt(query: str, context: list[str]) -> str:
        """Prompt for best practices"""
        context_text = "\n\n".join(context)

        return f"""You are a Kubernetes best practices consultant.
{_FORMAT_RULES}
Based on the following documentation, provide production-ready recommendations.

Documentation Context:
{context_text}

Question: {query}

Provide:
1. Recommended approach and why
2. Key considerations for production
3. Common pitfalls to avoid
4. Example implementation (in fenced code blocks)

Response:"""

    @staticmethod
    def build_migration_prompt(query: str, context: list[str]) -> str:
        """Prompt for migration/upgrade guidance"""
        context_text = "\n\n".join(context)

        return f"""You are a Kubernetes migration expert.
{_FORMAT_RULES}
Based on the following documentation, provide migration/upgrade guidance.

Documentation Context:
{context_text}

Migration Request: {query}

Provide:
1. Pre-migration checklist
2. Step-by-step migration process (commands in ```bash blocks)
3. Breaking changes or compatibility issues
4. Rollback plan
5. Post-migration validation

Response:"""

    @staticmethod
    def build_security_prompt(query: str, context: list[str]) -> str:
        """Prompt for security-related queries"""
        context_text = "\n\n".join(context)

        return f"""You are a Kubernetes security expert.
{_FORMAT_RULES}
Based on the following documentation, provide security guidance.

Documentation Context:
{context_text}

Security Question: {query}

Provide:
1. Security best practices for this scenario
2. Configuration examples with security hardening (in ```yaml blocks)
3. Potential security risks and mitigations
4. Compliance considerations (if applicable)

Response:"""

    @staticmethod
    def build_networking_prompt(query: str, context: list[str]) -> str:
        """Prompt for networking-related queries"""
        context_text = "\n\n".join(context)

        return f"""You are a Kubernetes networking expert.
{_FORMAT_RULES}
Based on the following documentation, provide networking guidance.

Documentation Context:
{context_text}

Networking Question: {query}

Provide:
1. Network architecture explanation
2. Configuration for the requested scenario (in ```yaml blocks)
3. Traffic flow diagram (in text)
4. Troubleshooting commands for connectivity (in ```bash blocks)

Response:"""

    @staticmethod
    def build_exam_prompt(query: str, context: list[str]) -> str:
        """Prompt for certification exam preparation"""
        context_text = "\n\n".join(context)

        return f"""You are a certified Kubernetes instructor (CKA/CKAD).
{_FORMAT_RULES}
Based on the following official documentation, answer the question in an exam-preparation format.

Documentation Context:
{context_text}

Question: {query}

Structure your answer with these sections:

## Core Concept
Precise definition or explanation — what the examiner expects you to know.

## Key Details
Bullet points covering the most important facts, flags, or constraints.

## Practical Example
A concrete command or config snippet (in a fenced code block) that demonstrates the concept.

## Common Exam Traps
Specific mistakes or edge cases that appear frequently in exam scenarios.

Response:"""

    @staticmethod
    def build_general_prompt(query: str, context: list[str]) -> str:
        """Generic prompt for general queries"""
        context_text = "\n\n".join(context)

        return f"""Based ONLY on the following documentation, provide an accurate answer.
{_FORMAT_RULES}
CONTENT RULES:
- Use ONLY information from the provided documentation.
- Keep the answer concise and focused.
- Do NOT add assumptions or extra information not in the docs.
- If the documentation doesn't contain the answer, say so.

Documentation:
{context_text}

Question: {query}

Answer:"""

    @classmethod
    def build_prompt(
        cls, query: str, context: list[str], query_type: Optional[QueryType] = None
    ) -> str:
        """Build appropriate prompt based on query type"""
        if query_type is None:
            query_type = classify_query(query)

        prompt_builders = {
            QueryType.CONFIGURATION: cls.build_configuration_prompt,
            QueryType.TROUBLESHOOTING: cls.build_troubleshooting_prompt,
            QueryType.BEST_PRACTICES: cls.build_best_practices_prompt,
            QueryType.MIGRATION: cls.build_migration_prompt,
            QueryType.SECURITY: cls.build_security_prompt,
            QueryType.NETWORKING: cls.build_networking_prompt,
            QueryType.EXAM: cls.build_exam_prompt,
            QueryType.GENERAL: cls.build_general_prompt,
        }

        builder = prompt_builders.get(query_type, cls.build_general_prompt)
        return builder(query, context)


# Convenience functions
def build_devops_prompt(query: str, context: list[str]) -> str:
    """Build DevOps-optimized prompt (auto-classifies query type)"""
    return DevOpsPromptBuilder.build_prompt(query, context)


def build_prompt_for_type(query: str, context: list[str], query_type: QueryType) -> str:
    """Build prompt for specific query type"""
    return DevOpsPromptBuilder.build_prompt(query, context, query_type)
