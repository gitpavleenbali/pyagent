"""
SkillRegistry - Central registry for managing skills
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from pyai.skills.skill import Skill


@dataclass
class SkillMetadata:
    """Metadata about a registered skill"""

    skill: Skill
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    enabled: bool = True


class SkillRegistry:
    """
    SkillRegistry - Central registry for managing and discovering skills.

    The registry provides:
    - Skill registration and discovery
    - Skill lookup by name or tags
    - Skill lifecycle management
    - Tool definition generation

    Example:
        >>> registry = SkillRegistry()
        >>> registry.register(SearchSkill())
        >>> registry.register(CodeSkill(), tags=["development"])
        >>>
        >>> # Find skills
        >>> skill = registry.get("search")
        >>> dev_skills = registry.find_by_tag("development")
        >>>
        >>> # Get tool definitions for LLM
        >>> tools = registry.get_tool_definitions()
    """

    def __init__(self):
        self._skills: Dict[str, SkillMetadata] = {}
        self._tag_index: Dict[str, List[str]] = {}  # tag -> skill names

    def register(
        self,
        skill: Skill,
        tags: Optional[List[str]] = None,
        version: str = "1.0.0",
        replace: bool = False,
    ) -> None:
        """
        Register a skill with the registry.

        Args:
            skill: The skill to register
            tags: Optional tags for categorization
            version: Skill version
            replace: If True, replace existing skill with same name
        """
        if skill.name in self._skills and not replace:
            raise ValueError(
                f"Skill '{skill.name}' already registered. Use replace=True to override."
            )

        tags = tags or []
        metadata = SkillMetadata(
            skill=skill,
            tags=tags,
            version=version,
        )

        self._skills[skill.name] = metadata

        # Update tag index
        for tag in tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = []
            if skill.name not in self._tag_index[tag]:
                self._tag_index[tag].append(skill.name)

    def unregister(self, name: str) -> bool:
        """Remove a skill from the registry"""
        if name not in self._skills:
            return False

        metadata = self._skills.pop(name)

        # Remove from tag index
        for tag in metadata.tags:
            if tag in self._tag_index:
                self._tag_index[tag] = [n for n in self._tag_index[tag] if n != name]

        return True

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name"""
        metadata = self._skills.get(name)
        if metadata and metadata.enabled:
            return metadata.skill
        return None

    def get_metadata(self, name: str) -> Optional[SkillMetadata]:
        """Get skill metadata"""
        return self._skills.get(name)

    def find_by_tag(self, tag: str) -> List[Skill]:
        """Find all skills with a given tag"""
        skill_names = self._tag_index.get(tag, [])
        return [
            self._skills[name].skill
            for name in skill_names
            if name in self._skills and self._skills[name].enabled
        ]

    def find_by_tags(self, tags: List[str], match_all: bool = False) -> List[Skill]:
        """
        Find skills matching tags.

        Args:
            tags: Tags to search for
            match_all: If True, skill must have all tags. If False, any tag matches.
        """
        if not tags:
            return []

        if match_all:
            # Find skills with all specified tags
            matching = set(self._tag_index.get(tags[0], []))
            for tag in tags[1:]:
                matching &= set(self._tag_index.get(tag, []))
        else:
            # Find skills with any of the specified tags
            matching = set()
            for tag in tags:
                matching |= set(self._tag_index.get(tag, []))

        return [
            self._skills[name].skill
            for name in matching
            if name in self._skills and self._skills[name].enabled
        ]

    def enable(self, name: str) -> bool:
        """Enable a skill"""
        if name in self._skills:
            self._skills[name].enabled = True
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a skill (keeps it registered but inactive)"""
        if name in self._skills:
            self._skills[name].enabled = False
            return True
        return False

    def list_skills(self, include_disabled: bool = False) -> List[str]:
        """List all registered skill names"""
        if include_disabled:
            return list(self._skills.keys())
        return [name for name, meta in self._skills.items() if meta.enabled]

    def list_tags(self) -> List[str]:
        """List all registered tags"""
        return list(self._tag_index.keys())

    def get_all_skills(self, include_disabled: bool = False) -> List[Skill]:
        """Get all registered skills"""
        skills = []
        for meta in self._skills.values():
            if include_disabled or meta.enabled:
                skills.append(meta.skill)
        return skills

    def get_tool_definitions(self, skill_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get OpenAI-compatible tool definitions.

        Args:
            skill_names: Optional list of specific skills to include
        """
        tools = []

        for name, metadata in self._skills.items():
            if not metadata.enabled:
                continue
            if skill_names and name not in skill_names:
                continue

            tools.append(metadata.skill.to_tool_definition())

        return tools

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self):
        return iter(self.get_all_skills())


# Global default registry
_default_registry = SkillRegistry()


def get_default_registry() -> SkillRegistry:
    """Get the default global skill registry"""
    return _default_registry


def register_skill(
    skill: Optional[Skill] = None, tags: Optional[List[str]] = None, **kwargs
) -> Union[Skill, callable]:
    """
    Register a skill with the default registry.

    Can be used as a decorator or called directly.

    Example:
        >>> @register_skill(tags=["web"])
        ... class MySearchSkill(Skill):
        ...     pass
    """

    def decorator(skill_or_class):
        if isinstance(skill_or_class, type):
            # It's a class, instantiate it
            instance = skill_or_class()
        else:
            instance = skill_or_class

        _default_registry.register(instance, tags=tags, **kwargs)
        return skill_or_class

    if skill is not None:
        return decorator(skill)
    return decorator
