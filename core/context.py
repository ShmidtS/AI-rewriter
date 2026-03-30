"""
GlobalContext: tracks narrative state across book blocks.
"""
from typing import List, Dict, Optional


class GlobalContext:
    def __init__(
        self,
        characters: Optional[List[Dict]] = None,
        plot_points: Optional[List[Dict]] = None,
        themes: Optional[List[str]] = None,
        style_notes: Optional[List[str]] = None,
    ):
        self.characters = characters or []
        self.plot_points = plot_points or []
        self.themes = themes or []
        self.style_notes = style_notes or []

    def to_json(self) -> Dict:
        return {
            "characters": self.characters,
            "plot_points": self.plot_points,
            "themes": self.themes,
            "style_notes": self.style_notes,
        }

    def update_from_response(self, context_update: Dict) -> None:
        if not context_update:
            return
        for char in context_update.get("characters", []):
            if char not in self.characters:
                self.characters.append(char)
        for point in context_update.get("plot_points", []):
            if point not in self.plot_points:
                self.plot_points.append(point)
        for theme in context_update.get("themes", []):
            if theme not in self.themes:
                self.themes.append(theme)
        for note in context_update.get("style_notes", []):
            if note not in self.style_notes:
                self.style_notes.append(note)

    @classmethod
    def from_json(cls, data: Dict) -> "GlobalContext":
        return cls(
            characters=data.get("characters", []),
            plot_points=data.get("plot_points", []),
            themes=data.get("themes", []),
            style_notes=data.get("style_notes", []),
        )
