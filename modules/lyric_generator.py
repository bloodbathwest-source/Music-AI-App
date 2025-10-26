"""
Lyric Generator Module
Generates lyrics based on themes, styles, and emotions
"""
import random


class LyricGenerator:
    """Generates song lyrics based on themes and styles"""
    
    # Lyric templates and word banks for different themes
    THEMES = {
        "love": {
            "adjectives": ["sweet", "tender", "endless", "beautiful", "gentle", "warm", "bright"],
            "nouns": ["heart", "love", "dreams", "light", "sky", "stars", "soul"],
            "verbs": ["hold", "feel", "shine", "dance", "sing", "fly", "dream"],
            "phrases": [
                "my heart beats for you",
                "in your arms I find home",
                "our love will never fade",
                "together we shine bright"
            ]
        },
        "sadness": {
            "adjectives": ["lonely", "dark", "cold", "empty", "broken", "lost", "gray"],
            "nouns": ["tears", "pain", "rain", "night", "shadow", "memories", "goodbye"],
            "verbs": ["cry", "fade", "fall", "lose", "break", "drift", "wander"],
            "phrases": [
                "tears fall like rain",
                "lost in the darkness",
                "memories haunt my mind",
                "can't find my way back"
            ]
        },
        "adventure": {
            "adjectives": ["wild", "free", "brave", "bold", "fearless", "endless", "vast"],
            "nouns": ["journey", "road", "mountains", "ocean", "sky", "horizon", "dreams"],
            "verbs": ["run", "chase", "explore", "discover", "climb", "soar", "venture"],
            "phrases": [
                "chasing the horizon",
                "running wild and free",
                "adventure calls my name",
                "beyond the distant shore"
            ]
        },
        "hope": {
            "adjectives": ["bright", "new", "rising", "golden", "promising", "shining", "peaceful"],
            "nouns": ["dawn", "future", "light", "hope", "dreams", "tomorrow", "sunrise"],
            "verbs": ["rise", "believe", "shine", "grow", "hope", "reach", "soar"],
            "phrases": [
                "tomorrow brings new light",
                "hope will guide the way",
                "rising from the ashes",
                "believe in better days"
            ]
        }
    }
    
    # Verse and chorus structures
    STRUCTURES = {
        "pop": ["verse", "verse", "chorus", "verse", "chorus", "bridge", "chorus"],
        "rock": ["verse", "chorus", "verse", "chorus", "bridge", "chorus", "chorus"],
        "jazz": ["verse", "verse", "chorus", "verse", "bridge", "verse"],
        "classical": ["verse", "verse", "verse", "chorus"]
    }
    
    def __init__(self, theme="love", style="pop", emotion="happy"):
        """
        Initialize the lyric generator
        
        Args:
            theme: Theme of the lyrics (e.g., "love", "sadness", "adventure", "hope")
            style: Musical style (e.g., "pop", "rock", "jazz", "classical")
            emotion: Emotional tone (e.g., "happy", "sad", "suspenseful")
        """
        self.theme = theme
        self.style = style
        self.emotion = emotion
        self.word_bank = self.THEMES.get(theme, self.THEMES["love"])
        self.structure = self.STRUCTURES.get(style, self.STRUCTURES["pop"])
    
    def generate_line(self):
        """
        Generate a single line of lyrics
        
        Returns:
            A line of lyrics
        """
        patterns = [
            lambda: f"{random.choice(self.word_bank['adjectives'])} {random.choice(self.word_bank['nouns'])} {random.choice(self.word_bank['verbs'])}",
            lambda: f"I {random.choice(self.word_bank['verbs'])} the {random.choice(self.word_bank['adjectives'])} {random.choice(self.word_bank['nouns'])}",
            lambda: f"When {random.choice(self.word_bank['nouns'])} {random.choice(self.word_bank['verbs'])}",
            lambda: random.choice(self.word_bank['phrases']),
            lambda: f"The {random.choice(self.word_bank['nouns'])} is {random.choice(self.word_bank['adjectives'])}",
        ]
        
        return random.choice(patterns)().capitalize()
    
    def generate_verse(self, lines=4):
        """
        Generate a verse
        
        Args:
            lines: Number of lines in the verse
            
        Returns:
            List of lyric lines forming a verse
        """
        return [self.generate_line() for _ in range(lines)]
    
    def generate_chorus(self, lines=4):
        """
        Generate a chorus (with some repetition for catchiness)
        
        Args:
            lines: Number of lines in the chorus
            
        Returns:
            List of lyric lines forming a chorus
        """
        chorus_lines = []
        hook = self.generate_line()  # Main hook line
        
        for i in range(lines):
            if i % 2 == 0:
                chorus_lines.append(hook)
            else:
                chorus_lines.append(self.generate_line())
        
        return chorus_lines
    
    def generate_bridge(self, lines=4):
        """
        Generate a bridge (with different mood/perspective)
        
        Args:
            lines: Number of lines in the bridge
            
        Returns:
            List of lyric lines forming a bridge
        """
        return [self.generate_line() for _ in range(lines)]
    
    def generate_lyrics(self):
        """
        Generate complete song lyrics following the structure
        
        Returns:
            Dictionary containing structured lyrics
        """
        lyrics = {
            "title": self._generate_title(),
            "sections": []
        }
        
        verse_count = 1
        chorus_lines = None  # Reuse same chorus
        
        for section_type in self.structure:
            section = {"type": section_type, "lines": []}
            
            if section_type == "verse":
                section["lines"] = self.generate_verse()
                section["label"] = f"Verse {verse_count}"
                verse_count += 1
            elif section_type == "chorus":
                if chorus_lines is None:
                    chorus_lines = self.generate_chorus()
                section["lines"] = chorus_lines
                section["label"] = "Chorus"
            elif section_type == "bridge":
                section["lines"] = self.generate_bridge()
                section["label"] = "Bridge"
            
            lyrics["sections"].append(section)
        
        return lyrics
    
    def _generate_title(self):
        """
        Generate a song title
        
        Returns:
            Song title string
        """
        title_patterns = [
            lambda: f"{random.choice(self.word_bank['adjectives']).title()} {random.choice(self.word_bank['nouns']).title()}",
            lambda: f"{random.choice(self.word_bank['phrases']).title()}",
            lambda: f"The {random.choice(self.word_bank['nouns']).title()}",
        ]
        
        return random.choice(title_patterns)()
    
    def format_lyrics(self, lyrics):
        """
        Format lyrics as a string for display
        
        Args:
            lyrics: Dictionary containing structured lyrics
            
        Returns:
            Formatted lyrics string
        """
        output = [f"**{lyrics['title']}**\n"]
        
        for section in lyrics["sections"]:
            output.append(f"\n[{section['label']}]")
            for line in section["lines"]:
                output.append(line)
        
        return "\n".join(output)
