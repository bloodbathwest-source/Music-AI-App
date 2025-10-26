"""
Lyric generation service
"""
import random
from typing import List, Dict


class LyricGenerator:
    """Generate song lyrics based on genre and emotion"""
    
    # Lyric templates and word banks
    VERSE_STRUCTURES = {
        'pop': [
            "{emotion_phrase}, {time_phrase}",
            "{action} in the {place}",
            "Can't {action} without {object}",
            "{feeling} when {condition}"
        ],
        'rock': [
            "{intensity} {action} {preposition} the {place}",
            "Won't {action} what they say",
            "{emotion_phrase}, breaking {object}",
            "Stand and {action} for {object}"
        ],
        'jazz': [
            "{time_phrase}, {emotion_phrase}",
            "Smooth {object} in the {place}",
            "Dreaming of {emotion_phrase}",
            "{feeling} like a {metaphor}"
        ],
        'classical': [
            "In {place} where {condition}",
            "{emotion_phrase} through the {time_phrase}",
            "Eternal {object}, {feeling}",
            "{metaphor} of {emotion_phrase}"
        ]
    }
    
    CHORUS_STRUCTURES = {
        'pop': [
            "{hook_word}, {hook_word}, {emotion_phrase}",
            "This is {emotion_phrase}",
            "Forever {feeling}, always {action}",
            "{hook_word} and {hook_word}"
        ],
        'rock': [
            "{hook_word}! {hook_word}! Breaking free",
            "We will {action}, we will {action}",
            "{intensity} {emotion_phrase}",
            "Never {action}, never {feeling}"
        ],
        'jazz': [
            "That {emotion_phrase}, oh {hook_word}",
            "Smooth and {feeling}",
            "{hook_word} in the {time_phrase}",
            "Keep on {action}, {hook_word}"
        ],
        'classical': [
            "Oh {emotion_phrase}, divine {object}",
            "Eternal {hook_word}, timeless {feeling}",
            "Through {time_phrase}, {emotion_phrase}",
            "{metaphor} eternal"
        ]
    }
    
    WORD_BANKS = {
        'emotion_phrase': {
            'happy': ['feeling the joy', 'riding the wave', 'dancing in light', 'sunshine bright'],
            'sad': ['tears in rain', 'lost in shadows', 'fading away', 'broken dreams'],
            'energetic': ['burning bright', 'electric nights', 'racing hearts', 'wild and free'],
            'calm': ['peaceful mind', 'gentle breeze', 'quiet moments', 'still waters'],
        },
        'action': ['dance', 'sing', 'move', 'run', 'fly', 'dream', 'believe', 'shine'],
        'place': ['night', 'sky', 'city', 'street', 'world', 'stars', 'heaven', 'darkness'],
        'object': ['love', 'dreams', 'heart', 'soul', 'light', 'time', 'music', 'hope'],
        'feeling': ['alive', 'free', 'lost', 'found', 'strong', 'brave', 'whole', 'real'],
        'time_phrase': ['tonight', 'always', 'forever', 'right now', 'every day', 'in time'],
        'condition': ['stars align', 'music plays', 'we meet', 'dreams come true', 'hearts collide'],
        'preposition': ['through', 'across', 'beyond', 'into', 'over', 'under'],
        'intensity': ['Hard', 'Fast', 'Loud', 'Strong', 'Wild', 'Deep'],
        'metaphor': ['phoenix', 'thunder', 'whisper', 'echo', 'flame', 'river'],
        'hook_word': ['Oh', 'Yeah', 'Hey', 'Whoa', 'Come on', 'Baby']
    }
    
    def __init__(self, genre: str = 'pop', emotion: str = 'happy'):
        """Initialize lyric generator"""
        self.genre = genre.lower()
        self.emotion = emotion.lower()
        
        # Use pop structures as fallback
        self.verse_templates = self.VERSE_STRUCTURES.get(self.genre, self.VERSE_STRUCTURES['pop'])
        self.chorus_templates = self.CHORUS_STRUCTURES.get(self.genre, self.CHORUS_STRUCTURES['pop'])
    
    def _fill_template(self, template: str) -> str:
        """Fill a template with random words from word banks"""
        result = template
        
        for placeholder, options in self.WORD_BANKS.items():
            if placeholder == 'emotion_phrase' and self.emotion in options:
                word = random.choice(options[self.emotion])
            elif isinstance(options, dict):
                # Skip emotion-specific words if we don't have the emotion
                continue
            else:
                word = random.choice(options)
            
            result = result.replace(f"{{{placeholder}}}", word)
        
        return result
    
    def generate_verse(self) -> List[str]:
        """Generate a verse (4 lines)"""
        lines = []
        templates = random.sample(self.verse_templates, min(4, len(self.verse_templates)))
        
        for template in templates:
            lines.append(self._fill_template(template))
        
        return lines
    
    def generate_chorus(self) -> List[str]:
        """Generate a chorus (4 lines)"""
        lines = []
        templates = random.sample(self.chorus_templates, min(4, len(self.chorus_templates)))
        
        for template in templates:
            lines.append(self._fill_template(template))
        
        return lines
    
    def generate_bridge(self) -> List[str]:
        """Generate a bridge (2-4 lines)"""
        # Bridge uses different templates
        bridge_templates = [
            "And when {condition}, {feeling}",
            "{emotion_phrase}, {time_phrase}",
            "This {object} will {action}",
            "{intensity} {feeling}, {action}"
        ]
        
        lines = []
        for template in random.sample(bridge_templates, random.randint(2, 4)):
            lines.append(self._fill_template(template))
        
        return lines
    
    def generate_full_lyrics(self) -> str:
        """Generate complete song lyrics with verse-chorus-verse-chorus-bridge-chorus structure"""
        sections = []
        
        # Verse 1
        sections.append("[Verse 1]")
        sections.extend(self.generate_verse())
        sections.append("")
        
        # Chorus 1
        sections.append("[Chorus]")
        sections.extend(self.generate_chorus())
        sections.append("")
        
        # Verse 2
        sections.append("[Verse 2]")
        sections.extend(self.generate_verse())
        sections.append("")
        
        # Chorus 2
        sections.append("[Chorus]")
        sections.extend(self.generate_chorus())
        sections.append("")
        
        # Bridge
        sections.append("[Bridge]")
        sections.extend(self.generate_bridge())
        sections.append("")
        
        # Final Chorus
        sections.append("[Chorus]")
        sections.extend(self.generate_chorus())
        
        return "\n".join(sections)
