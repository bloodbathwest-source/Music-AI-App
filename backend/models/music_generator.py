"""AI Music Generation Module using genetic algorithm and neural networks."""
import random
import json
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


class MusicLSTM(nn.Module):
    """LSTM model for music sequence generation."""
    
    def __init__(self, input_size=128, hidden_size=256, num_layers=2, output_size=128):
        super(MusicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, hidden=None):
        """Forward pass through the LSTM."""
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            hidden = (h0, c0)
            
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        out = self.softmax(out)
        return out, hidden


class MusicGenerator:
    """Advanced music generator with self-learning capabilities."""
    
    # Musical scale definitions
    SCALES = {
        'major': [0, 2, 4, 5, 7, 9, 11],
        'minor': [0, 2, 3, 5, 7, 8, 10],
        'dorian': [0, 2, 3, 5, 7, 9, 10],
        'phrygian': [0, 1, 3, 5, 7, 8, 10],
        'lydian': [0, 2, 4, 6, 7, 9, 11],
        'mixolydian': [0, 2, 4, 5, 7, 9, 10],
        'pentatonic': [0, 2, 4, 7, 9]
    }
    
    # Root note mappings
    ROOT_NOTES = {
        'C': 60, 'C#': 61, 'D': 62, 'D#': 63, 'E': 64, 'F': 65,
        'F#': 66, 'G': 67, 'G#': 68, 'A': 69, 'A#': 70, 'B': 71
    }
    
    # Genre-specific patterns
    GENRE_PATTERNS = {
        'pop': {'tempo': 120, 'complexity': 0.6, 'rhythm_variation': 0.7},
        'jazz': {'tempo': 140, 'complexity': 0.8, 'rhythm_variation': 0.9},
        'classical': {'tempo': 100, 'complexity': 0.9, 'rhythm_variation': 0.5},
        'rock': {'tempo': 130, 'complexity': 0.7, 'rhythm_variation': 0.8},
        'electronic': {'tempo': 128, 'complexity': 0.7, 'rhythm_variation': 0.6},
        'ambient': {'tempo': 80, 'complexity': 0.5, 'rhythm_variation': 0.4}
    }
    
    # Mood-based parameters
    MOOD_PARAMS = {
        'happy': {'velocity_range': (80, 110), 'note_density': 0.8},
        'sad': {'velocity_range': (50, 80), 'note_density': 0.5},
        'suspenseful': {'velocity_range': (60, 100), 'note_density': 0.6},
        'energetic': {'velocity_range': (90, 120), 'note_density': 0.9},
        'calm': {'velocity_range': (40, 70), 'note_density': 0.4},
        'dramatic': {'velocity_range': (70, 120), 'note_density': 0.7}
    }
    
    def __init__(self, feedback_data: Optional[List[Dict]] = None):
        """Initialize the music generator.
        
        Args:
            feedback_data: Historical feedback data for self-learning
        """
        self.lstm_model = MusicLSTM()
        self.feedback_data = feedback_data or []
        self.learned_patterns = self._analyze_feedback()
        
    def _analyze_feedback(self) -> Dict:
        """Analyze feedback data to improve generation."""
        if not self.feedback_data:
            return {}
        
        patterns = {}
        for feedback in self.feedback_data:
            key = f"{feedback.get('genre', 'unknown')}_{feedback.get('mood', 'unknown')}"
            if key not in patterns:
                patterns[key] = {
                    'avg_rating': 0,
                    'count': 0,
                    'parameters': []
                }
            
            rating = feedback.get('avg_rating', 0)
            params = feedback.get('parameters', '{}')
            if isinstance(params, str):
                params = json.loads(params)
            
            patterns[key]['avg_rating'] = (
                (patterns[key]['avg_rating'] * patterns[key]['count'] + rating) /
                (patterns[key]['count'] + 1)
            )
            patterns[key]['count'] += 1
            patterns[key]['parameters'].append(params)
        
        return patterns
    
    def generate_individual(
        self,
        genre: str,
        key_root: str,
        mode: str,
        mood: str,
        length: int = 32
    ) -> Dict:
        """Generate a single musical individual using genetic algorithm principles.
        
        Args:
            genre: Musical genre
            key_root: Root note of the key
            mode: Musical mode (scale type)
            mood: Emotional mood
            length: Number of notes to generate
            
        Returns:
            Dictionary containing melody, harmony, and metadata
        """
        # Get parameters
        root = self.ROOT_NOTES.get(key_root, 60)
        scale = self.SCALES.get(mode, self.SCALES['major'])
        genre_params = self.GENRE_PATTERNS.get(genre, self.GENRE_PATTERNS['pop'])
        mood_params = self.MOOD_PARAMS.get(mood, self.MOOD_PARAMS['happy'])
        
        # Generate scale notes across multiple octaves
        melody_notes = []
        for octave in range(-1, 3):
            for interval in scale:
                melody_notes.append(root + interval + (octave * 12))
        
        # Apply learned patterns if available
        pattern_key = f"{genre}_{mood}"
        if pattern_key in self.learned_patterns:
            complexity_boost = self.learned_patterns[pattern_key]['avg_rating'] / 5.0
            genre_params['complexity'] *= (1 + complexity_boost * 0.2)
        
        # Generate melody
        melody = []
        for i in range(length):
            note = random.choice(melody_notes)
            duration = random.choice([0.25, 0.5, 1.0, 1.5, 2.0])
            velocity = random.randint(*mood_params['velocity_range'])
            melody.append((note, duration, velocity))
        
        # Generate simple harmony (thirds and fifths)
        harmony = []
        for note, dur, vel in melody:
            if random.random() < 0.6:  # 60% chance of harmony
                interval = random.choice([3, 4, 7])  # thirds or fifth
                harmony_note = note - interval
                harmony.append((harmony_note, dur, vel - 10))
        
        # Generate rhythm pattern
        rhythm = self._generate_rhythm(length, genre_params['rhythm_variation'])
        
        return {
            'melody': melody,
            'harmony': harmony,
            'rhythm': rhythm,
            'tempo': genre_params['tempo'],
            'metadata': {
                'genre': genre,
                'mood': mood,
                'key': key_root,
                'mode': mode
            }
        }
    
    def _generate_rhythm(self, length: int, variation: float) -> List[int]:
        """Generate rhythm pattern.
        
        Args:
            length: Number of beats
            variation: Rhythm variation factor (0-1)
            
        Returns:
            List of rhythm values
        """
        rhythm = []
        for _ in range(length):
            if random.random() < variation:
                rhythm.append(random.choice([1, 2, 3, 4]))
            else:
                rhythm.append(1)
        return rhythm
    
    def evolve_music(
        self,
        genre: str,
        key_root: str,
        mode: str,
        mood: str,
        generations: int = 20,
        population_size: int = 20
    ) -> Dict:
        """Evolve music using genetic algorithm.
        
        Args:
            genre: Musical genre
            key_root: Root note
            mode: Musical mode
            mood: Emotional mood
            generations: Number of evolution generations
            population_size: Size of population per generation
            
        Returns:
            Best individual from evolution
        """
        # Initialize population
        population = [
            self.generate_individual(genre, key_root, mode, mood)
            for _ in range(population_size)
        ]
        
        for gen in range(generations):
            # Fitness evaluation (based on musical quality heuristics)
            population.sort(key=self._fitness, reverse=True)
            
            # Selection: keep top 50%
            survivors = population[:population_size // 2]
            
            # Crossover and mutation
            offspring = []
            for i in range(population_size // 2):
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child, key_root, mode, mood)
                offspring.append(child)
            
            population = survivors + offspring
        
        # Return best individual
        population.sort(key=self._fitness, reverse=True)
        return population[0]
    
    def _fitness(self, individual: Dict) -> float:
        """Calculate fitness score for an individual.
        
        Args:
            individual: Musical individual
            
        Returns:
            Fitness score
        """
        melody = individual['melody']
        score = 0.0
        
        # Melodic contour (prefer smooth movement)
        for i in range(len(melody) - 1):
            interval = abs(melody[i+1][0] - melody[i][0])
            if interval <= 2:
                score += 1.0
            elif interval <= 7:
                score += 0.5
        
        # Note variety
        unique_notes = len(set(note for note, _, _ in melody))
        score += unique_notes * 0.5
        
        # Rhythm consistency
        durations = [dur for _, dur, _ in melody]
        if len(set(durations)) > 1:
            score += 2.0
        
        return score
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover two parents to create offspring.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Child individual
        """
        crossover_point = len(parent1['melody']) // 2
        
        child = {
            'melody': parent1['melody'][:crossover_point] + parent2['melody'][crossover_point:],
            'harmony': parent1['harmony'][:crossover_point//2] + parent2['harmony'][crossover_point//2:],
            'rhythm': parent1['rhythm'][:crossover_point] + parent2['rhythm'][crossover_point:],
            'tempo': (parent1['tempo'] + parent2['tempo']) // 2,
            'metadata': parent1['metadata'].copy()
        }
        
        return child
    
    def _mutate(
        self,
        individual: Dict,
        key_root: str,
        mode: str,
        mood: str,
        mutation_rate: float = 0.1
    ) -> Dict:
        """Mutate an individual.

        Args:
            individual: Individual to mutate
            key_root: Root note
            mode: Musical mode
            mood: Emotional mood
            mutation_rate: Probability of mutation

        Returns:
            Mutated individual
        """
        mood_params = self.MOOD_PARAMS.get(mood, self.MOOD_PARAMS['happy'])
        root = self.ROOT_NOTES.get(key_root, 60)
        scale = self.SCALES.get(mode, self.SCALES['major'])
        
        melody_notes = []
        for octave in range(-1, 3):
            for interval in scale:
                melody_notes.append(root + interval + (octave * 12))
        
        # Mutate melody
        new_melody = []
        for note, dur, vel in individual['melody']:
            if random.random() < mutation_rate:
                note = random.choice(melody_notes)
                dur = random.choice([0.25, 0.5, 1.0, 1.5, 2.0])
                vel = random.randint(*mood_params['velocity_range'])
            new_melody.append((note, dur, vel))
        
        individual['melody'] = new_melody
        return individual
    
    def learn_from_feedback(self, feedback_data: List[Dict]):
        """Update the model with new feedback data.
        
        Args:
            feedback_data: New feedback entries
        """
        self.feedback_data.extend(feedback_data)
        self.learned_patterns = self._analyze_feedback()
