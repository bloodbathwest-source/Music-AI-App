"""
Music Generator Module
Handles melody generation and evolution using genetic algorithms
"""
import random


class MusicGenerator:
    """Generates music using evolutionary algorithms and scale-based melody creation"""
    
    # Define musical scales
    SCALES = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
        "dorian": [0, 2, 3, 5, 7, 9, 10],
        "phrygian": [0, 1, 3, 5, 7, 8, 10],
        "lydian": [0, 2, 4, 6, 7, 9, 11],
        "mixolydian": [0, 2, 4, 5, 7, 9, 10],
        "locrian": [0, 1, 3, 5, 6, 8, 10]
    }
    
    # Map note names to MIDI note numbers
    ROOT_NOTES = {
        "C": 60, "C#": 61, "Db": 61,
        "D": 62, "D#": 63, "Eb": 63,
        "E": 64,
        "F": 65, "F#": 66, "Gb": 66,
        "G": 67, "G#": 68, "Ab": 68,
        "A": 69, "A#": 70, "Bb": 70,
        "B": 71
    }
    
    def __init__(self, key_root="C", mode="major", genre="pop", emotion="happy"):
        """
        Initialize the music generator
        
        Args:
            key_root: Root note of the key (e.g., "C", "D", "E")
            mode: Scale mode (e.g., "major", "minor", "dorian")
            genre: Music genre (e.g., "pop", "jazz", "classical", "rock")
            emotion: Emotional tone (e.g., "happy", "sad", "suspenseful")
        """
        self.key_root = key_root
        self.mode = mode
        self.genre = genre
        self.emotion = emotion
        self.root_note = self.ROOT_NOTES.get(key_root, 60)
        self.scale = self.SCALES.get(mode, self.SCALES["major"])
        self.melody_notes = self._get_melody_notes()
    
    def _get_melody_notes(self):
        """Generate available notes based on the key and mode"""
        return [self.root_note + interval + 12 for interval in self.scale]
    
    def generate_individual(self, length=32):
        """
        Generate a single individual for the genetic algorithm
        
        Args:
            length: Number of notes in the melody
            
        Returns:
            Dictionary containing melody as list of (note, duration, velocity) tuples
        """
        melody = []
        for _ in range(length):
            note = random.choice(self.melody_notes)
            duration = random.choice([0.5, 1, 1.5, 2])  # More varied durations
            velocity = random.randint(80, 120)  # Varied dynamics
            melody.append((note, duration, velocity))
        return {"melody": melody, "fitness": 0}
    
    def calculate_fitness(self, individual):
        """
        Calculate fitness score for an individual
        
        Args:
            individual: Dictionary containing melody
            
        Returns:
            Fitness score (higher is better)
        """
        melody = individual["melody"]
        fitness = 0
        
        # Prefer melodic contours (penalize large jumps)
        for i in range(len(melody) - 1):
            interval = abs(melody[i][0] - melody[i+1][0])
            if interval <= 2:  # Step motion
                fitness += 5
            elif interval <= 5:  # Small leap
                fitness += 2
            else:  # Large leap
                fitness -= 1
        
        # Reward variety in rhythm
        durations = [note[1] for note in melody]
        unique_durations = len(set(durations))
        fitness += unique_durations * 3
        
        # Reward dynamic variety
        velocities = [note[2] for note in melody]
        velocity_range = max(velocities) - min(velocities)
        fitness += velocity_range / 10
        
        # Genre-specific adjustments
        if self.genre == "jazz":
            # Reward syncopation and varied rhythms
            fitness += unique_durations * 2
        elif self.genre == "classical":
            # Reward smooth melodic motion
            fitness += sum(1 for i in range(len(melody) - 1) 
                          if abs(melody[i][0] - melody[i+1][0]) <= 2) * 2
        
        return fitness
    
    def evolve(self, generations=20, population_size=20, melody_length=32):
        """
        Evolve a melody using genetic algorithm
        
        Args:
            generations: Number of generations to evolve
            population_size: Size of the population
            melody_length: Length of each melody
            
        Returns:
            Best individual after evolution
        """
        # Initialize population
        population = [self.generate_individual(melody_length) for _ in range(population_size)]
        
        for gen in range(generations):
            # Calculate fitness for all individuals
            for individual in population:
                individual["fitness"] = self.calculate_fitness(individual)
            
            # Sort by fitness
            population.sort(key=lambda x: x["fitness"], reverse=True)
            
            # Selection: keep top 50%
            elite = population[:population_size // 2]
            
            # Crossover and mutation: generate new individuals
            offspring = []
            while len(offspring) < population_size // 2:
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                offspring.append(child)
            
            # Combine elite and offspring
            population = elite + offspring
        
        # Calculate final fitness and return best
        for individual in population:
            individual["fitness"] = self.calculate_fitness(individual)
        
        return max(population, key=lambda x: x["fitness"])
    
    def _crossover(self, parent1, parent2):
        """
        Perform crossover between two parents
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            
        Returns:
            Child individual
        """
        melody1 = parent1["melody"]
        melody2 = parent2["melody"]
        
        # Single-point crossover
        crossover_point = random.randint(1, len(melody1) - 1)
        child_melody = melody1[:crossover_point] + melody2[crossover_point:]
        
        return {"melody": child_melody, "fitness": 0}
    
    def _mutate(self, individual, mutation_rate=0.1):
        """
        Mutate an individual
        
        Args:
            individual: Individual to mutate
            mutation_rate: Probability of mutation for each note
            
        Returns:
            Mutated individual
        """
        melody = individual["melody"]
        mutated_melody = []
        
        for note, duration, velocity in melody:
            if random.random() < mutation_rate:
                # Mutate note
                note = random.choice(self.melody_notes)
            if random.random() < mutation_rate:
                # Mutate duration
                duration = random.choice([0.5, 1, 1.5, 2])
            if random.random() < mutation_rate:
                # Mutate velocity
                velocity = random.randint(80, 120)
            
            mutated_melody.append((note, duration, velocity))
        
        return {"melody": mutated_melody, "fitness": 0}
