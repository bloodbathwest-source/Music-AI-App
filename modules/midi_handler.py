"""
MIDI Handler Module
Handles MIDI file creation and audio conversion
"""
import io
from midiutil import MIDIFile


class MIDIHandler:
    """Handles MIDI file creation and manipulation"""
    
    def __init__(self, tempo=120, instrument=0):
        """
        Initialize the MIDI handler
        
        Args:
            tempo: Tempo in BPM (beats per minute)
            instrument: MIDI instrument number (0-127)
        """
        self.tempo = tempo
        self.instrument = instrument
    
    def create_midi_file(self, melody, track_name="Generated Music"):
        """
        Create a MIDI file from a melody
        
        Args:
            melody: List of (note, duration, velocity) tuples
            track_name: Name of the MIDI track
            
        Returns:
            BytesIO buffer containing MIDI file data
        """
        # Create MIDI file with 1 track
        midi = MIDIFile(1)
        
        track = 0
        channel = 0
        time = 0
        
        # Add track name and tempo
        midi.addTrackName(track, time, track_name)
        midi.addTempo(track, time, self.tempo)
        
        # Add program change (instrument selection)
        midi.addProgramChange(track, channel, time, self.instrument)
        
        # Add notes to the track
        for note, duration, velocity in melody:
            midi.addNote(track, channel, note, time, duration, velocity)
            time += duration
        
        # Write to buffer
        midi_buffer = io.BytesIO()
        midi.writeFile(midi_buffer)
        midi_buffer.seek(0)
        
        return midi_buffer
    
    def melody_to_midi(self, individual, filename=None):
        """
        Convert a melody individual to MIDI format
        
        Args:
            individual: Dictionary containing melody
            filename: Optional filename to save to (if None, returns buffer)
            
        Returns:
            MIDI buffer or None (if saved to file)
        """
        melody = individual["melody"]
        midi_buffer = self.create_midi_file(melody)
        
        if filename:
            with open(filename, 'wb') as f:
                f.write(midi_buffer.getvalue())
            return None
        
        return midi_buffer
    
    def get_midi_stats(self, melody):
        """
        Get statistics about a melody
        
        Args:
            melody: List of (note, duration, velocity) tuples
            
        Returns:
            Dictionary containing statistics
        """
        notes = [note for note, _, _ in melody]
        durations = [duration for _, duration, _ in melody]
        velocities = [velocity for _, _, velocity in melody]
        
        total_duration = sum(durations)
        
        stats = {
            "total_notes": len(melody),
            "total_duration": total_duration,
            "min_note": min(notes),
            "max_note": max(notes),
            "note_range": max(notes) - min(notes),
            "avg_velocity": sum(velocities) / len(velocities),
            "unique_notes": len(set(notes)),
            "unique_durations": len(set(durations))
        }
        
        return stats
