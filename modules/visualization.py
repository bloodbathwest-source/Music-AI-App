"""
Visualization Module
Handles melody plotting and album art generation
"""
import matplotlib.pyplot as plt
import io
import random
from PIL import Image, ImageDraw, ImageFont


class Visualizer:
    """Handles visualization of music data and album art generation"""
    
    def plot_melody(self, melody, title="Melody Visualization"):
        """
        Create a visualization of the melody
        
        Args:
            melody: List of (note, duration, velocity) tuples
            title: Title for the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Extract data
        notes = [note for note, _, _ in melody]
        durations = [duration for _, duration, _ in melody]
        velocities = [velocity for _, _, velocity in melody]
        
        # Calculate time positions
        time_positions = [0]
        for duration in durations[:-1]:
            time_positions.append(time_positions[-1] + duration)
        
        # Plot 1: Note progression
        ax1.plot(time_positions, notes, marker='o', linestyle='-', linewidth=2, markersize=4)
        ax1.set_xlabel('Time (beats)', fontsize=12)
        ax1.set_ylabel('MIDI Note Number', fontsize=12)
        ax1.set_title(f'{title} - Note Progression', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add note names to y-axis
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        y_ticks = ax1.get_yticks()
        y_labels = [f"{note_names[int(y) % 12]}{int(y) // 12 - 1}" for y in y_ticks]
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels(y_labels)
        
        # Plot 2: Velocity over time
        ax2.bar(time_positions, velocities, width=[d * 0.8 for d in durations], 
                alpha=0.7, color='coral')
        ax2.set_xlabel('Time (beats)', fontsize=12)
        ax2.set_ylabel('Velocity', fontsize=12)
        ax2.set_title('Dynamics (Velocity)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_to_buffer(self, fig):
        """
        Convert a matplotlib figure to a BytesIO buffer
        
        Args:
            fig: Matplotlib figure object
            
        Returns:
            BytesIO buffer containing PNG image
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf
    
    def generate_album_art(self, title, artist="AI Music Generator", theme="love", size=(800, 800)):
        """
        Generate simple album art
        
        Args:
            title: Song/album title
            artist: Artist name
            theme: Visual theme for colors
            size: Size of the image (width, height)
            
        Returns:
            PIL Image object
        """
        # Create image with gradient background
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)
        
        # Color schemes based on theme
        color_schemes = {
            "love": [(255, 182, 193), (255, 105, 180)],  # Pink tones
            "sadness": [(70, 130, 180), (25, 25, 112)],   # Blue tones
            "adventure": [(255, 140, 0), (255, 215, 0)],  # Orange/gold tones
            "hope": [(255, 215, 0), (255, 255, 224)],     # Yellow/light tones
            "pop": [(255, 105, 180), (138, 43, 226)],     # Pink/purple
            "jazz": [(72, 61, 139), (218, 165, 32)],      # Dark blue/gold
            "classical": [(139, 0, 0), (220, 20, 60)],    # Dark red tones
            "rock": [(105, 105, 105), (255, 69, 0)]       # Gray/red tones
        }
        
        colors = color_schemes.get(theme, color_schemes["love"])
        
        # Create gradient background
        for y in range(size[1]):
            ratio = y / size[1]
            r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
            g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
            b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
            draw.line([(0, y), (size[0], y)], fill=(r, g, b))
        
        # Add geometric shapes for visual interest
        self._add_geometric_shapes(draw, size, colors)
        
        # Add text
        try:
            # Try to use a nicer font if available
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
            artist_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        except:
            # Fallback to default font
            title_font = ImageFont.load_default()
            artist_font = ImageFont.load_default()
        
        # Draw title
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        title_x = (size[0] - title_width) // 2
        title_y = size[1] // 2 - 100
        
        # Add text shadow
        draw.text((title_x + 2, title_y + 2), title, font=title_font, fill=(0, 0, 0, 128))
        draw.text((title_x, title_y), title, font=title_font, fill=(255, 255, 255))
        
        # Draw artist
        artist_bbox = draw.textbbox((0, 0), artist, font=artist_font)
        artist_width = artist_bbox[2] - artist_bbox[0]
        artist_x = (size[0] - artist_width) // 2
        artist_y = title_y + title_height + 30
        
        draw.text((artist_x + 1, artist_y + 1), artist, font=artist_font, fill=(0, 0, 0, 128))
        draw.text((artist_x, artist_y), artist, font=artist_font, fill=(255, 255, 255))
        
        return img
    
    def _add_geometric_shapes(self, draw, size, colors):
        """
        Add geometric shapes to the album art
        
        Args:
            draw: ImageDraw object
            size: Image size (width, height)
            colors: Color scheme
        """
        # Add some circles
        for _ in range(random.randint(3, 6)):
            x = random.randint(0, size[0])
            y = random.randint(0, size[1])
            radius = random.randint(30, 100)
            opacity = random.randint(20, 60)
            color = random.choice(colors)
            # Draw semi-transparent circles
            for i in range(radius, 0, -2):
                alpha = int(opacity * (1 - i / radius))
                draw.ellipse([x - i, y - i, x + i, y + i], 
                           outline=(*color, alpha), width=2)
    
    def image_to_buffer(self, image, format='PNG'):
        """
        Convert a PIL Image to a BytesIO buffer
        
        Args:
            image: PIL Image object
            format: Image format (PNG, JPEG, etc.)
            
        Returns:
            BytesIO buffer containing image data
        """
        buf = io.BytesIO()
        image.save(buf, format=format)
        buf.seek(0)
        return buf


def get_music_writers_table():
    """
    Get a table of respected music writers
    
    Returns:
        List of dictionaries containing music writer information
    """
    writers = [
        {"Name": "Ludwig van Beethoven", "Era": "Classical", "Notable Work": "Symphony No. 9"},
        {"Name": "Wolfgang Amadeus Mozart", "Era": "Classical", "Notable Work": "Requiem"},
        {"Name": "Johann Sebastian Bach", "Era": "Baroque", "Notable Work": "Brandenburg Concertos"},
        {"Name": "George Gershwin", "Era": "Jazz/Classical", "Notable Work": "Rhapsody in Blue"},
        {"Name": "Duke Ellington", "Era": "Jazz", "Notable Work": "Take the A Train"},
        {"Name": "The Beatles", "Era": "Rock/Pop", "Notable Work": "Sgt. Pepper's Lonely Hearts Club Band"},
        {"Name": "Bob Dylan", "Era": "Folk/Rock", "Notable Work": "Like a Rolling Stone"},
        {"Name": "Stevie Wonder", "Era": "Soul/R&B", "Notable Work": "Songs in the Key of Life"},
        {"Name": "Miles Davis", "Era": "Jazz", "Notable Work": "Kind of Blue"},
        {"Name": "Igor Stravinsky", "Era": "Modern Classical", "Notable Work": "The Rite of Spring"},
    ]
    return writers
