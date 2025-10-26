"""
Album art generation service
Creates simple geometric album art using PIL
"""
from PIL import Image, ImageDraw, ImageFont
import random
import os
import re
from typing import Tuple


class AlbumArtGenerator:
    """Generate album art using geometric patterns"""
    
    # Color schemes based on genre/emotion
    COLOR_SCHEMES = {
        'happy': [
            ('#FFD700', '#FF69B4', '#87CEEB'),  # Gold, Pink, Sky Blue
            ('#FFA500', '#FFFF00', '#00FF00'),  # Orange, Yellow, Green
            ('#FF6B6B', '#4ECDC4', '#FFE66D'),  # Coral, Teal, Yellow
        ],
        'sad': [
            ('#2C3E50', '#34495E', '#7F8C8D'),  # Dark blues and grays
            ('#1A1A2E', '#16213E', '#0F3460'),  # Deep blues
            ('#4A5568', '#718096', '#A0AEC0'),  # Slate grays
        ],
        'energetic': [
            ('#FF0000', '#FF4500', '#FF6347'),  # Reds
            ('#FF1493', '#FF69B4', '#FFB6C1'),  # Hot pinks
            ('#FF8C00', '#FFA500', '#FFD700'),  # Orange-golds
        ],
        'calm': [
            ('#E0F2F1', '#B2DFDB', '#80CBC4'),  # Mint greens
            ('#E1F5FE', '#B3E5FC', '#81D4FA'),  # Light blues
            ('#F3E5F5', '#E1BEE7', '#CE93D8'),  # Lavenders
        ],
    }
    
    def __init__(self, emotion: str = 'happy', genre: str = 'pop'):
        """Initialize album art generator"""
        self.emotion = emotion.lower()
        self.genre = genre.lower()
        self.colors = random.choice(self.COLOR_SCHEMES.get(self.emotion, self.COLOR_SCHEMES['happy']))
    
    def generate_gradient_background(self, size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """Generate gradient background"""
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)
        
        # Create gradient from top to bottom
        color1 = self._hex_to_rgb(self.colors[0])
        color2 = self._hex_to_rgb(self.colors[1])
        
        for y in range(size[1]):
            ratio = y / size[1]
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            draw.line([(0, y), (size[0], y)], fill=(r, g, b))
        
        return img
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def add_geometric_shapes(self, img: Image.Image) -> Image.Image:
        """Add geometric shapes to the image"""
        draw = ImageDraw.Draw(img, 'RGBA')
        width, height = img.size
        
        # Add random circles
        num_circles = random.randint(3, 7)
        for _ in range(num_circles):
            x = random.randint(0, width)
            y = random.randint(0, height)
            radius = random.randint(30, 150)
            color = self._hex_to_rgb(random.choice(self.colors))
            alpha = random.randint(50, 150)
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=color + (alpha,),
                outline=None
            )
        
        # Add rectangles
        num_rects = random.randint(2, 5)
        for _ in range(num_rects):
            x1 = random.randint(0, width - 100)
            y1 = random.randint(0, height - 100)
            x2 = x1 + random.randint(50, 200)
            y2 = y1 + random.randint(50, 200)
            color = self._hex_to_rgb(random.choice(self.colors))
            alpha = random.randint(50, 150)
            draw.rectangle([x1, y1, x2, y2], fill=color + (alpha,))
        
        return img
    
    def add_title(self, img: Image.Image, title: str) -> Image.Image:
        """Add title text to the image"""
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Try to use a nice font, fall back to default if not available
        try:
            font_size = 48
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Get text size for centering
        # Use textbbox for newer PIL versions
        try:
            bbox = draw.textbbox((0, 0), title, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(title, font=font)
        
        # Draw text with outline
        x = (width - text_width) // 2
        y = height - 100
        
        # Draw outline
        outline_color = (0, 0, 0)
        for adj_x in range(-2, 3):
            for adj_y in range(-2, 3):
                draw.text((x + adj_x, y + adj_y), title, font=font, fill=outline_color)
        
        # Draw main text
        draw.text((x, y), title, font=font, fill=(255, 255, 255))
        
        return img
    
    def generate(self, title: str, output_path: str) -> str:
        """Generate complete album art"""
        # Sanitize the output path to prevent path injection
        output_path = os.path.normpath(output_path)
        base_dir = os.path.normpath(os.path.join(os.getcwd(), 'static', 'images'))
        
        # Ensure the path is within the allowed directory
        if not output_path.startswith(base_dir):
            output_path = os.path.join(base_dir, os.path.basename(output_path))
        
        # Create base gradient
        img = self.generate_gradient_background()
        
        # Add geometric shapes
        img = self.add_geometric_shapes(img)
        
        # Add title
        if title:
            img = self.add_title(img, title)
        
        # Save image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, 'PNG')
        
        return output_path
