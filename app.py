"""
Music-Evolving AI Web App with Reward System
A Streamlit application for generating music with AI and gamification features.
"""

import io
import random
import uuid
from datetime import date, datetime
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import streamlit as st
from torch import nn
from midiutil import MIDIFile

from music_collection import (
    CollectionFormatError,
    collection_to_json,
    melody_to_wav_bytes,
    parse_collection,
)

# ============================================================================
# REWARD SYSTEM CLASSES AND FUNCTIONS
# ============================================================================

class Achievement:
    """Represents an achievement that can be unlocked."""

    def __init__(self, achievement_id: str, name: str, description: str,
                 icon: str, criteria_type: str, criteria_value: int):
        self.id = achievement_id
        self.name = name
        self.description = description
        self.icon = icon
        self.criteria_type = criteria_type
        self.criteria_value = criteria_value
        self.unlocked = False
        self.unlock_date = None


class SillyTitle:
    """Represents a silly title that users can unlock and display."""

    def __init__(self, title_id: str, title: str, unlock_condition: str):
        self.id = title_id
        self.title = title
        self.unlock_condition = unlock_condition
        self.unlocked = False


class VirtualCollectible:
    """Represents a virtual badge or sticker."""

    def __init__(self, collectible_id: str, name: str, emoji: str, rarity: str):
        self.id = collectible_id
        self.name = name
        self.emoji = emoji
        self.rarity = rarity  # common, rare, epic, legendary


class RewardSystem:
    """Manages the reward system including achievements, titles, and collectibles."""

    def __init__(self):
        self.achievements = self._initialize_achievements()
        self.titles = self._initialize_titles()
        self.collectibles = self._initialize_collectibles()
        self.fun_rewards = [
            "🎵 Music Joke: Why did the music note go to school? To get a little sharper!",
            "🎹 Trivia: The first computer-generated music was created in 1951!",
            "🎸 Fun Fact: The longest guitar solo ever recorded lasted 24 hours!",
            "🎼 Tip: Try experimenting with different keys to create unique moods!",
            "🎺 Quote: 'Music is the universal language of mankind.' - Henry Wadsworth Longfellow",
            "🥁 Did you know? The fastest drummer can hit 1,208 beats per minute!",
            "🎻 Challenge: Try creating a track in a genre you've never explored!",
            "🎤 Wisdom: Every great musician started as a beginner. Keep creating!"
        ]

    def _initialize_achievements(self) -> List[Achievement]:
        """Initialize all available achievements."""
        return [
            Achievement("first_track", "First Track Created",
                       "Create your very first music track", "🎵", "tracks_created", 1),
            Achievement("track_master", "Track Master",
                       "Create 10 music tracks", "🎼", "tracks_created", 10),
            Achievement("genre_explorer", "Genre Explorer",
                       "Try all 4 different genres", "🌍", "genres_explored", 4),
            Achievement("key_crusher", "Key Crusher",
                       "Experiment with 5 different keys", "🎹", "keys_used", 5),
            Achievement("mode_mixer", "Mode Mixer",
                       "Try all available modes", "🎚️", "modes_used", 3),
            Achievement("emotion_artist", "Emotion Artist",
                       "Express all emotions through music", "🎭", "emotions_used", 3),
            Achievement("daily_creator", "Daily Creator",
                       "Create music on consecutive days", "📅", "daily_streak", 3),
            Achievement("generation_guru", "Generation Guru",
                       "Use 100 generations in a single track", "⚡", "max_generations", 100),
        ]

    def _initialize_titles(self) -> List[SillyTitle]:
        """Initialize all available silly titles."""
        return [
            SillyTitle("lofi_legend", "Lofi Legend 🎧", "Create 5 tracks"),
            SillyTitle("melody_maestro", "Melody Maestro 🎼", "Create 10 tracks"),
            SillyTitle("key_commander", "Key Commander 🎹", "Use all 7 keys"),
            SillyTitle("genre_genius", "Genre Genius 🧠", "Try all genres"),
            SillyTitle("emotion_explorer", "Emotion Explorer 🎭", "Use all emotions"),
            SillyTitle("beat_baron", "Beat Baron 👑", "Create 20 tracks"),
            SillyTitle("harmony_hero", "Harmony Hero 🦸", "Create 50 tracks"),
            SillyTitle("rhythm_royalty", "Rhythm Royalty 👸", "Create 100 tracks"),
        ]

    def _initialize_collectibles(self) -> List[VirtualCollectible]:
        """Initialize all available virtual collectibles."""
        return [
            VirtualCollectible("note_sticker", "Musical Note", "🎵", "common"),
            VirtualCollectible("star_badge", "Rising Star", "⭐", "common"),
            VirtualCollectible("fire_badge", "On Fire", "🔥", "rare"),
            VirtualCollectible("trophy_badge", "Trophy", "🏆", "rare"),
            VirtualCollectible("crown_badge", "Crown Jewel", "👑", "epic"),
            VirtualCollectible("diamond_badge", "Diamond", "💎", "epic"),
            VirtualCollectible("unicorn_badge", "Unicorn", "🦄", "legendary"),
            VirtualCollectible("rocket_badge", "Rocket", "🚀", "legendary"),
        ]

    def get_random_fun_reward(self) -> str:
        """Return a random fun reward."""
        return random.choice(self.fun_rewards)

    def check_and_unlock_achievements(self, user_stats: Dict) -> List[Achievement]:
        """Check if any achievements should be unlocked based on user stats."""
        newly_unlocked = []

        for achievement in self.achievements:
            if not achievement.unlocked:
                if self._check_achievement_criteria(achievement, user_stats):
                    achievement.unlocked = True
                    achievement.unlock_date = datetime.now()
                    newly_unlocked.append(achievement)

        return newly_unlocked

    def _check_achievement_criteria(self, achievement: Achievement,
                                   user_stats: Dict) -> bool:
        """Check if achievement criteria is met."""
        stat_value = user_stats.get(achievement.criteria_type, 0)
        return stat_value >= achievement.criteria_value

    def check_and_unlock_titles(self, user_stats: Dict) -> List[SillyTitle]:
        """Check if any titles should be unlocked."""
        newly_unlocked = []

        for title in self.titles:
            if not title.unlocked:
                if self._check_title_criteria(title, user_stats):
                    title.unlocked = True
                    newly_unlocked.append(title)

        return newly_unlocked

    def _check_title_criteria(self, title: SillyTitle, user_stats: Dict) -> bool:
        """Check if title criteria is met."""
        tracks = user_stats.get("tracks_created", 0)
        keys = user_stats.get("keys_used", 0)
        genres = user_stats.get("genres_explored", 0)
        emotions = user_stats.get("emotions_used", 0)

        # Simple criteria mapping
        if "5 tracks" in title.unlock_condition:
            return tracks >= 5
        elif "10 tracks" in title.unlock_condition:
            return tracks >= 10
        elif "20 tracks" in title.unlock_condition:
            return tracks >= 20
        elif "50 tracks" in title.unlock_condition:
            return tracks >= 50
        elif "100 tracks" in title.unlock_condition:
            return tracks >= 100
        elif "all 7 keys" in title.unlock_condition:
            return keys >= 7
        elif "all genres" in title.unlock_condition:
            return genres >= 4
        elif "all emotions" in title.unlock_condition:
            return emotions >= 3

        return False

    def award_random_collectible(self) -> Optional[VirtualCollectible]:
        """Award a random collectible based on rarity."""
        # Weighted random selection based on rarity
        weights = {
            "common": 50,
            "rare": 30,
            "epic": 15,
            "legendary": 5
        }

        available = list(self.collectibles)
        rarities = [c.rarity for c in available]
        rarity_weights = [weights[r] for r in rarities]

        if available:
            return random.choices(available, weights=rarity_weights, k=1)[0]
        return None

    def calculate_level(self, total_actions: int) -> int:
        """Calculate user level based on total actions."""
        # Level up every 10 actions
        return min(total_actions // 10, 100)

    def get_progress_to_next_level(self, total_actions: int) -> Tuple[int, int]:
        """Get progress towards next level."""
        current_level = self.calculate_level(total_actions)
        actions_for_current = current_level * 10
        actions_for_next = (current_level + 1) * 10
        progress = total_actions - actions_for_current
        needed = actions_for_next - actions_for_current
        return progress, needed


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize session state for user stats and rewards."""
    if 'user_stats' not in st.session_state:
        st.session_state.user_stats = {
            'tracks_created': 0,
            'genres_explored': 0,
            'keys_used': 0,
            'modes_used': 0,
            'emotions_used': 0,
            'daily_streak': 0,
            'max_generations': 0,
            'total_actions': 0,
            'last_login': None,
            'genres_list': set(),
            'keys_list': set(),
            'modes_list': set(),
            'emotions_list': set(),
        }

    if 'reward_system' not in st.session_state:
        st.session_state.reward_system = RewardSystem()

    if 'unlocked_achievements' not in st.session_state:
        st.session_state.unlocked_achievements = []

    if 'unlocked_titles' not in st.session_state:
        st.session_state.unlocked_titles = []

    if 'collected_items' not in st.session_state:
        st.session_state.collected_items = []

    if 'selected_title' not in st.session_state:
        st.session_state.selected_title = None

    if 'rewards_enabled' not in st.session_state:
        st.session_state.rewards_enabled = True

    if 'show_notifications' not in st.session_state:
        st.session_state.show_notifications = []

    if 'track_collection' not in st.session_state:
        st.session_state.track_collection = []

    if 'wav_cache' not in st.session_state:
        st.session_state.wav_cache = {}


# ============================================================================
# MUSIC GENERATION FUNCTIONS
# ============================================================================

class MusicLSTM(nn.Module):
    """Simple LSTM model for music generation."""

    def __init__(self, input_size=128, hidden_size=256, output_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the network."""
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output


def generate_individual(melody_notes: List[int], length: int = 32) -> Dict:
    """Generate a random individual for genetic algorithm."""
    return {
        "melody": [(random.choice(melody_notes), 1, 100) for _ in range(length)]
    }


def evolve_music(melody_notes: List[int], generations: int,
                 population_size: int = 20) -> Dict:
    """Evolve music using a simple genetic algorithm."""
    population = [generate_individual(melody_notes) for _ in range(population_size)]

    for _ in range(generations):
        # Fitness: sum of note pitches (simple heuristic)
        population.sort(key=lambda x: sum(n for n, _, _ in x["melody"]), reverse=True)
        # Keep top 50% and generate new individuals for bottom 50%
        survivors = population[:population_size // 2]
        new_individuals = [generate_individual(melody_notes)
                          for _ in range(population_size - len(survivors))]
        population = survivors + new_individuals

    # Return the best individual
    return max(population, key=lambda x: sum(n for n, _, _ in x["melody"]))


def create_midi_from_melody(melody: List[Tuple[int, float, int]],
                            tempo: int = 120) -> io.BytesIO:
    """Create MIDI file from melody data."""
    midi = MIDIFile(1)
    track = 0
    time_offset = 0
    midi.addTempo(track, time_offset, tempo)

    time = 0
    for note, duration, volume in melody:
        midi.addNote(track, 0, note, time, duration, volume)
        time += duration

    midi_buffer = io.BytesIO()
    midi.writeFile(midi_buffer)
    midi_buffer.seek(0)
    return midi_buffer


# ============================================================================
# TRACK COLLECTION, AUDIO PLAYBACK, AND BACKUP/RESTORE
# ============================================================================

def get_track_wav(track: Dict) -> bytes:
    """Render (and cache) the WAV audio for a saved track."""
    cache_key = track["id"]
    if cache_key not in st.session_state.wav_cache:
        st.session_state.wav_cache[cache_key] = melody_to_wav_bytes(
            track["melody"], track.get("tempo", 120))
    return st.session_state.wav_cache[cache_key]


def save_track_to_collection(melody: List[Tuple[int, float, int]], genre: str,
                             key_root: str, mode: str, emotion: str,
                             generations: int, tempo: int) -> Dict:
    """Keep a generated track in the session collection and return it."""
    track = {
        "id": uuid.uuid4().hex,
        "name": f"{genre.title()} — {key_root} {mode} ({emotion})",
        "melody": [tuple(note) for note in melody],
        "genre": genre,
        "key_root": key_root,
        "mode": mode,
        "emotion": emotion,
        "generations": generations,
        "tempo": tempo,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    st.session_state.track_collection.append(track)
    return track


def export_collection() -> bytes:
    """Serialize the track collection and reward progress to JSON bytes."""
    reward_system = st.session_state.reward_system
    return collection_to_json(
        tracks=st.session_state.track_collection,
        user_stats=st.session_state.user_stats,
        achievements=[
            {"id": a.id, "unlocked": a.unlocked, "unlock_date": a.unlock_date}
            for a in reward_system.achievements
        ],
        titles=[
            {"id": t.id, "unlocked": t.unlocked} for t in reward_system.titles
        ],
        collectibles=[c.id for c in st.session_state.collected_items],
        selected_title=st.session_state.selected_title,
        rewards_enabled=st.session_state.rewards_enabled,
    )


def apply_collection(doc: Dict):
    """Replace the session state with a validated collection document.

    ``doc`` must come from ``music_collection.parse_collection``, which has
    already validated every field; unknown achievement/title/collectible
    ids (e.g. from a different app version) are ignored.
    """
    stats = doc["user_stats"]
    st.session_state.user_stats = {
        **stats,
        'genres_list': set(stats['genres_list']),
        'keys_list': set(stats['keys_list']),
        'modes_list': set(stats['modes_list']),
        'emotions_list': set(stats['emotions_list']),
        'last_login': (date.fromisoformat(stats['last_login'])
                       if stats['last_login'] else None),
    }

    reward_system = RewardSystem()
    for achievement in reward_system.achievements:
        saved = doc["achievements"].get(achievement.id)
        if saved and saved["unlocked"]:
            achievement.unlocked = True
            if saved["unlock_date"]:
                try:
                    achievement.unlock_date = datetime.fromisoformat(
                        saved["unlock_date"])
                except ValueError:
                    achievement.unlock_date = None
    for title in reward_system.titles:
        if doc["titles"].get(title.id):
            title.unlocked = True
    st.session_state.reward_system = reward_system
    st.session_state.unlocked_achievements = [
        a for a in reward_system.achievements if a.unlocked]
    st.session_state.unlocked_titles = [
        t for t in reward_system.titles if t.unlocked]

    collectibles_by_id = {c.id: c for c in reward_system.collectibles}
    st.session_state.collected_items = [
        collectibles_by_id[cid] for cid in doc["collectibles"]
        if cid in collectibles_by_id
    ]

    unlocked_title_names = {t.title for t in st.session_state.unlocked_titles}
    selected = doc["selected_title"]
    st.session_state.selected_title = (
        selected if selected in unlocked_title_names else None)
    st.session_state.rewards_enabled = doc["rewards_enabled"]

    st.session_state.track_collection = [
        {**track,
         "id": uuid.uuid4().hex,
         "melody": [tuple(note) for note in track["melody"]]}
        for track in doc["tracks"]
    ]
    st.session_state.wav_cache = {}
    st.session_state.show_notifications = []


def display_collection():
    """Show the saved tracks with in-browser playback and downloads."""
    st.subheader("🎵 My Collection")

    tracks = st.session_state.track_collection
    if not tracks:
        st.info("No tracks saved yet. Generate a track above and it will be "
                "kept here. You can back the whole collection up from "
                "⚙️ Settings → Data Management so it survives a closed tab.")
        return

    st.write(f"**{len(tracks)} track(s) saved.** They live in this browser "
             "session — download a backup from ⚙️ Settings → Data Management "
             "to keep them for good.")

    for track in reversed(tracks):
        with st.expander(f"🎧 {track['name']}"):
            st.caption(
                f"{track['genre']} · key of {track['key_root']} · "
                f"{track['mode']} · {track['emotion']} · "
                f"{track['generations']} generations · {track['tempo']} BPM · "
                f"created {track['created_at']}")
            wav_bytes = get_track_wav(track)
            st.audio(wav_bytes, format="audio/wav")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    "⬇️ MIDI",
                    data=create_midi_from_melody(track["melody"], track["tempo"]),
                    file_name=f"music_{track['id'][:8]}.mid",
                    mime="audio/midi",
                    key=f"midi_{track['id']}",
                )
            with col2:
                st.download_button(
                    "⬇️ WAV",
                    data=wav_bytes,
                    file_name=f"music_{track['id'][:8]}.wav",
                    mime="audio/wav",
                    key=f"wav_{track['id']}",
                )
            with col3:
                if st.button("🗑️ Remove", key=f"del_{track['id']}"):
                    st.session_state.track_collection = [
                        t for t in tracks if t["id"] != track["id"]]
                    st.session_state.wav_cache.pop(track["id"], None)
                    st.rerun()


def visualize_melody(melody: List[Tuple[int, float, int]]) -> plt.Figure:
    """Create a visualization of the melody."""
    fig, ax = plt.subplots(figsize=(12, 4))

    notes = [n for n, _, _ in melody]
    times = list(range(len(notes)))

    ax.plot(times, notes, marker='o', linestyle='-', linewidth=2, markersize=4)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Note Pitch')
    ax.set_title('Generated Melody Visualization')
    ax.grid(True, alpha=0.3)

    return fig


# ============================================================================
# USER STATS UPDATE FUNCTIONS
# ============================================================================

def update_user_stats(genre: str, key_root: str, mode: str,
                     emotion: str, generations: int):
    """Update user statistics based on actions."""
    stats = st.session_state.user_stats

    # Increment tracks created
    stats['tracks_created'] += 1
    stats['total_actions'] += 1

    # Track unique genres, keys, modes, emotions
    stats['genres_list'].add(genre)
    stats['genres_explored'] = len(stats['genres_list'])

    stats['keys_list'].add(key_root)
    stats['keys_used'] = len(stats['keys_list'])

    stats['modes_list'].add(mode)
    stats['modes_used'] = len(stats['modes_list'])

    stats['emotions_list'].add(emotion)
    stats['emotions_used'] = len(stats['emotions_list'])

    # Update max generations used
    if generations > stats['max_generations']:
        stats['max_generations'] = generations

    # Update daily streak (simplified - would need proper date tracking in production)
    current_date = datetime.now().date()
    if stats['last_login']:
        last_date = stats['last_login']
        if (current_date - last_date).days == 1:
            stats['daily_streak'] += 1
        elif (current_date - last_date).days > 1:
            stats['daily_streak'] = 1
    else:
        stats['daily_streak'] = 1

    stats['last_login'] = current_date


def check_for_rewards():
    """Check and award new rewards based on updated stats."""
    if not st.session_state.rewards_enabled:
        return

    reward_system = st.session_state.reward_system
    stats = st.session_state.user_stats

    # Check for new achievements
    new_achievements = reward_system.check_and_unlock_achievements(stats)
    for achievement in new_achievements:
        st.session_state.unlocked_achievements.append(achievement)
        st.session_state.show_notifications.append({
            'type': 'achievement',
            'data': achievement
        })

    # Check for new titles
    new_titles = reward_system.check_and_unlock_titles(stats)
    for title in new_titles:
        st.session_state.unlocked_titles.append(title)
        st.session_state.show_notifications.append({
            'type': 'title',
            'data': title
        })

    # Random chance for collectible (20% chance)
    if random.random() < 0.2:
        collectible = reward_system.award_random_collectible()
        if collectible:
            st.session_state.collected_items.append(collectible)
            st.session_state.show_notifications.append({
                'type': 'collectible',
                'data': collectible
            })

    # Random chance for fun reward (30% chance)
    if random.random() < 0.3:
        fun_reward = reward_system.get_random_fun_reward()
        st.session_state.show_notifications.append({
            'type': 'fun',
            'data': fun_reward
        })


def display_notifications():
    """Display reward notifications to the user."""
    if st.session_state.show_notifications:
        for notification in st.session_state.show_notifications:
            if notification['type'] == 'achievement':
                achievement = notification['data']
                st.success(f"🏆 Achievement Unlocked: **{achievement.name}** {achievement.icon}\n\n"
                          f"_{achievement.description}_")
            elif notification['type'] == 'title':
                title = notification['data']
                st.success(f"✨ New Title Unlocked: **{title.title}**")
            elif notification['type'] == 'collectible':
                collectible = notification['data']
                st.success(f"🎁 New Collectible: {collectible.emoji} **{collectible.name}** "
                          f"({collectible.rarity.upper()})")
            elif notification['type'] == 'fun':
                st.info(notification['data'])

        # Clear notifications after displaying
        st.session_state.show_notifications = []


# ============================================================================
# UI COMPONENTS
# ============================================================================

def display_profile_dashboard():
    """Display user profile and rewards dashboard."""
    st.header("🎮 Your Profile & Rewards")

    if not st.session_state.rewards_enabled:
        st.info("Rewards system is currently disabled. Enable it in Settings to see "
                "your achievements!")
        return

    stats = st.session_state.user_stats
    reward_system = st.session_state.reward_system

    # User level and progress
    level = reward_system.calculate_level(stats['total_actions'])
    progress, needed = reward_system.get_progress_to_next_level(stats['total_actions'])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Level", level)
    with col2:
        st.metric("Tracks Created", stats['tracks_created'])
    with col3:
        st.metric("Total Actions", stats['total_actions'])

    # Progress bar to next level
    st.subheader("Progress to Next Level")
    progress_percent = (progress / needed * 100) if needed > 0 else 100
    st.progress(progress / needed if needed > 0 else 1.0)
    st.caption(f"{progress}/{needed} actions ({progress_percent:.1f}%)")

    # Display selected title
    if st.session_state.selected_title:
        st.subheader("Current Title")
        st.write(f"### {st.session_state.selected_title}")

    # Statistics
    st.subheader("📊 Statistics")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Genres Explored:** {stats['genres_explored']}/4")
        st.write(f"**Keys Used:** {stats['keys_used']}/7")
        st.write(f"**Modes Tried:** {stats['modes_used']}/3")

    with col2:
        st.write(f"**Emotions Used:** {stats['emotions_used']}/3")
        st.write(f"**Daily Streak:** {stats['daily_streak']} days")
        st.write(f"**Max Generations:** {stats['max_generations']}")

    # Achievements
    st.subheader("🏆 Achievements")
    unlocked = [a for a in reward_system.achievements if a.unlocked]
    locked = [a for a in reward_system.achievements if not a.unlocked]

    st.write(f"**Unlocked: {len(unlocked)}/{len(reward_system.achievements)}**")

    if unlocked:
        for achievement in unlocked:
            st.success(f"{achievement.icon} **{achievement.name}** - {achievement.description}")

    if locked:
        with st.expander("Show Locked Achievements"):
            for achievement in locked:
                st.write(f"🔒 **{achievement.name}** - {achievement.description}")

    # Titles
    st.subheader("✨ Silly Titles")
    unlocked_titles = [t for t in reward_system.titles if t.unlocked]

    if unlocked_titles:
        st.write("**Your Unlocked Titles:**")
        title_options = [t.title for t in unlocked_titles]
        selected = st.selectbox("Select a title to display",
                               ["None"] + title_options,
                               index=0 if not st.session_state.selected_title
                               else (title_options.index(st.session_state.selected_title) + 1
                                     if st.session_state.selected_title in title_options else 0))

        if selected != "None":
            st.session_state.selected_title = selected
        else:
            st.session_state.selected_title = None

    # Collectibles
    st.subheader("🎁 Virtual Collectibles")
    if st.session_state.collected_items:
        st.write(f"**Collection: {len(st.session_state.collected_items)} items**")

        # Group by rarity
        collectibles_by_rarity = {}
        for item in st.session_state.collected_items:
            if item.rarity not in collectibles_by_rarity:
                collectibles_by_rarity[item.rarity] = []
            collectibles_by_rarity[item.rarity].append(item)

        for rarity in ["legendary", "epic", "rare", "common"]:
            if rarity in collectibles_by_rarity:
                items = collectibles_by_rarity[rarity]
                st.write(f"**{rarity.upper()}:**")
                cols = st.columns(min(len(items), 4))
                for idx, item in enumerate(items):
                    with cols[idx % 4]:
                        st.write(f"{item.emoji} {item.name}")
    else:
        st.info("No collectibles yet! Keep creating music to earn rewards!")


def display_settings():
    """Display settings for customizing the reward system."""
    st.header("⚙️ Settings")

    st.subheader("Reward System")

    rewards_enabled = st.checkbox(
        "Enable Rewards System",
        value=st.session_state.rewards_enabled,
        help="Toggle the reward system on/off for a minimalistic experience"
    )

    if rewards_enabled != st.session_state.rewards_enabled:
        st.session_state.rewards_enabled = rewards_enabled
        if rewards_enabled:
            st.success("Rewards system enabled! 🎉")
        else:
            st.info("Rewards system disabled. You can still create music without distractions.")

    st.subheader("Data Management")

    st.write("**Back up your collection.** Your tracks and reward progress "
             "live in this browser session. Download a backup file to keep "
             "them, and restore it here any time to pick up where you left "
             "off. Restoring replaces the current session's tracks and "
             "progress.")

    st.download_button(
        "⬇️ Download my collection",
        data=export_collection(),
        file_name="music_ai_collection.json",
        mime="application/json",
    )

    uploaded_collection = st.file_uploader(
        "Restore from a collection file", type=["json"],
        key="collection_upload")
    if uploaded_collection is not None:
        if st.button("♻️ Restore this collection"):
            try:
                parsed = parse_collection(uploaded_collection.getvalue())
            except CollectionFormatError as exc:
                st.error(f"That file could not be restored: {exc}")
            else:
                apply_collection(parsed)
                st.success("✅ Collection restored!")
                st.rerun()

    st.divider()

    # Two-step reset driven by a session-state flag: a button nested inside
    # another button's block can never fire, because on the rerun triggered
    # by clicking "Confirm" the outer button reads False again.
    if "confirm_reset" not in st.session_state:
        st.session_state.confirm_reset = False

    if not st.session_state.confirm_reset:
        if st.button("Reset All Progress", type="secondary"):
            st.session_state.confirm_reset = True
            st.rerun()
    else:
        st.warning("Are you sure? This will erase all your progress.")
        if st.button("⚠️ Confirm Reset"):
            # Reset all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function."""
    st.set_page_config(
        page_title="Music-Evolving AI with Rewards",
        page_icon="🎵",
        layout="wide"
    )

    initialize_session_state()

    st.title("🎵 Music-Evolving AI Web App")

    # Display user's selected title if available
    if st.session_state.selected_title and st.session_state.rewards_enabled:
        st.caption(f"Playing as: {st.session_state.selected_title}")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🎼 Create Music", "🎮 Profile & Rewards", "⚙️ Settings"])

    # Display notifications
    display_notifications()

    if page == "🎼 Create Music":
        display_music_creation_page()
    elif page == "🎮 Profile & Rewards":
        display_profile_dashboard()
    elif page == "⚙️ Settings":
        display_settings()


def display_music_creation_page():
    """Display the music creation interface."""
    st.header("Create Your Music")

    # User inputs
    col1, col2 = st.columns(2)

    with col1:
        genre = st.selectbox("Genre", ["pop", "jazz", "classical", "rock"])
        key_root = st.selectbox("Key Root", ["C", "D", "E", "F", "G", "A", "B"])
        mode = st.selectbox("Mode", ["major", "minor", "dorian"])

    with col2:
        emotion = st.selectbox("Emotion", ["happy", "sad", "suspenseful"])
        generations = st.slider("Generations", 10, 100, 20,
                               help="Higher generations may produce more evolved music")
        tempo = st.slider("Tempo (BPM)", 60, 180, 120)

    if st.button("🎹 Generate Music", type="primary"):
        with st.spinner("🎼 Evolving music..."):
            # Define musical parameters
            root_notes = {"C": 60, "D": 62, "E": 64, "F": 65, "G": 67, "A": 69, "B": 71}
            root = root_notes[key_root]

            scales = {
                "major": [0, 2, 4, 5, 7, 9, 11],
                "minor": [0, 2, 3, 5, 7, 8, 10],
                "dorian": [0, 2, 3, 5, 7, 9, 10]
            }

            melody_notes = [root + i + 12 for i in scales[mode]]

            # Evolve music
            best_individual = evolve_music(melody_notes, generations)

            # Update user stats
            update_user_stats(genre, key_root, mode, emotion, generations)

            # Check for rewards
            check_for_rewards()

            # Visualization
            st.subheader("📊 Melody Visualization")
            fig = visualize_melody(best_individual["melody"])
            st.pyplot(fig)
            plt.close()

            # Keep the track in the collection so it can be replayed,
            # downloaded, and backed up instead of vanishing on rerun.
            track = save_track_to_collection(
                best_individual["melody"], genre, key_root, mode, emotion,
                generations, tempo)

            # Play the actual generated notes right here in the browser
            st.subheader("🎵 Your Generated Music")
            wav_bytes = get_track_wav(track)
            st.audio(wav_bytes, format="audio/wav")

            # Display download buttons
            col_midi, col_wav = st.columns(2)
            with col_midi:
                st.download_button(
                    label="⬇️ Download MIDI",
                    data=create_midi_from_melody(track["melody"], tempo),
                    file_name=f"music_{genre}_{key_root}_{mode}.mid",
                    mime="audio/midi"
                )
            with col_wav:
                st.download_button(
                    label="⬇️ Download WAV",
                    data=wav_bytes,
                    file_name=f"music_{genre}_{key_root}_{mode}.wav",
                    mime="audio/wav"
                )

            st.success("✅ Music generated and saved to your collection!")

            # Display quick stats
            if st.session_state.rewards_enabled:
                stats = st.session_state.user_stats
                reward_system = st.session_state.reward_system
                level = reward_system.calculate_level(stats['total_actions'])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Your Level", level)
                with col2:
                    st.metric("Total Tracks", stats['tracks_created'])
                with col3:
                    achievements_unlocked = len([a for a in reward_system.achievements
                                                if a.unlocked])
                    st.metric("Achievements",
                             f"{achievements_unlocked}/{len(reward_system.achievements)}")

    st.divider()
    display_collection()


if __name__ == "__main__":
    main()
