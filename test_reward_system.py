#!/usr/bin/env python3
"""
Test script for the reward system functionality.
This script tests various aspects of the reward system without needing Streamlit to run.
"""

import sys
sys.path.insert(0, '.')

from app import (
    Achievement, SillyTitle, VirtualCollectible, RewardSystem,
    generate_individual, evolve_music, create_midi_from_melody
)


def test_achievement_system():
    """Test achievement unlocking logic."""
    print("Testing Achievement System...")

    reward_system = RewardSystem()

    # Test initial state
    assert all(not a.unlocked for a in reward_system.achievements), \
        "All achievements should start locked"

    # Test achievement unlocking with 1 track
    user_stats = {
        'tracks_created': 1,
        'genres_explored': 1,
        'keys_used': 1,
        'modes_used': 1,
        'emotions_used': 1,
        'daily_streak': 1,
        'max_generations': 50
    }

    unlocked = reward_system.check_and_unlock_achievements(user_stats)
    assert len(unlocked) == 1, "Should unlock 'First Track Created' achievement"
    assert unlocked[0].id == 'first_track', "Should unlock first_track achievement"

    # Test achievement unlocking with 10 tracks
    user_stats['tracks_created'] = 10
    unlocked = reward_system.check_and_unlock_achievements(user_stats)
    assert len(unlocked) == 1, "Should unlock 'Track Master' achievement"

    print("✓ Achievement system tests passed!")


def test_title_system():
    """Test title unlocking logic."""
    print("\nTesting Title System...")

    reward_system = RewardSystem()

    # Test initial state
    assert all(not t.unlocked for t in reward_system.titles), \
        "All titles should start locked"

    # Test title unlocking with 5 tracks
    user_stats = {
        'tracks_created': 5,
        'genres_explored': 2,
        'keys_used': 3,
        'modes_used': 2,
        'emotions_used': 2,
        'daily_streak': 1,
        'max_generations': 50
    }

    unlocked = reward_system.check_and_unlock_titles(user_stats)
    assert len(unlocked) >= 1, "Should unlock at least one title with 5 tracks"

    # Test with all genres
    user_stats['genres_explored'] = 4
    reward_system = RewardSystem()  # Reset
    unlocked = reward_system.check_and_unlock_titles(user_stats)
    unlocked_ids = [t.id for t in unlocked]
    assert 'genre_genius' in unlocked_ids, "Should unlock Genre Genius title"

    print("✓ Title system tests passed!")


def test_collectible_system():
    """Test collectible awarding logic."""
    print("\nTesting Collectible System...")

    reward_system = RewardSystem()

    # Test random collectible awarding
    collectible = reward_system.award_random_collectible()
    assert collectible is not None, "Should award a collectible"
    assert collectible.rarity in ['common', 'rare', 'epic', 'legendary'], \
        "Collectible should have valid rarity"

    # Test multiple awards to verify randomness
    rarities = []
    for _ in range(100):
        c = reward_system.award_random_collectible()
        rarities.append(c.rarity)

    # Common should be most frequent
    common_count = rarities.count('common')
    legendary_count = rarities.count('legendary')
    assert common_count > legendary_count, \
        "Common collectibles should be more frequent than legendary"

    print("✓ Collectible system tests passed!")


def test_level_system():
    """Test level progression system."""
    print("\nTesting Level System...")

    reward_system = RewardSystem()

    # Test level calculation
    assert reward_system.calculate_level(0) == 0, "0 actions should be level 0"
    assert reward_system.calculate_level(9) == 0, "9 actions should be level 0"
    assert reward_system.calculate_level(10) == 1, "10 actions should be level 1"
    assert reward_system.calculate_level(25) == 2, "25 actions should be level 2"
    assert reward_system.calculate_level(100) == 10, "100 actions should be level 10"

    # Test progress calculation
    progress, needed = reward_system.get_progress_to_next_level(15)
    assert progress == 5, "Progress should be 5 actions"
    assert needed == 10, "Needed should be 10 actions"

    print("✓ Level system tests passed!")


def test_fun_rewards():
    """Test fun reward generation."""
    print("\nTesting Fun Rewards...")

    reward_system = RewardSystem()

    # Test getting random fun rewards
    reward1 = reward_system.get_random_fun_reward()
    assert isinstance(reward1, str), "Fun reward should be a string"
    assert len(reward1) > 0, "Fun reward should not be empty"

    # Test that we get different rewards
    rewards = set()
    for _ in range(50):
        rewards.add(reward_system.get_random_fun_reward())

    assert len(rewards) > 1, "Should get different fun rewards"

    print("✓ Fun rewards tests passed!")


def test_music_generation():
    """Test basic music generation functionality."""
    print("\nTesting Music Generation...")

    # Test individual generation
    melody_notes = [60, 62, 64, 65, 67, 69, 71]  # C major scale
    individual = generate_individual(melody_notes, length=8)

    assert 'melody' in individual, "Individual should have melody"
    assert len(individual['melody']) == 8, "Melody should have correct length"
    assert all(len(note) == 3 for note in individual['melody']), \
        "Each note should have (pitch, duration, velocity)"

    # Test evolution
    best = evolve_music(melody_notes, generations=5, population_size=10)
    assert 'melody' in best, "Best individual should have melody"
    assert len(best['melody']) > 0, "Best melody should not be empty"

    # Test MIDI creation
    midi_buffer = create_midi_from_melody(best['melody'], tempo=120)
    assert midi_buffer.tell() == 0, "MIDI buffer should be at start"
    midi_buffer.seek(0, 2)  # Seek to end
    size = midi_buffer.tell()
    assert size > 0, "MIDI file should have content"

    print("✓ Music generation tests passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("REWARD SYSTEM TEST SUITE")
    print("=" * 60)

    try:
        test_achievement_system()
        test_title_system()
        test_collectible_system()
        test_level_system()
        test_fun_rewards()
        test_music_generation()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
