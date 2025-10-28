#!/usr/bin/env python3
"""
Manual verification script for the reward system.
This simulates the reward system workflow without needing a full Streamlit UI.
"""

import sys
# Add current directory to path to allow importing app module directly
# This is acceptable for standalone verification scripts
sys.path.insert(0, '.')

from app import RewardSystem


def simulate_user_journey():
    """Simulate a user's journey through the app."""
    print("=" * 60)
    print("REWARD SYSTEM - USER JOURNEY SIMULATION")
    print("=" * 60)

    # Initialize the reward system
    reward_system = RewardSystem()

    # Initialize user stats
    user_stats = {
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

    print("\n📊 Initial State:")
    print(f"  Level: {reward_system.calculate_level(user_stats['total_actions'])}")
    print(f"  Tracks Created: {user_stats['tracks_created']}")
    print(f"  Achievements Unlocked: 0/{len(reward_system.achievements)}")

    # Simulate creating first track
    print("\n🎵 User creates their first track (Genre: pop, Key: C, Mode: major)...")
    user_stats['tracks_created'] = 1
    user_stats['total_actions'] = 1
    user_stats['genres_list'].add('pop')
    user_stats['genres_explored'] = len(user_stats['genres_list'])
    user_stats['keys_list'].add('C')
    user_stats['keys_used'] = len(user_stats['keys_list'])
    user_stats['modes_list'].add('major')
    user_stats['modes_used'] = len(user_stats['modes_list'])
    user_stats['emotions_list'].add('happy')
    user_stats['emotions_used'] = len(user_stats['emotions_list'])
    user_stats['max_generations'] = 20

    new_achievements = reward_system.check_and_unlock_achievements(user_stats)
    if new_achievements:
        for achievement in new_achievements:
            print(f"  🏆 Achievement Unlocked: {achievement.name} {achievement.icon}")
            print(f"     {achievement.description}")

    new_titles = reward_system.check_and_unlock_titles(user_stats)
    if new_titles:
        for title in new_titles:
            print(f"  ✨ New Title Unlocked: {title.title}")

    # Award a collectible for demo purposes
    collectible = reward_system.award_random_collectible()
    print(f"  🎁 Collectible Awarded: {collectible.emoji} {collectible.name} "
          f"({collectible.rarity.upper()})")

    # Get a fun reward
    fun_reward = reward_system.get_random_fun_reward()
    print(f"  💡 {fun_reward}")

    # Simulate creating more tracks
    print("\n🎵 User creates 4 more tracks with different genres and keys...")
    user_stats['tracks_created'] = 5
    user_stats['total_actions'] = 5
    user_stats['genres_list'].update(['jazz', 'classical'])
    user_stats['genres_explored'] = len(user_stats['genres_list'])
    user_stats['keys_list'].update(['D', 'E', 'F'])
    user_stats['keys_used'] = len(user_stats['keys_list'])

    new_achievements = reward_system.check_and_unlock_achievements(user_stats)
    if new_achievements:
        for achievement in new_achievements:
            print(f"  🏆 Achievement Unlocked: {achievement.name} {achievement.icon}")

    new_titles = reward_system.check_and_unlock_titles(user_stats)
    if new_titles:
        for title in new_titles:
            print(f"  ✨ New Title Unlocked: {title.title}")

    level = reward_system.calculate_level(user_stats['total_actions'])
    print("\n📈 Progress Update:")
    print(f"  Level: {level}")
    print(f"  Tracks Created: {user_stats['tracks_created']}")
    print(f"  Genres Explored: {user_stats['genres_explored']}/4")
    print(f"  Keys Used: {user_stats['keys_used']}/7")

    # Simulate creating 5 more tracks to reach 10 total
    print("\n🎵 User creates 5 more tracks...")
    user_stats['tracks_created'] = 10
    user_stats['total_actions'] = 10
    user_stats['genres_list'].add('rock')
    user_stats['genres_explored'] = len(user_stats['genres_list'])
    user_stats['keys_list'].update(['G', 'A'])
    user_stats['keys_used'] = len(user_stats['keys_list'])

    new_achievements = reward_system.check_and_unlock_achievements(user_stats)
    if new_achievements:
        for achievement in new_achievements:
            print(f"  🏆 Achievement Unlocked: {achievement.name} {achievement.icon}")

    new_titles = reward_system.check_and_unlock_titles(user_stats)
    if new_titles:
        for title in new_titles:
            print(f"  ✨ New Title Unlocked: {title.title}")

    level = reward_system.calculate_level(user_stats['total_actions'])
    progress, needed = reward_system.get_progress_to_next_level(user_stats['total_actions'])

    print("\n📊 Final State:")
    print(f"  Level: {level} (Progress: {progress}/{needed} to next level)")
    print(f"  Tracks Created: {user_stats['tracks_created']}")
    print(f"  Genres Explored: {user_stats['genres_explored']}/4")
    print(f"  Keys Used: {user_stats['keys_used']}/7")

    unlocked_count = len([a for a in reward_system.achievements if a.unlocked])
    print(f"  Achievements Unlocked: {unlocked_count}/{len(reward_system.achievements)}")

    unlocked_titles = [t for t in reward_system.titles if t.unlocked]
    print(f"  Titles Unlocked: {len(unlocked_titles)}/{len(reward_system.titles)}")

    if unlocked_titles:
        print("\n  Available Titles:")
        for title in unlocked_titles:
            print(f"    • {title.title}")

    print("\n" + "=" * 60)
    print("✅ User journey simulation complete!")
    print("=" * 60)


def display_all_rewards():
    """Display all available rewards."""
    print("\n" + "=" * 60)
    print("ALL AVAILABLE REWARDS")
    print("=" * 60)

    reward_system = RewardSystem()

    print("\n🏆 ACHIEVEMENTS:")
    for achievement in reward_system.achievements:
        print(f"  {achievement.icon} {achievement.name}")
        print(f"     {achievement.description}")

    print("\n✨ SILLY TITLES:")
    for title in reward_system.titles:
        print(f"  {title.title}")
        print(f"     Unlock: {title.unlock_condition}")

    print("\n🎁 VIRTUAL COLLECTIBLES:")
    by_rarity = {}
    for collectible in reward_system.collectibles:
        if collectible.rarity not in by_rarity:
            by_rarity[collectible.rarity] = []
        by_rarity[collectible.rarity].append(collectible)

    for rarity in ['legendary', 'epic', 'rare', 'common']:
        if rarity in by_rarity:
            print(f"\n  {rarity.upper()}:")
            for collectible in by_rarity[rarity]:
                print(f"    {collectible.emoji} {collectible.name}")

    print("\n💡 FUN REWARDS (samples):")
    for i, reward in enumerate(reward_system.fun_rewards[:3], 1):
        print(f"  {i}. {reward}")
    print(f"  ... and {len(reward_system.fun_rewards) - 3} more!")


if __name__ == "__main__":
    simulate_user_journey()
    display_all_rewards()
