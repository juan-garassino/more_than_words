from typing import List, Tuple, Dict, Optional
import numpy as np
import random
from .tokenizer import Token, SymbolicTokenizer

class TamagotchiSimulator:
    """
    Simulates a Tamagotchi-style environment to generate training data.
    Creates long trajectories that can be chunked for training.
    """

    def __init__(self, tokenizer: SymbolicTokenizer):
        self.tokenizer = tokenizer
        self.reset()

    def reset(self):
        """Initialize game state"""
        self.state = {
            'hunger': 0.5,      # 0=full, 1=starving
            'dirt': 0.3,        # 0=clean, 1=filthy
            'tiredness': 0.4,   # 0=energetic, 1=exhausted
            'happiness': 0.6,   # 0=sad, 1=joyful
            'sickness': 0.0,    # 0=healthy, 1=very sick
            'location': 0,      # 0-3 room index
            'food_present': True,
            'timestep': 0
        }

        # Emotion vector
        self.emotions = {
            'happiness': 0.6,
            'energy': 0.6,
            'stress': 0.3,
            'hunger': 0.5,
            'loneliness': 0.4
        }

    def get_lambda(self) -> float:
        """Compute λ based on emotional state"""
        # High stress/hunger increases λ (more ambiguity)
        # High happiness/energy decreases λ (more clarity)
        lambda_emotion = (
            0.4 * self.emotions['stress'] +
            0.3 * self.emotions['hunger'] +
            0.2 * self.emotions['loneliness'] -
            0.3 * self.emotions['happiness'] -
            0.2 * self.emotions['energy']
        )

        # Clamp to [0.2, 0.9]
        return np.clip(0.5 + lambda_emotion, 0.2, 0.9)

    def step(self, action_token: Token) -> Tuple[Token, List[Token], List[float]]:
        """
        Execute action and return (action, world_tokens, emotion_vector)
        """
        action = action_token.symbol

        # Update state based on action
        if action == 'FEED':
            if self.state['food_present']:
                self.state['hunger'] = max(0, self.state['hunger'] - 0.3)
                self.state['happiness'] += 0.1
                self.state['food_present'] = False
                self.emotions['hunger'] = self.state['hunger']

        elif action == 'CLEAN':
            self.state['dirt'] = max(0, self.state['dirt'] - 0.4)
            self.state['happiness'] += 0.05

        elif action == 'PLAY':
            self.state['happiness'] = min(1, self.state['happiness'] + 0.2)
            self.state['tiredness'] += 0.15
            self.state['hunger'] += 0.1
            self.emotions['loneliness'] = max(0, self.emotions['loneliness'] - 0.2)

        elif action == 'SLEEP_CMD':
            self.state['tiredness'] = max(0, self.state['tiredness'] - 0.5)
            self.emotions['energy'] = 1.0 - self.state['tiredness']
            self.emotions['stress'] = max(0, self.emotions['stress'] - 0.3)

        elif action.startswith('MOVE_'):
            direction_map = {'MOVE_N': 1, 'MOVE_S': -1, 'MOVE_E': 1, 'MOVE_W': -1}
            delta = direction_map.get(action, 0)
            self.state['location'] = (self.state['location'] + delta) % 4
            self.state['tiredness'] += 0.05

        # Natural decay
        self.state['hunger'] = min(1, self.state['hunger'] + 0.02)
        self.state['dirt'] = min(1, self.state['dirt'] + 0.01)
        self.state['tiredness'] = min(1, self.state['tiredness'] + 0.03)

        # Sickness logic
        if self.state['hunger'] > 0.8 or self.state['dirt'] > 0.7:
            self.state['sickness'] = min(1, self.state['sickness'] + 0.05)
        else:
            self.state['sickness'] = max(0, self.state['sickness'] - 0.02)

        # Update emotions
        self.emotions['happiness'] = self.state['happiness']
        self.emotions['energy'] = 1.0 - self.state['tiredness']
        self.emotions['stress'] = (self.state['hunger'] + self.state['dirt']) / 2
        self.emotions['hunger'] = self.state['hunger']

        self.state['timestep'] += 1

        # Generate world tokens based on λ
        world_tokens = self._generate_world_tokens()

        # Suggest next action
        next_action = self._suggest_action()

        # Emotion vector
        emotion_vec = [self.emotions[dim] for dim in self.tokenizer.emotion_dims]

        return next_action, world_tokens, emotion_vec

    def _generate_world_tokens(self) -> List[Token]:
        """Generate world tokens, filtered by λ"""
        lambda_val = self.get_lambda()

        # All possible world tokens
        all_world_tokens = []

        # Hunger
        if self.state['hunger'] < 0.3:
            all_world_tokens.append(Token('HUNGRY_L', 'WORLD'))
        elif self.state['hunger'] < 0.7:
            all_world_tokens.append(Token('HUNGRY_M', 'WORLD'))
        else:
            all_world_tokens.append(Token('HUNGRY_H', 'WORLD'))

        # Dirt
        if self.state['dirt'] < 0.3:
            all_world_tokens.append(Token('DIRTY_L', 'WORLD'))
        elif self.state['dirt'] < 0.7:
            all_world_tokens.append(Token('DIRTY_M', 'WORLD'))
        else:
            all_world_tokens.append(Token('DIRTY_H', 'WORLD'))

        # Tiredness
        if self.state['tiredness'] < 0.3:
            all_world_tokens.append(Token('TIRED_L', 'WORLD'))
        elif self.state['tiredness'] < 0.7:
            all_world_tokens.append(Token('TIRED_M', 'WORLD'))
        else:
            all_world_tokens.append(Token('TIRED_H', 'WORLD'))

        # Happiness
        if self.state['happiness'] < 0.3:
            all_world_tokens.append(Token('HAPPY_L', 'WORLD'))
        elif self.state['happiness'] < 0.7:
            all_world_tokens.append(Token('HAPPY_M', 'WORLD'))
        else:
            all_world_tokens.append(Token('HAPPY_H', 'WORLD'))

        # Sickness
        if self.state['sickness'] > 0.5:
            all_world_tokens.append(Token('SICK', 'WORLD'))
        else:
            all_world_tokens.append(Token('HEALTHY', 'WORLD'))

        # Food
        if self.state['food_present']:
            all_world_tokens.append(Token('FOOD_PRESENT', 'WORLD'))
        else:
            all_world_tokens.append(Token('FOOD_ABSENT', 'WORLD'))

        # Location
        all_world_tokens.append(Token(f'LOC_{self.state["location"]}', 'WORLD'))

        # Apply λ filtering - keep top k tokens
        k_max = len(all_world_tokens)
        k_actual = max(2, int(k_max * (1 - lambda_val)))

        # Priority order: critical stats first
        priority_order = [
            'HUNGRY_H', 'SICK', 'TIRED_H', 'DIRTY_H',  # Critical
            'FOOD_PRESENT', 'FOOD_ABSENT',              # Important
            'HUNGRY_M', 'TIRED_M', 'DIRTY_M',          # Medium
            'HAPPY_H', 'HAPPY_M', 'HAPPY_L',           # Lower priority
            'HUNGRY_L', 'TIRED_L', 'DIRTY_L', 'HEALTHY' # Least critical
        ]

        # Sort tokens by priority
        def token_priority(token):
            try:
                return priority_order.index(token.symbol)
            except ValueError:
                return len(priority_order)

        sorted_tokens = sorted(all_world_tokens, key=token_priority)

        return sorted_tokens[:k_actual]

    def _suggest_action(self) -> Token:
        """AI suggests next action based on state"""
        # Simple rule-based AI
        if self.state['hunger'] > 0.7 and self.state['food_present']:
            return Token('FEED', 'ACTION')
        elif self.state['tiredness'] > 0.7:
            return Token('SLEEP_CMD', 'ACTION')
        elif self.state['dirt'] > 0.7:
            return Token('CLEAN', 'ACTION')
        elif self.state['happiness'] < 0.3:
            return Token('PLAY', 'ACTION')
        elif self.state['hunger'] > 0.5 and not self.state['food_present']:
            # Go to kitchen
            if self.state['location'] != 0:
                return Token('MOVE_N', 'ACTION')
            else:
                self.state['food_present'] = True  # Refill food
                return Token('INSPECT', 'ACTION')
        else:
            return random.choice([
                Token('WAIT', 'ACTION'),
                Token('INSPECT', 'ACTION'),
                Token('PLAY', 'ACTION')
            ])

    def generate_long_story(self, length: int = 500) -> List[Tuple[Token, List[Token], List[float]]]:
        """Generate a long trajectory"""
        self.reset()
        trajectory = []

        for _ in range(length):
            # Get suggested action
            action = self._suggest_action()

            # Add some randomness
            if random.random() < 0.2:
                action = Token(random.choice(self.tokenizer.action_vocab), 'ACTION')

            # Execute step
            next_action, world_tokens, emotions = self.step(action)

            trajectory.append((action, world_tokens, emotions))

        return trajectory
