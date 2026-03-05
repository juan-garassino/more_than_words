import json
import argparse
from tqdm import tqdm
from src.tokenizer import SymbolicTokenizer
from src.simulator import TamagotchiSimulator

def generate_jsonl(output_path, num_stories=1000, story_length=128):
    print(f"Initializing tokenizer and simulator...")
    tokenizer = SymbolicTokenizer()
    simulator = TamagotchiSimulator(tokenizer)
    
    print(f"Generating {num_stories} stories of length {story_length}...")
    
    with open(output_path, 'w') as f:
        for i in tqdm(range(num_stories)):
            # Generate a trajectory
            trajectory = simulator.generate_long_story(length=story_length)
            
            # Convert to serializable format
            # Each step in trajectory is (action_token, world_tokens_list, emotion_vector)
            
            story_steps = []
            for action, world_tokens, emotions in trajectory:
                step = {
                    "action": action.symbol,
                    "world": [t.symbol for t in world_tokens],
                    "emotions": {
                        dim: val for dim, val in zip(tokenizer.emotion_dims, emotions)
                    },
                    "lambda": simulator.get_lambda() # Capture lambda at generation time (approx)
                }
                story_steps.append(step)
            
            # Write full story object to line
            json_record = {
                "story_id": i,
                "steps": story_steps
            }
            
            f.write(json.dumps(json_record) + "\n")
            
    print(f"Successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="ML_Training/data/tamagotchi_raw.jsonl", help="Output JSONL path")
    parser.add_argument("--stories", type=int, default=100, help="Number of stories")
    parser.add_argument("--length", type=int, default=200, help="Steps per story")
    
    args = parser.parse_args()
    
    generate_jsonl(args.output, args.stories, args.length)
