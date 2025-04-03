from src.utils.config import load_config

def main():
    try:
        config = load_config('config.yml')
        print("Configuration loaded and validated successfully!")
        
        # Print some key configuration details
        print("\nEnabled features:", config['features']['enabled_features'])
        print("\nEnabled tasks:")
        for task_name, task_config in config['tasks'].items():
            if task_config['enabled']:
                print(f"- {task_name} (weight: {task_config['weight']})")
        
        print("\nModel architecture:", config['model']['architecture'])
        print("Training device:", config['device'])
        
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")

if __name__ == "__main__":
    main() 