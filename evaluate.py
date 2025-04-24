import json
import os

def evaluate_okvqa(file_path):
    """
    Evaluate the accuracy of a model on OKVQA dataset.
    - Convert model_answer to lowercase
    - Check if model_answer is in gold_answers
    - Calculate accuracy
    """
    correct = 0
    total = 0
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    model_answer = data['model_answer'].lower()
                    gold_answers = [ans.lower() for ans in data['gold_answers']]
                    
                    if model_answer in gold_answers:
                        correct += 1
                    
                    total += 1
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Error processing line in {file_path}: {e}")
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total

def evaluate_clevr(file_path):
    """
    Evaluate the accuracy of a model on CLEVR dataset.
    - Check if gold_answer exactly matches model_answer
    - Calculate accuracy
    """
    correct = 0
    total = 0
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    model_answer = data['model_answer']
                    gold_answer = data['gold_answer']
                    
                    if model_answer.lower() == gold_answer.lower():
                        correct += 1
                    
                    total += 1
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Error processing line in {file_path}: {e}")
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total

def main():
    models = ['gemma', 'janus', 'qwen']
    
    # Define the base directory for the vqa_outputs folder
    vqa_outputs_dir = "/home/mkapadni/work/speech_project/vqa_outputs"  # Adjust if needed
    
    results = {}
    
    print("Evaluating model accuracy...")
    
    for model in models:
        results[model] = {}
        
        # Evaluate CLEVR
        clevr_file = os.path.join(vqa_outputs_dir, f"clevr_results_{model}.jsonl")
        clevr_accuracy, clevr_correct, clevr_total = evaluate_clevr(clevr_file)
        results[model]['clevr'] = {
            'accuracy': clevr_accuracy,
            'correct': clevr_correct,
            'total': clevr_total
        }
        
        # Evaluate OKVQA
        okvqa_file = os.path.join(vqa_outputs_dir, f"okvqa_results_{model}.jsonl")
        okvqa_accuracy, okvqa_correct, okvqa_total = evaluate_okvqa(okvqa_file)
        results[model]['okvqa'] = {
            'accuracy': okvqa_accuracy,
            'correct': okvqa_correct,
            'total': okvqa_total
        }
    
    # Print results
    print("\nAccuracy Results:")
    print("=" * 60)
    print(f"{'Model':<10} | {'CLEVR':<25} | {'OKVQA':<25}")
    print("-" * 60)
    
    for model in models:
        clevr_acc = results[model]['clevr']['accuracy'] * 100
        okvqa_acc = results[model]['okvqa']['accuracy'] * 100
        print(f"{model:<10} | {clevr_acc:.2f}% ({results[model]['clevr']['correct']}/{results[model]['clevr']['total']}) | {okvqa_acc:.2f}% ({results[model]['okvqa']['correct']}/{results[model]['okvqa']['total']})")
    
    print("=" * 60)

if __name__ == "__main__":
    main()