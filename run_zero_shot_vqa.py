import os
import json
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoProcessor,
    BitsAndBytesConfig, 
    Gemma3ForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

# Set Hugging Face cache directory
os.environ["TRANSFORMERS_CACHE"] = "/data/user_data/mkapadni/hf_cache/models"

# Define process_vision_info if qwen_vl_utils not available
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    def process_vision_info(messages):
        image_inputs = []
        video_inputs = []
        for message in messages:
            if "content" in message:
                for content in message["content"]:
                    if content["type"] == "image":
                        if isinstance(content["image"], str):
                            image = Image.open(content["image"]).convert("RGB")
                        else:
                            image = content["image"]
                        image_inputs.append(image)
        return image_inputs, video_inputs

def load_janus_model():
    """Load the Janus Pro 1B model"""
    print("Loading Janus Pro 1B model...")
    janus_model_path = "deepseek-ai/Janus-Pro-1B"  # Changed to 1B model
    janus_processor = VLChatProcessor.from_pretrained(janus_model_path)
    janus_tokenizer = janus_processor.tokenizer
    
    janus_model = AutoModelForCausalLM.from_pretrained(
        janus_model_path, trust_remote_code=True
    ).to(torch.bfloat16).cuda().eval()
    
    return {
        "model": janus_model,
        "processor": janus_processor,
        "tokenizer": janus_tokenizer
    }

def load_gemma_model():
    """Load the Gemma 3-4b-it model (multimodal)"""
    print("Loading Gemma 3-4b-it model...")
    model_id = "google/gemma-3-4b-it"
    
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", token = "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"
    ).eval()
    
    processor = AutoProcessor.from_pretrained(model_id,token = "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")
    
    return {
        "model": model,
        "processor": processor
    }

def load_qwen_model():
    """Load the Qwen2.5-VL model"""
    print("Loading Qwen2.5-VL model...")
    qwen_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_model_id, torch_dtype="auto", device_map="auto"
    )
    
    qwen_processor = AutoProcessor.from_pretrained(qwen_model_id)
    
    return {
        "model": qwen_model,
        "processor": qwen_processor
    }

def load_okvqa_dataset():
    """Load the OK-VQA test dataset"""
    print("Loading OK-VQA dataset...")
    dataset = load_dataset("Multimodal-Fatima/OK-VQA_test")
    return dataset["test"]

def load_clevr_dataset(json_path, images_dir):
    """Load the CLEVR dataset from a JSON file and images directory"""
    print(f"Loading CLEVR dataset from {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    examples = []
    for q in data.get("questions", []):
        image_path = os.path.join(images_dir, q["image_filename"])
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping...")
            continue
            
        examples.append({
            "image_path": image_path,
            "question": q["question"],
            "answer": q["answer"],
            "image_filename": q["image_filename"]
        })
    
    print(f"Loaded {len(examples)} valid examples from CLEVR dataset")
    return examples

def get_janus_answer(model_data, conversation):
    """Get answer from Janus Pro model with limited token generation"""
    vl_gpt = model_data["model"]
    vl_chat_processor = model_data["processor"]
    tokenizer = model_data["tokenizer"]
    
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # Generate only 1-2 tokens as per request
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=2,  # Limiting to 1-2 tokens
        do_sample=False,
        use_cache=True,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer



def get_gemma_answer(model_data, image, question):
    """Get answer from Gemma model with limited token generation"""
    model = model_data["model"]
    processor = model_data["processor"]
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a visual QA system. Answer in one word or phrase only."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    
    input_len = inputs["input_ids"].shape[-1]
    
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=2, do_sample=False)
        generation = generation[0][input_len:]
    
    answer = processor.decode(generation, skip_special_tokens=True)
    return answer

def get_qwen_answer(model_data, image, question):
    """Get answer from Qwen2.5-VL model with limited token generation"""
    model = model_data["model"]
    processor = model_data["processor"]
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate only 1-2 tokens as per request
    generated_ids = model.generate(**inputs, max_new_tokens=2)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return answer

def process_okvqa(model_name, model_data, output_dir):
    """Process OK-VQA dataset with a specific model"""
    dataset = load_okvqa_dataset()
    
    # Create results file
    results_file = os.path.join(output_dir, f"okvqa_results_{model_name}.jsonl")
    # Initialize the file
    with open(results_file, "w") as f:
        f.write("")
    
    # Model-specific prompts for OK-VQA
    prompts = {
    "janus": "This image is provided to help answer a specific question. Please provide a concise answer (one word or short phrase) to the question based on the context and content of the image.",
    "gemma": "Given this image for context, answer the following question with a single word or brief phrase.",
    "qwen": "As a visual QA system, use the context provided by this image to respond with a one-word or very short phrase answer to the question."
}

    prompt = prompts.get(model_name, "Answer with a single word or phrase:")
    
    print(f"Processing OK-VQA dataset with {model_name} model ({len(dataset)} questions)...")
    
    for i, item in enumerate(tqdm(dataset)):
        pil_image = item["image"]  # PIL image object
        question = item["question"]
        gold_answers = item["answers"] if "answers" in item else item.get("answers_original", [])
        
        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]
        
        full_question = f"{prompt} {question}"
        
        try:
            if model_name == "janus":
                # Directly pass the PIL image object to `load_pil_images`
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{full_question}",
                        "images": [pil_image],  # Pass PIL image object directly
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                
                answer = get_janus_answer(model_data, conversation)
            elif model_name == "gemma":
                answer = get_gemma_answer(model_data, pil_image, full_question)
            elif model_name == "qwen":
                answer = get_qwen_answer(model_data, pil_image, full_question)
            
            # Save result immediately
            result = {
                "idx": i,
                "question": question,
                "gold_answers": gold_answers,
                "model_answer": answer,
            }
            
            with open(results_file, "a") as f:
                f.write(json.dumps(result) + "\n")
            
        except Exception as e:
            print(f"Error processing item {i} with model {model_name}: {e}")
            # Save error result
            result = {
                "idx": i,
                "question": question,
                "gold_answers": gold_answers,
                "model_answer": f"ERROR: {str(e)}",
            }
            
            with open(results_file, "a") as f:
                f.write(json.dumps(result) + "\n")
    
    print(f"OK-VQA processing complete. Results saved to {results_file}.")



def process_clevr(model_name, model_data, clevr_path, images_dir, output_dir):
    """Process CLEVR dataset with a specific model"""
    dataset = load_clevr_dataset(clevr_path, images_dir)
    
    # Create results file
    results_file = os.path.join(output_dir, f"clevr_results_{model_name}.jsonl")
    # Initialize the file
    with open(results_file, "w") as f:
        f.write("")
    
    # Model-specific prompts for CLEVR
    prompts = {
        "janus": "Answer with a single word. No explanation. Just the answer.",
        "gemma": "This image has 3D shapes with different properties. Answer this question with one word only:",
        "qwen": "Look at the 3D objects in this image and answer with only one word:"
    }
    
    prompt = prompts.get(model_name, "Answer with a single word:")
    
    print(f"Processing CLEVR dataset with {model_name} model ({len(dataset)} questions)...")
    
    for i, item in enumerate(tqdm(dataset)):
        image_path = item["image_path"]
        image = Image.open(image_path).convert("RGB")
        question = item["question"]
        gold_answer = item["answer"]
        
        full_question = f"{prompt} {question}"
        
        try:
            if model_name == "janus":
                # Create a conversation object similar to OK-VQA processing
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{full_question}",
                        "images": [image],  # Pass PIL image object directly
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                
                answer = get_janus_answer(model_data, conversation)
            elif model_name == "gemma":
                answer = get_gemma_answer(model_data, image, full_question)
            elif model_name == "qwen":
                answer = get_qwen_answer(model_data, image, full_question)
            
            # Save result immediately
            result = {
                "idx": i,
                "question": question,
                "gold_answer": gold_answer,
                "model_answer": answer,
                "image_filename": item["image_filename"]
            }
            
            with open(results_file, "a") as f:
                f.write(json.dumps(result) + "\n")
            
        except Exception as e:
            print(f"Error processing item {i} with model {model_name}: {e}")
            # Save error result
            result = {
                "idx": i,
                "question": question,
                "gold_answer": gold_answer,
                "model_answer": f"ERROR: {str(e)}",
                "image_filename": item["image_filename"]
            }
            
            with open(results_file, "a") as f:
                f.write(json.dumps(result) + "\n")
    
    print(f"CLEVR processing complete. Results saved to {results_file}.")

def main():
    parser = argparse.ArgumentParser(description="Model-Specific Zero-shot VQA Processing")
    
    # Model selection
    parser.add_argument("--model", type=str, default="gemma", choices=["janus", "gemma", "qwen"], 
                      help="Specify which model to run (janus, gemma, or qwen)")
    
    # Dataset selection
    parser.add_argument("--dataset", type=str, default="clevr", choices=["okvqa", "clevr", "both"], 
                      help="Specify which dataset to process (okvqa, clevr, or both)")
    
    # CLEVR dataset paths
    parser.add_argument("--clevr_path", type=str, 
                      default="/home/mkapadni/work/speech_project/clevr_dataset/CLEVR_val_questions.json", 
                      help="Path to CLEVR questions JSON file")
    parser.add_argument("--images_dir", type=str, 
                      default="/home/mkapadni/work/speech_project/clevr_dataset/images/val", 
                      help="Path to CLEVR images directory")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default="vqa_outputs", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the specified model only
    model_data = None
    if args.model == "janus":
        model_data = load_janus_model()
    elif args.model == "gemma":
        model_data = load_gemma_model()
    elif args.model == "qwen":
        model_data = load_qwen_model()
    
    # Process the specified dataset(s)
    if args.dataset in ["okvqa", "both"]:
        process_okvqa(args.model, model_data, args.output_dir)
    
    if args.dataset in ["clevr", "both"]:
        if not os.path.exists(args.clevr_path):
            print(f"Error: CLEVR questions file not found at {args.clevr_path}")
        elif not os.path.exists(args.images_dir):
            print(f"Error: CLEVR images directory not found at {args.images_dir}")
        else:
            process_clevr(args.model, model_data, args.clevr_path, args.images_dir, args.output_dir)
    
    print(f"Processing complete! Results saved to {args.output_dir}/ directory.")

if __name__ == "__main__":
    main()