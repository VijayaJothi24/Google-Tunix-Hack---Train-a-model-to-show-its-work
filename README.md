

Gemma2 2B IT Hackathon Notebook
![Google Maps AI Agent](https://github.com/VijayaJothi24/Google-Tunix-Hack---Train-a-model-to-show-its-work/blob/main/Google_Project.png)


## 📌 Overview
This notebook demonstrates fine-tuning and evaluation of the **Gemma2 2B Instruction-Tuned (IT)** model on Kaggle.  
It follows the hackathon requirements:
- Deterministic inference parameters
- Checkpoint saving during a 9-hour run
- Single-session evaluation
- Optional unrestricted mode with Kaggle Model upload

[![Watch the video](https://img.youtube.com/vi/6m1ILiAe-4g/0.jpg)](https://youtu.be/6m1ILiAe-4g)


 ```html<div align="center"> <iframewidth="720" height="405" src="https://www.youtube.com/embed/6m1ILiAe-4g"  title="YouTube video player"frameborder="0"  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>

  
   ## ⚙️ Environment Setup
Install required libraries:
```bash
%pip install -q transformers torch jax flax orbax-checkpoint kagglehub wandb
Authentication
Weights & Biases (W&B): Add your WANDB_API_KEY as a Kaggle Secret or environment variable.

python
import os
os.environ["WANDB_API_KEY"] = "your_wandb_api_key_here"
Hugging Face Hub: Request access to Gemma2 2B IT. Add your Hugging Face token:

python
os.environ["HUGGINGFACE_HUB_TOKEN"] = "your_hf_token_here"
Kaggle API (optional for dataset/model upload): Place your kaggle.json in ~/.kaggle/.

🚀 Usage
Load base model and tokenizer

python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ["HUGGINGFACE_HUB_TOKEN"])
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=os.environ["HUGGINGFACE_HUB_TOKEN"])
Fine-tuning loop

Train with LoRA adapters.

Save checkpoints regularly:

python
save_actor_checkpoint(step, lora_params)
Evaluation

Load the latest checkpoint.

Run deterministic inference with:

Code
TEMPERATURE=1e-4, TOP_K=1, TOP_P=1.0, MAX_GENERATION_STEPS=768, SEED=42
Unrestricted Mode (optional)

Upload final Flax-format checkpoints to Kaggle Models.

Set unrestricted_kaggle_model = "username/model_name".

📂 Project Structure
Code
/kaggle/working/
  ├── ckpts/actor/<step>/model_params   # Saved checkpoints
  ├── unrestricted/jax/size/...          # For Kaggle Model upload
README.md                                # This file
notebook.ipynb                           # Main notebook
📝 Notes
Ensure at least one checkpoint is saved during the 9-hour run.

The last checkpoint will be used for evaluation.

Hugging Face access is required for Gemma2 models.

Keep all API keys private (use Kaggle Secrets).

🙌 Reflections
Learned about gated model access and secure token handling.

Faced challenges with JAX/CUDA plugin compatibility.

Suggestions: better Kaggle GPU support for JAX, streamlined Hugging Face gated repo access.
