from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig
from flask_cors import CORS

def load_model_tokenizer():
    peft_model_id = "Yudsky/lora-Medical-flan-T5-small"
    config = PeftConfig.from_pretrained(peft_model_id)

    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    model = PeftModel.from_pretrained(model, peft_model_id)
    return model, tokenizer

# Load model dan tokenizer
model, tokenizer = load_model_tokenizer()

# Menentukan device (GPU jika tersedia, jika tidak CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)

# Fungsi inferensi untuk menghasilkan respons dari model
def inference(model, tokenizer, input_sent):
    input_ids = tokenizer(input_sent, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    
    outputs = model.generate(
        input_ids=input_ids,
        top_p=0.9,
        max_length=512,
        no_repeat_ngram_size=2,
        num_beams=10,
        repetition_penalty=2.0
    )
    return tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]

@app.route("/")
def home():
    return render_template("start.html")

@app.route("/tes")
def tes():
    return render_template("tes.html")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    if not data or "input" not in data:
        return jsonify({"error": "Invalid input"}), 400

    input_text = data["input"]
    response = inference(model, tokenizer, input_text)
    return jsonify({"response": response})

if __name__ == "__main__":
    # Jalankan server Flask tanpa livereload
    app.run(host="0.0.0.0", port=5000)
