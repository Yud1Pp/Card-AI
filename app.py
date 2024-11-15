from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig
from flask_cors import CORS
from livereload import Server

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
model.eval()  # Pindahkan model ke evaluasi mode

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)

# Fungsi inferensi untuk menghasilkan respons dari model
def inference(model, tokenizer, input_sent):
    input_ids = tokenizer(input_sent, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    
    # Generate output dari model
    outputs = model.generate(input_ids=input_ids,
                             top_p=0.9,
                             max_length=512,
                             no_repeat_ngram_size=2,
                             num_beams=10,
                             repetition_penalty=2.0)
    return tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]

@app.route("/")
def home():
    return render_template("start.html")  # Render halaman utama (start.html)

@app.route("/tes")
def tes():
    return render_template("tes.html")  # Render halaman tes.html

@app.route("/chatbot", methods=["POST"])
def chatbot():
    # Verifikasi apakah input valid
    data = request.get_json()
    if not data or "input" not in data:
        return jsonify({"error": "Invalid input"}), 400

    input_text = data["input"]
    response = inference(model, tokenizer, input_text)
    print(f"User input: {input_text}, Response: {response}")
    return jsonify({"response": response})

if __name__ == "__main__":
    # Konfigurasikan LiveReload
    server = Server(app.wsgi_app)
    app.debug = True  # Aktifkan mode debug untuk Flask
    
    # Pantau perubahan di direktori proyek (templates, static, dll.)
    server.watch("templates/*")  # Pantau semua file di folder templates
    server.watch("static/*")     # Pantau semua file di folder static
    
    # Jalankan server
    server.serve(port=5100, debug=True)