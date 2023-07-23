import gradio as gr
import argparse
import datetime
import os, sys
import numpy as np
import midi_util
import json
import mido
from midi_util import VocabConfig
np.set_printoptions(precision=4, suppress=True, linewidth=200)

os.environ["RWKV_JIT_ON"] = "1" #### set these before import RWKV

settings = {
    "device": "GPU(cuda and cl required)",
    "strategy": "cuda fp16",
    "RWKV model": [x for x in os.listdir("./models") if (x.endswith(".pth") or x.endswith(".pt") or x.endswith(".ckpt"))][0],
    "batch num": 1,
    "output type": "txt and midi",
    "temperature": 1,
    "top_k": 8,
    "top_p": 0.8
}#default settings of webui

if os.path.exists("webui_settings.json"):
    with open("webui_settings.json", "r") as f:
        settings = json.load(f)

print(settings)

def getLastAvailableName(output_format):
    time = datetime.datetime.today()
    datemark = f"{time.year}_{time.month}_{time.day}"
    if output_format == "txt only":
        if os.path.exists(f"./raw/{datemark}"):
            files = [x.replace(".txt", "") for x in os.listdir(f"./raw/{datemark}")]
        else:
            return 0
    elif output_format == "txt and midi":
        if os.path.exists(f"./midi/{datemark}"):
            files = [x.replace(".mid", "").replace(".midi", "") for x in os.listdir(f"./midi/{datemark}")]
        else:
            return 0
    files_formatted = [x for x in files if x[0:5].isdigit()]
    if len(files_formatted) == 0:
        return 0
    return int(files_formatted[-1][0:5]) + 1

def inference(device,strategy,model,batch_num,output_format,input_file,temperature,top_k,top_p):
    settings_new = {
        "device": device,
        "strategy": strategy,
        "RWKV model": model,
        "batch num": batch_num,
        "output type": output_format,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p
    }
    sf = open("webui_settings.json", "w")
    json.dump(settings_new, sf)
    sf.close()

    time = datetime.datetime.today()
    datemark = f"{time.year}_{time.month}_{time.day}"
    timemark = f"{time.year}_{time.month}_{time.day}_{time.hour}_{time.minute}_{int(time.second)}"
    last_count = getLastAvailableName(output_format)
    if device == "CPU":
        os.environ["RWKV_CUDA_ON"] = "0"
    else:
        os.environ["RWKV_CUDA_ON"] = "1"

    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE

    MODEL_FILE = "./models/"+model
    model = RWKV(model=MODEL_FILE, strategy=strategy)
    pipeline = PIPELINE(model, "./tokenizer-midi.json")
    cfg = VocabConfig.from_json("./vocab_config.json")

    text: str
    if input_file == None:
        text = ""
        formatted_filename = "None"
    else:
        if input_file.name.endswith(".mid") or input_file.name.endswith(".midi"):
            mid = mido.MidiFile(input_file.name)
            text = midi_util.convert_midi_to_str(cfg, mid)
            formatted_filename = input_file.name.replace(".midi","").replace(".mid","") + "_midi"
        elif input_file.name.endswith(".txt"):
            with open(input_file.name, "r") as f:
                text = f.read()
            formatted_filename = input_file.name.replace(".txt","") + "_txt"
        text = text.strip()
        text.replace("<start>", "")
        text.replace("<end>", "")

    for TRIAL in range(batch_num):
        if batch_num > 1:
            print(f"{TRIAL + 1} of {batch_num} midi is building...")
        else:
            print("1 midi is building...")
        # ccc = "<pad>" # the model is trained using <pad> (0) as separator
        # ccc_output = "<start>" # however the str_to_midi.py requires <start> & <end> as separator

        # uncomment this to continue a melody -> style similar to sample2 & sample3
        ccc = text
        ccc = "<pad> " + ccc
        ccc_output = "<start> " + ccc

        if not os.path.exists(f"./raw/{datemark}"):
            os.makedirs(f"./raw/{datemark}")
        if not os.path.exists(f"./midi/{datemark}"):
            os.makedirs(f"./midi/{datemark}")

        fout = open(f"./raw/{datemark}/{last_count + TRIAL:05d}_{formatted_filename}_{timemark}.txt", "w")
        fout.write(ccc_output)

        occurrence = {}
        state = None
        for i in range(4096):  # only trained with ctx4096 (will be longer soon)

            if i == 0:
                out, state = model.forward(pipeline.encode(ccc), state)
            else:
                out, state = model.forward([token], state)

            for n in occurrence:
                out[n] -= (0 + occurrence[n] * 0.5)

            out[0] += (i - 2000) / 500  # try not to be too short or too long
            out[127] -= 1  # avoid "t125"

            # uncomment for piano-only mode
            # out[128:12416] -= 1e10
            # out[13952:20096] -= 1e10

            # find the best sampling for your taste
            # token = pipeline.sample_logits(out, temperature=1.0, top_k=8, top_p=0.8)
            token = pipeline.sample_logits(out, temperature=temperature, top_k=top_k, top_p=top_p)

            if token == 0: break

            for n in occurrence: occurrence[n] *= 0.997  #### decay repetition penalty
            if token >= 128 or token == 127:
                occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
            else:
                occurrence[token] = 0.3 + (occurrence[token] if token in occurrence else 0)

            fout.write(" " + pipeline.decode([token]))
            fout.flush()

        fout.write(" <end>")
        fout.close()
        if output_format == "txt and midi":
            with open(f"./raw/{datemark}/{last_count + TRIAL:05d}_{formatted_filename}_{timemark}.txt", "r") as f:
                mid = midi_util.convert_str_to_midi(cfg, f.read())
                mid.save(os.path.abspath(f"./midi/{datemark}/{last_count + TRIAL:05d}_{formatted_filename}_{timemark}.mid"))

    print("task completed successfully.")
    if output_format == "txt only":
        return [f"./raw/{datemark}/{last_count + x:05d}_{formatted_filename}_{timemark}.txt" for x in range(batch_num)]
    else:
        return [f"./midi/{datemark}/{last_count + x:05d}_{formatted_filename}_{timemark}.mid" for x in range(batch_num)]


iface = gr.Interface(
    fn=inference,
    title="RMKV midi creation",
    inputs=[
        gr.Radio(["CPU", "GPU(cuda and cl required)"], label="device", show_label=True, value=settings["device"]),
        gr.Radio(["cpu fp32",
                  "cpu bf16",
                  "cpu fp32i8",
                  "cuda fp16",
                  "cuda fp16i8",
                  "cuda fp16i8 *20 -> cuda fp16",
                  "cuda fp16i8 *20+",
                  "cuda fp16i8 *20 -> cpu fp32",
                  "cuda:0 fp16 *10 -> cuda:1 fp16",
                  "cuda:0 fp16 *10 -> cuda:1 fp16 *8 -> cpu fp32",
                  ], label="strategy", show_label=True, value=settings["strategy"]),
        gr.Dropdown(choices=[x for x in os.listdir("./models") if (x.endswith(".pth") or x.endswith(".pt") or x.endswith(".ckpt"))], label="RWKV model", show_label=True, value=settings["RWKV model"]),
        gr.Slider(1, 100, label="batch num", show_label=True, step=1, value=settings["batch num"]),
        gr.Radio(["txt only", "txt and midi"], label="output type", show_label=True, value=settings["output type"]),
        gr.File(file_types=[".txt", ".midi", ".mid"], label="input file (optional)", show_label=True),
        gr.Slider(0, 2, label="temperature", show_label=True, value=settings["temperature"]),
        gr.Slider(0, 10, label="top_k", show_label=True, value=settings["top_k"]),
        gr.Slider(0, 1, label="top_p", show_label=True, value=settings["top_p"]),
    ],
    outputs=[gr.File()]
)
iface.launch()
