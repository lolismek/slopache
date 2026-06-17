#!/usr/bin/env python3
"""Build + submit a Wan 2.2 I2V (4-step LightX2V) job to ComfyUI and wait for it.
Runs ON the box. Animates one staged still (ComfyUI/input/<image>) into a clip.

Usage: run_wan_i2v.py <input_image> <filename_prefix> [width height length seed]
"""
import json, sys, time, urllib.request

HOST = "http://127.0.0.1:8188"
img    = sys.argv[1] if len(sys.argv) > 1 else "bunica_bus.jpg"
prefix = sys.argv[2] if len(sys.argv) > 2 else "bunica_i2v_test"
W   = int(sys.argv[3]) if len(sys.argv) > 3 else 768
H   = int(sys.argv[4]) if len(sys.argv) > 4 else 768
LEN = int(sys.argv[5]) if len(sys.argv) > 5 else 81
SEED= int(sys.argv[6]) if len(sys.argv) > 6 else 42

import os
POS = os.environ.get("POS_PROMPT",
      ("A 3D Pixar-style elderly Romanian grandmother sitting on a city bus. "
       "She waves at the camera with one hand, smiles warmly and laughs, leaning "
       "slightly forward. The city street scrolls past the bus window behind her. "
       "Gentle handheld camera motion, natural lively movement."))
NEG = os.environ.get("NEG_PROMPT",
      ("static, still, frozen, motionless, blurry, low quality, distorted, "
       "deformed face, extra fingers, watermark, text, jpeg artifacts, oversaturated"))

wf = {
 "1":  {"class_type":"LoadImage","inputs":{"image":img}},
 "2":  {"class_type":"CLIPLoader","inputs":{"clip_name":"umt5_xxl_fp8_e4m3fn_scaled.safetensors","type":"wan"}},
 "3":  {"class_type":"CLIPTextEncode","inputs":{"text":POS,"clip":["2",0]}},
 "4":  {"class_type":"CLIPTextEncode","inputs":{"text":NEG,"clip":["2",0]}},
 "5":  {"class_type":"UNETLoader","inputs":{"unet_name":"wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors","weight_dtype":"default"}},
 "6":  {"class_type":"ModelSamplingSD3","inputs":{"model":["5",0],"shift":5.0}},
 "7":  {"class_type":"LoraLoaderModelOnly","inputs":{"model":["6",0],"lora_name":"wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors","strength_model":1.0}},
 "8":  {"class_type":"UNETLoader","inputs":{"unet_name":"wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors","weight_dtype":"default"}},
 "9":  {"class_type":"ModelSamplingSD3","inputs":{"model":["8",0],"shift":5.0}},
 "10": {"class_type":"LoraLoaderModelOnly","inputs":{"model":["9",0],"lora_name":"wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors","strength_model":1.0}},
 "11": {"class_type":"VAELoader","inputs":{"vae_name":"wan_2.1_vae.safetensors"}},
 "12": {"class_type":"WanImageToVideo","inputs":{"positive":["3",0],"negative":["4",0],"vae":["11",0],
         "width":W,"height":H,"length":LEN,"batch_size":1,"start_image":["1",0]}},
 # 4-step fast path: total steps=4, split at 2 (high-noise 0->2, low-noise 2->4), cfg 1
 "13": {"class_type":"KSamplerAdvanced","inputs":{"model":["7",0],"add_noise":"enable","noise_seed":SEED,
         "steps":4,"cfg":1.0,"sampler_name":"euler","scheduler":"simple",
         "positive":["12",0],"negative":["12",1],"latent_image":["12",2],
         "start_at_step":0,"end_at_step":2,"return_with_leftover_noise":"enable"}},
 "14": {"class_type":"KSamplerAdvanced","inputs":{"model":["10",0],"add_noise":"disable","noise_seed":SEED,
         "steps":4,"cfg":1.0,"sampler_name":"euler","scheduler":"simple",
         "positive":["12",0],"negative":["12",1],"latent_image":["13",0],
         "start_at_step":2,"end_at_step":4,"return_with_leftover_noise":"disable"}},
 "15": {"class_type":"VAEDecode","inputs":{"samples":["14",0],"vae":["11",0]}},
 "16": {"class_type":"CreateVideo","inputs":{"images":["15",0],"fps":16.0}},
 "17": {"class_type":"SaveVideo","inputs":{"video":["16",0],"filename_prefix":prefix,"format":"mp4","codec":"h264"}},
}

data = json.dumps({"prompt": wf}).encode()
r = urllib.request.urlopen(urllib.request.Request(HOST+"/prompt", data=data,
        headers={"Content-Type":"application/json"}), timeout=30)
pid = json.load(r)["prompt_id"]
print(f"[i2v] submitted prompt_id={pid}  ({W}x{H}, {LEN}f, seed {SEED})", flush=True)

t0 = time.time()
while True:
    time.sleep(5)
    h = json.load(urllib.request.urlopen(HOST+f"/history/{pid}", timeout=30))
    if pid in h:
        outs = h[pid].get("outputs", {})
        status = h[pid].get("status", {}).get("status_str", "")
        files = []
        for node in outs.values():
            for key in ("videos","gifs","images"):
                for v in node.get(key, []):
                    files.append(v.get("filename"))
        print(f"[i2v] DONE status={status} elapsed={time.time()-t0:.0f}s files={files}", flush=True)
        break
    if time.time()-t0 > 1200:
        print("[i2v] TIMEOUT", flush=True); break
