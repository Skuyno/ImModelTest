import copy
import json
import random
import requests
import uuid
import time
from pathlib import Path
 
# Перед прогоном глянь:
# - в prompts_<model>.json параметр силы зови "strength" (читаем params["strength"])
# - Qwen 2511 и Flux2-Klein пока двух-картиночные — сделай одно-картиночные, иначе тянут лишний референс
# - тумблеры Lightning/Turbo в воркфлоу держи false (целимся в Primitive из ветки on_false)
# - я код не гонял, обкатай сперва на одной модели
 
server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())
 
INPUT_ROOT = Path("input")     # inputs/<task>/*.png — входные кадры по задачам
OUTPUT_ROOT = Path("output")
 
# Какие модели гоняем в этот раз (имена = ключи CONFIGS и имена prompts_<model>.json)
MODELS_TO_RUN = ["qwen-image-edit-2511", "flux1-kontext-dev", "firered-image-edit-1.1",
                 "flux2-klein-9b", "flux2-dev", "longcat-image-edit"]
 
# Словарь по которому можем менять нужные настройки в воркфлоу
CONFIGS = {
    "flux1-kontext-dev": {
        "workflow": "workflows/flux_kontext_dev_basic.json",
        "nodes": {
            "image":    ("190", "image"),
            "positive": ("192:6", "text"),
            "seed":     ("192:31", "seed"),
            "steps":    ("192:31", "steps"),
            "strength": ("192:35", "guidance"),
            # негатива нет (ConditioningZeroOut)
        },
    },
    "qwen-image-edit-2511": {
        "workflow": "workflows/image_qwen_image_edit_2511.json",
        "nodes": {
            "image":    ("41", "image"),
            "positive": ("170:151", "prompt"),
            "negative": ("170:149", "prompt"),
            "seed":     ("170:169", "seed"),
            "steps":    ("170:166", "value"),   # Primitive за свитчем, lightning off
            "strength": ("170:154", "value"),   # cfg-Primitive за свитчем
        },
    },
    "firered-image-edit-1.1": {
        "workflow": "workflows/image_firered_image_edit1_1.json",
        "nodes": {
            "image":    ("143", "image"),
            "positive": ("192:187", "prompt"),
            "negative": ("192:188", "prompt"),
            "seed":     ("192:189", "seed"),
            "steps":    ("192:174", "value"),   # Primitive за свитчем
            "strength": ("192:178", "value"),   # cfg-Primitive за свитчем
        },
    },
    "flux2-klein-9b": {
        "workflow": "workflows/image_flux2_klein_image_edit_9b_base.json",
        "nodes": {
            "image":    ("76", "image"),
            "positive": ("75:74", "text"),
            "negative": ("75:67", "text"),
            "seed":     ("75:73", "noise_seed"),
            "steps":    ("75:62", "steps"),
            "strength": ("75:63", "cfg"),
        },
    },
    "flux2-dev": {
        "workflow": "workflows/image_flux2.json",
        "nodes": {
            "image":    ("46", "image"),
            "positive": ("68:6", "text"),
            "seed":     ("68:25", "noise_seed"),
            "steps":    ("68:91", "value"),     # Primitive за свитчем
            "strength": ("68:26", "guidance"),
            # негатива нет (BasicGuider)
        },
    },
    "longcat-image-edit": {
        "workflow": "workflows/image_longcat_image_edit.json",
        "nodes": {
            "image":    ("13", "image"),
            "positive": ("22:4", "prompt"),
            "negative": ("22:5", "prompt"),
            "seed":     ("22:7", "seed"),
            "steps":    ("22:7", "steps"),
            "strength": ("22:7", "cfg"),
        },
    },
}
 
 
# Запускаем промпт в работу
def queue_prompt(workflow: dict) -> str:
    p = {"prompt": workflow, "client_id": client_id}
    r = requests.post(f"http://{server_address}/prompt", json=p)
    data = r.json()
    if "prompt_id" not in data:
        print(json.dumps(data, ensure_ascii=False, indent=2))   # покажет error / node_errors
        raise SystemExit("ComfyUI отклонил промпт — см. выше")
    return data["prompt_id"]
 
 
# Получаем инфу по нашему промпту
def get_history(prompt_id: str) -> dict:
    return requests.get(f"http://{server_address}/history/{prompt_id}").json()
 
 
# Получаем vram
def get_vram() -> dict:
    return requests.get(f"http://{server_address}/system_stats").json()["devices"][0]
 
 
# Закидываем картинку в инпут комфи, возвращаем имя для LoadImage
def upload_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        files = {"image": (image_path.name, f)}
        data = {"overwrite": "true"}
        resp = requests.post(f"http://{server_address}/upload/image", files=files, data=data)
    return resp.json()["name"]
 
 
# Подменяем инфу в нодах по карте из конфига
def inject(workflow_path: str, nodes: dict, values: dict) -> dict:
    with open(workflow_path, encoding="utf-8") as f:
        wf = json.load(f)
    for role, (node_id, field) in nodes.items():
        if values.get(role) is not None:
            wf[node_id]["inputs"][field] = values[role]
    return wf
 
 
# Просто ждём конца генерации, ничего не сохраняем (для прогрева)
def wait_done(prompt_id: str):
    while prompt_id not in get_history(prompt_id):
        time.sleep(0.1)
 
 
# Ждём генерацию, меряем метрики, сохраняем картинки
def run_and_save(prompt_id: str, dest_dir: Path, stem: str):
    start = time.perf_counter()
    vram_peak = 0
    while True:
        device = get_vram()
        used = (device["vram_total"] - device["vram_free"]) / (1024 * 1024)  # Получаем мегабайты
        vram_peak = max(vram_peak, used)
        out = get_history(prompt_id)
        if prompt_id in out:
            break
        time.sleep(0.1)
    gen_time = time.perf_counter() - start
 
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for node_id in out[prompt_id]["outputs"]:
        node_output = out[prompt_id]["outputs"][node_id]
        if "images" in node_output:
            for i, img in enumerate(node_output["images"]):
                resp = requests.get(f"http://{server_address}/view", params=img)
                ext = Path(img["filename"]).suffix or ".png"
                dest = dest_dir / f"{stem}_{i}{ext}"
                with open(dest, "wb") as fp:
                    fp.write(resp.content)
                saved.append(str(dest))
 
    return {"gen_time_s": gen_time, "vram_peak_mb": vram_peak}, saved
 
 
# Собираем словарь значений под подмену
def build_values(task: dict, image_name: str | None = None) -> dict:
    values = {
        "positive": task["positive"],
        "negative": task["negative"],
        "seed": random.randint(0, 2**32 - 1),
        "steps": task["params"]["steps"],
        "strength": task["params"]["strength"],
    }
    if image_name is not None:
        values["image"] = image_name
    return values
 
 
# Прогоняем одну модель целиком
def process_model(model: str):
    config = CONFIGS[model]
    with open(f"prompts/prompts_{model}.json", encoding="utf-8") as f:
        prompts_data = json.load(f)
    filled = copy.deepcopy(prompts_data)
    out_dir = OUTPUT_ROOT / model
    has_image = "image" in config["nodes"]

    # Грузим модель в врам, результат выбрасываем (Без этого у нас время будет не точно замеряться)
    first_name = next(iter(prompts_data["tasks"]))
    first_task = prompts_data["tasks"][first_name]
    warm_img = None
    if has_image:
        sample = next(iter((INPUT_ROOT / first_name).glob("*")))
        warm_img = upload_image(sample)
    warm_wf = inject(config["workflow"], config["nodes"], build_values(first_task, warm_img))
    wait_done(queue_prompt(warm_wf))

    # Проходимся по задачам, гоняем и пишем результат
    for task_name, task in prompts_data["tasks"].items():
        # edit-модель крутит по картинкам из inputs/<task>
        inputs = sorted((INPUT_ROOT / task_name).glob("*")) if has_image else [None]
 
        for image_path in inputs:
            image_name = upload_image(image_path) if image_path is not None else None
            values = build_values(task, image_name)
            stem = f"{task_name}_{values['seed']}"
            if image_path is not None:
                stem = f"{task_name}_{image_path.stem}_{values['seed']}"
 
            workflow = inject(config["workflow"], config["nodes"], values)
            prompt_id = queue_prompt(workflow)
            metrics, outputs = run_and_save(prompt_id, out_dir / task_name, stem)
 
            filled["tasks"][task_name]["results"].append({
                "input": str(image_path) if image_path else None,
                "seed": values["seed"],
                "output": outputs,
                "auto": metrics,
                "manual": {"quality": None, "artifacts": None, "adherence": None, "preservation": None},
            })
 
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(filled, f, ensure_ascii=False, indent=2)
 
 
if __name__ == "__main__":
    for model in MODELS_TO_RUN:
        process_model(model)