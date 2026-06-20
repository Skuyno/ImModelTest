import copy
import json
import logging
import random
import shutil
import requests
import uuid
import time
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("run.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("runner")


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
        # ComfyUI забраковал промпт — печатаем причину (node_errors) и кидаем ошибку,
        # которую перехватит main и пойдёт к следующей модели
        log.error("ComfyUI отклонил промпт:\n%s", json.dumps(data, ensure_ascii=False, indent=2))
        raise RuntimeError("prompt rejected by ComfyUI")
    log.debug("queued prompt_id=%s", data["prompt_id"])
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
    name = resp.json()["name"]
    log.debug("upload %s -> %s", image_path.name, name)
    return name


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
                log.debug("saved %s", dest)

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


# Заводим папку под новый прогон (output/1, output/2, ...) + кладём копию промптов
def make_run_dir() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    used = [int(p.name) for p in OUTPUT_ROOT.iterdir() if p.is_dir() and p.name.isdigit()]
    run_dir = OUTPUT_ROOT / str(max(used, default=0) + 1)
    run_dir.mkdir()
    prompts_copy = run_dir / "prompts"
    prompts_copy.mkdir()
    for model in MODELS_TO_RUN:
        src = Path(f"prompts/prompts_{model}.json")
        if src.exists():
            shutil.copy2(src, prompts_copy / src.name)
    return run_dir


# Прогоняем одну модель целиком
def process_model(model: str, run_dir: Path):
    t0 = time.perf_counter()
    log.info("=== МОДЕЛЬ: %s ===", model)

    config = CONFIGS[model]
    with open(f"prompts/prompts_{model}.json", encoding="utf-8") as f:
        prompts_data = json.load(f)
    filled = copy.deepcopy(prompts_data)
    out_dir = run_dir / model
    has_image = "image" in config["nodes"]
    n_tasks = len(prompts_data["tasks"])
    log.info("воркфлоу: %s | задач: %d | edit-режим: %s", config["workflow"], n_tasks, has_image)

    # Грузим модель в врам, результат выбрасываем (Без этого у нас время будет не точно замеряться)
    first_name = next(iter(prompts_data["tasks"]))
    first_task = prompts_data["tasks"][first_name]
    warm_img = None
    if has_image:
        sample = next(iter((INPUT_ROOT / first_name).glob("*")))
        warm_img = upload_image(sample)
    log.info("прогрев — грузим модель в VRAM (результат выбрасываем)...")
    warm_wf = inject(config["workflow"], config["nodes"], build_values(first_task, warm_img))
    wait_done(queue_prompt(warm_wf))
    log.info("прогрев готов (%.0fs)", time.perf_counter() - t0)

    # Проходимся по задачам, гоняем и пишем результат
    for ti, (task_name, task) in enumerate(prompts_data["tasks"].items(), 1):
        # edit-модель крутит по картинкам из inputs/<task>
        inputs = sorted((INPUT_ROOT / task_name).glob("*")) if has_image else [None]
        log.info("[задача %d/%d] '%s' — картинок: %d", ti, n_tasks, task_name, len(inputs))

        # копим метрики по всем картинкам задачи, потом усредняем
        times, vrams, all_outputs = [], [], []
        for ii, image_path in enumerate(inputs, 1):
            image_name = upload_image(image_path) if image_path is not None else None
            values = build_values(task, image_name)
            stem = f"{task_name}_{values['seed']}"
            if image_path is not None:
                stem = f"{task_name}_{image_path.stem}_{values['seed']}"

            workflow = inject(config["workflow"], config["nodes"], values)
            prompt_id = queue_prompt(workflow)
            metrics, outputs = run_and_save(prompt_id, out_dir / task_name, stem)

            log.info("   img %d/%d %-22s | seed=%d | %.1fs | VRAM %.0f MB | файлов: %d",
                     ii, len(inputs),
                     image_path.name if image_path is not None else "(text2img)",
                     values["seed"], metrics["gen_time_s"], metrics["vram_peak_mb"], len(outputs))
            if metrics["gen_time_s"] < 1.0:
                log.warning("   ВНИМАНИЕ: время %.2fs подозрительно мало — похоже на кэш-хит ComfyUI",
                            metrics["gen_time_s"])

            times.append(metrics["gen_time_s"])
            vrams.append(metrics["vram_peak_mb"])
            all_outputs.extend(outputs)

        n = len(times)
        avg_time = sum(times) / n if n else 0.0
        avg_vram = sum(vrams) / n if n else 0.0
        filled["tasks"][task_name]["results"] = {
            "n_images": n,
            "avg": {
                "gen_time_s": round(avg_time, 2),
                "vram_peak_mb": round(avg_vram, 1),
            },
            "outputs": all_outputs,
            "manual": {"quality": None, "artifacts": None, "adherence": None, "preservation": None},
        }
        log.info("[задача %d/%d] '%s' ИТОГ — среднее: %.1fs | VRAM %.0f MB | картинок: %d",
                 ti, n_tasks, task_name, avg_time, avg_vram, n)

    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(filled, f, ensure_ascii=False, indent=2)
    log.info("=== %s ГОТОВО за %.0fs | результаты: %s ===", model, time.perf_counter() - t0, res_path)


if __name__ == "__main__":
    run_dir = make_run_dir()
    log.info("СТАРТ прогона #%s -> %s. Модели (%d): %s",
             run_dir.name, run_dir, len(MODELS_TO_RUN), ", ".join(MODELS_TO_RUN))
    t_all = time.perf_counter()
    ok, failed = [], []
    for model in MODELS_TO_RUN:
        try:
            process_model(model, run_dir)
            ok.append(model)
        except Exception:
            # полный трейс в лог, но прогон не роняем — идём к следующей модели
            log.exception("МОДЕЛЬ %s УПАЛА — пропускаю, иду дальше", model)
            failed.append(model)
    log.info("ВСЁ. Прогон #%s за %.0fs | успешно: %s | упало: %s",
             run_dir.name, time.perf_counter() - t_all, ok or "—", failed or "—")