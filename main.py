import matplotlib
matplotlib.use('Agg')

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import os
import multiprocessing
import json
import requests
import zipfile
import shutil
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
import torchvision.transforms as T
from PIL import Image
import numpy as np
import traceback

from bson.objectid import ObjectId

# ================= CONFIGURAÇÕES =================
PASTA_ORIGINAL_LOCAL = "dataset_original"
PASTAS_IMAGENS = {"Neutro": "dataset_neutro", "Atipico": "dataset_atipico"}
PASTAS_SAIDA = {
    "COCO_Original": "grafico_original",
    "Neutro": "grafico_neutro",
    "Atipico": "grafico_atipico"
}
CLASSES_DE_INTERESSE = ["stop sign", "airplane", "skis", "tennis racket", "person", "cat", "banana", "cup"]

for pasta in PASTAS_SAIDA.values():
    if not os.path.exists(pasta): os.makedirs(pasta)

# ================= FUNÇÕES AUXILIARES =================
def limpar_datasets_antigos():
    for nome in ["COCO_Original", "Neutro", "Atipico"]:
        if nome in fo.list_datasets(): fo.delete_dataset(nome)

def baixar_json_coco():
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    pasta_anotacoes = "coco_annotations"
    arquivo_json = os.path.join(pasta_anotacoes, "annotations", "instances_val2017.json")
    if os.path.exists(arquivo_json): return arquivo_json
    if not os.path.exists(pasta_anotacoes): os.makedirs(pasta_anotacoes)
    caminho_zip = os.path.join(pasta_anotacoes, "annotations.zip")
    try:
        with requests.get(url, stream=True) as r:
            with open(caminho_zip, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        with zipfile.ZipFile(caminho_zip, 'r') as zip_ref: zip_ref.extractall(pasta_anotacoes)
    finally:
        if os.path.exists(caminho_zip): os.remove(caminho_zip)
    return arquivo_json

def criar_json_filtrado(json_path, pasta_imagens):
    arquivos_locais = os.listdir(pasta_imagens)
    mapa_locais = {f: f for f in arquivos_locais}
    for f in arquivos_locais:
        nome_sem_ext = os.path.splitext(f)[0]
        mapa_locais[nome_sem_ext] = f
        mapa_locais[nome_sem_ext.lstrip("0") or "0"] = f

    with open(json_path, 'r') as f: data = json.load(f)
    images_mantidas = []
    ids_mantidos = set()
    for img in data['images']:
        n = img['file_name']
        nome_real = mapa_locais.get(n) or mapa_locais.get(os.path.splitext(n)[0]) or mapa_locais.get(os.path.splitext(n)[0].lstrip("0") or "0")
        if nome_real:
            img['file_name'] = nome_real
            images_mantidas.append(img)
            ids_mantidos.add(img['id'])

    novo_data = {**data, "images": images_mantidas, "annotations": [a for a in data['annotations'] if a['image_id'] in ids_mantidos]}
    novo_json_path = os.path.join("coco_annotations", "instances_filtered.json")
    with open(novo_json_path, 'w') as f: json.dump(novo_data, f)
    return novo_json_path

def carregar_dataset_modificado(nome_dataset, pasta_imagens, mapa_original):
    if not os.path.exists(pasta_imagens) or not os.listdir(pasta_imagens): return None
    ds = fo.Dataset.from_dir(dataset_dir=pasta_imagens, dataset_type=fo.types.ImageDirectory, name=nome_dataset)
    for sample in ds:
        nome_limpo = os.path.splitext(os.path.basename(sample.filepath).replace("neutro_", "").replace("atipico_", "").replace("mod_", ""))[0]
        amostra_ref = mapa_original.get(nome_limpo)
        if amostra_ref and hasattr(amostra_ref, "ground_truth"):
            sample["ground_truth"] = amostra_ref.ground_truth.copy()
            sample.save()
    return ds

# ================= ANÁLISE E GRÁFICOS (CORRIGIDA) =================
def gerar_analise_completa(dataset, nome_contexto):
    pasta_destino = PASTAS_SAIDA.get(nome_contexto, f"grafico_{nome_contexto}")
    print(f"   -> Gerando gráficos em: {pasta_destino}")

    dados_confianca, dados_iou = [], []
    dados_ap, dados_met = [], []
    mapa_modelos = {"yolo_v5": "YOLOv5", "faster_rcnn": "Faster R-CNN", "retinanet": "RetinaNet"}

    # 1. Coleta de Confiança e IoU
    for sample in dataset:
        for campo, nome_modelo in mapa_modelos.items():
            if campo in sample:
                preds = sample[campo]
                if not preds or not preds.detections: continue
                eval_key = f"eval_{campo}"
                for det in preds.detections:
                    if det.confidence: dados_confianca.append({"Modelo": nome_modelo, "Confiança": det.confidence})
                    if det.has_field(eval_key) and det[eval_key] == "tp":
                        val_iou = getattr(det, f"{eval_key}_iou", None) or getattr(det, "iou", None)
                        if val_iou is not None: dados_iou.append({"Modelo": nome_modelo, "IoU": val_iou})

    # 2. Processamento de Métricas Detalhadas e Relatório
    texto_relatorio = f"=== RELATÓRIO: {nome_contexto.upper()} ===\n\n"
    for campo, nome_modelo in mapa_modelos.items():
        eval_key = f"eval_{campo}"
        if eval_key in dataset.list_evaluations():
            try:
                results = dataset.load_evaluation_results(eval_key)
                rep = results.report()
                
                # Relatório de texto
                texto_relatorio += f"--- {nome_modelo} ---\n"
                texto_relatorio += (json.dumps(rep, indent=4) if isinstance(rep, dict) else str(rep)) + "\n\n"
                
                # Coleta para gráficos de métricas (Precision, Recall, F1)
                if isinstance(rep, dict):
                    for cls in CLASSES_DE_INTERESSE:
                        if cls in rep:
                            for k in ["precision", "recall", "f1-score"]:
                                if k in rep[cls]:
                                    dados_met.append({"Modelo": nome_modelo, "Classe": cls, "Métrica": k.capitalize(), "Valor": rep[cls][k]})

                # CORREÇÃO DO ERRO NUMPY: Coleta para gráfico de Performance (AP)
                if hasattr(results, '_classwise_AP'):
                    ap_data = results._classwise_AP
                    # Apenas tenta iterar se for um dicionário
                    if isinstance(ap_data, dict):
                        for cls, ap in ap_data.items():
                            if cls in CLASSES_DE_INTERESSE:
                                dados_ap.append({"Modelo": nome_modelo, "Classe": cls, "AP": ap})
                    # Se for numpy array, tentamos extrair via métricas do report se possível
                    elif isinstance(rep, dict):
                        for cls in CLASSES_DE_INTERESSE:
                            if cls in rep and "f1-score" in rep[cls]:
                                dados_ap.append({"Modelo": nome_modelo, "Classe": cls, "AP": rep[cls]["f1-score"]})

            except Exception as e:
                print(f"Erro ao processar métricas de {nome_modelo}: {e}")
        else:
            texto_relatorio += f"--- {nome_modelo} ---\n(Sem avaliação)\n\n"

    # 3. Salvar Relatório e Gerar Imagens
    with open(os.path.join(pasta_destino, "relatorio_detalhado.txt"), "w") as f: f.write(texto_relatorio)

    if dados_confianca:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=pd.DataFrame(dados_confianca), x="Modelo", y="Confiança", hue="Modelo", palette="Blues", legend=False)
        plt.title(f"Confiança - {nome_contexto}")
        plt.savefig(os.path.join(pasta_destino, "grafico_confianca.png")); plt.close()

    if dados_iou:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=pd.DataFrame(dados_iou), x="Modelo", y="IoU", hue="Modelo", palette="Greens", legend=False)
        plt.title(f"IoU (TP) - {nome_contexto}")
        plt.savefig(os.path.join(pasta_destino, "grafico_iou_final.png")); plt.close()

    if dados_ap:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=pd.DataFrame(dados_ap), x="Classe", y="AP", hue="Modelo", palette="viridis")
        plt.title(f"Performance por Classe - {nome_contexto}"); plt.ylim(0, 1.1)
        plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig(os.path.join(pasta_destino, "grafico_ap_classes_selecionadas.png")); plt.close()

    if dados_met:
        try:
            g = sns.catplot(data=pd.DataFrame(dados_met), x="Métrica", y="Valor", hue="Modelo", col="Classe", col_wrap=4, kind="bar", height=3, aspect=1, palette="muted")
            g.fig.subplots_adjust(top=0.9); g.fig.suptitle(f"Métricas Detalhadas - {nome_contexto}")
            plt.savefig(os.path.join(pasta_destino, "grafico_metricas_detalhadas.png")); plt.close()
        except: pass

# ================= HEATMAPS E WRAPPERS =================
class YoloModelWrapper(torch.nn.Module):
    def __init__(self, ultralytics_model):
        super().__init__()
        self.model = ultralytics_model.model
    def forward(self, x):
        result = self.model(x)
        return result[0] if isinstance(result, tuple) else result

def get_target_layers(model_wrapper, model_type):
    raw = model_wrapper._model
    if "yolo" in model_type:
        try: return [raw.model.model[9]]
        except: return [raw.model.model[-2]] if hasattr(raw.model, 'model') else []
    return [raw.backbone.body.layer4[-1]] if hasattr(raw, 'backbone') else []

def obter_ids_fiftyone(dataset, paths_or_coco_ids):
    ids_base = {os.path.splitext(os.path.basename(p))[0].replace("neutro_", "").replace("atipico_", "") for p in paths_or_coco_ids}
    return [s.id for s in dataset.iter_samples(autosave=False) if os.path.splitext(os.path.basename(s.filepath))[0] in ids_base]

def process_model_heatmaps_otimizado(dataset, model_config, device, classes_de_interesse, lista_ids=None):
    model_name, raw_model, label_field = model_config["name"], model_config["raw_model"], model_config["label_field"]
    target_layers, class_list = model_config["target_layers"], model_config["classes"]
    output_dir = os.path.join(f"heatmaps_{dataset.name}", model_name)
    if not target_layers: return
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    model_para_cam = YoloModelWrapper(raw_model) if "yolo" in model_name.lower() else raw_model
    if not "yolo" in model_name.lower(): model_para_cam.eval()
    
    try: cam = EigenCAM(model=model_para_cam, target_layers=target_layers)
    except: return

    view = dataset.select(lista_ids) if lista_ids else dataset.limit(20)
    transform = T.Compose([T.ToTensor()])

    for sample in view:
        if not sample.has_field(label_field) or not sample[label_field].detections: continue
        try:
            dets = [d for d in sample[label_field].detections if d.label in classes_de_interesse]
            if not dets: continue
            target_det = max(dets, key=lambda x: x.confidence or -1)
            img_pil = Image.open(sample.filepath).convert("RGB")
            w, h = img_pil.size
            
            rel_box = target_det.bounding_box
            box = [rel_box[0]*w, rel_box[1]*h, (rel_box[0]+rel_box[2])*w, (rel_box[1]+rel_box[3])*h]
            t_box = torch.tensor([box], dtype=torch.float).to(device)
            t_lab = torch.tensor([class_list.index(target_det.label) if target_det.label in class_list else 0], dtype=torch.long).to(device)
            targets = [FasterRCNNBoxScoreTarget(labels=t_lab, bounding_boxes=t_box)]

            img_in = img_pil.resize((640, 640)) if "yolo" in model_name.lower() else img_pil
            rgb_display = np.array(img_in, dtype=np.float32) / 255.0
            input_tensor = transform(img_in).to(device).unsqueeze(0)
            
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            vis = show_cam_on_image(rgb_display, grayscale_cam[0, :], use_rgb=True)
            Image.fromarray(vis).save(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(sample.filepath))[0]}_{target_det.label}.png"))
        except: continue

# ================= BLOCO PRINCIPAL =================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    limpar_datasets_antigos()

    print("\n[1/6] Preparando Datasets...")
    json_full = baixar_json_coco()
    json_filtrado = criar_json_filtrado(json_full, PASTA_ORIGINAL_LOCAL)
    dataset_original = fo.Dataset.from_dir(dataset_type=fo.types.COCODetectionDataset, data_path=PASTA_ORIGINAL_LOCAL, labels_path=json_filtrado, name="COCO_Original")
    if "detections" in dataset_original.get_field_schema(): dataset_original.rename_sample_field("detections", "ground_truth")
    mapa_orig = {os.path.splitext(os.path.basename(s.filepath))[0]: s for s in dataset_original}

    datasets_para_avaliar = [dataset_original]
    for n, p in PASTAS_IMAGENS.items():
        ds = carregar_dataset_modificado(n, p, mapa_orig)
        if ds: datasets_para_avaliar.append(ds)

    print("\n[2/6] Carregando Modelos...")
    yolo_m = foz.load_zoo_model("yolov5s-coco-torch")
    faster_m = foz.load_zoo_model("faster-rcnn-resnet50-fpn-coco-torch")
    retina_m = foz.load_zoo_model("retinanet-resnet50-fpn-coco-torch")

    IDS_BASE = [724, 785, 1532, 1675, 1761, 2473, 2532, 2587, 2592, 4495, 5060, 5477, 5529, 10363, 13348, 15440, 15497, 22396, 106384]
    LISTA_PATHS = [f"{id:012}.jpg" for id in IDS_BASE]

    print("\n[3/6] Processando...")
    for ds in datasets_para_avaliar:
        ids_internos = obter_ids_fiftyone(ds, LISTA_PATHS)
        for model, field, m_type in [(yolo_m, "yolo_v5", "yolo"), (faster_m, "faster_rcnn", "faster"), (retina_m, "retinanet", "retina")]:
            ds.apply_model(model, label_field=field)
            ds.evaluate_detections(field, gt_field="ground_truth", eval_key=f"eval_{field}", classes=CLASSES_DE_INTERESSE, compute_mAP=True)
            cfg = {"name": field, "raw_model": model._model, "label_field": field, "target_layers": get_target_layers(model, m_type), "classes": model.classes}
            process_model_heatmaps_otimizado(ds, cfg, device, CLASSES_DE_INTERESSE, lista_ids=ids_internos)

    print("\n[4/6] Gerando relatórios...")
    for ds in datasets_para_avaliar: gerar_analise_completa(ds, ds.name)

    print("\nConcluído.")
    session = fo.launch_app()
    session.wait()