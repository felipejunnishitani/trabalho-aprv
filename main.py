import matplotlib
matplotlib.use('Agg') 

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
from fiftyone import ViewField as L
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torchvision
from torchvision.transforms import ToTensor
import numpy as np
import cv2
from ultralytics import YOLO

import traceback
import os

# Imports específicos do Grad-CAM
from pytorch_grad_cam import EigenCAM 
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import torchvision.transforms as T

# Tamanho padrão para redimensionamento de imagens no YOLOv5
YOLO_IMG_SIZE = 640

# Certifica-se de que a pasta 'heatmaps' existe
if not os.path.exists("heatmaps"):
    os.makedirs("heatmaps")

print("--- CONFIGURAÇÃO INICIAL ---")
# --- DEFINIR DEVICE ---
print("Definindo device (cuda ou cpu)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando device: {device}")

print("Carregando dataset...") 
# Usando max_samples=5 para agilizar o teste
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples=5, 
    label_types=["detections"],
    shuffle=False
)

# interface visual do FiftyOne
print("Iniciando interface do FiftyOne...")
session = fo.launch_app(dataset)

print("---")
print("Abra esta URL no seu navegador para ver as imagens:")
print(session)
print("---")

# --- CARREGAMENTO DOS MODELOS ---
print("Carregando e aplicando YOLOv5...")
model_yolo_wrapper = foz.load_zoo_model("yolov5s-coco-torch")
model_yolo_raw = model_yolo_wrapper._model 
dataset.apply_model(model_yolo_wrapper, label_field="yolo_v5")

print("Carregando e aplicando Faster R-CNN...")
model_faster_wrapper = foz.load_zoo_model("faster-rcnn-resnet50-fpn-coco-torch")
model_faster_raw = model_faster_wrapper._model 
dataset.apply_model(model_faster_wrapper, label_field="faster_rcnn")

print("Carregando e aplicando RetinaNet...")
model_retinanet_wrapper = foz.load_zoo_model("retinanet-resnet50-fpn-coco-torch")
model_retinanet_raw = model_retinanet_wrapper._model 
dataset.apply_model(model_retinanet_wrapper, label_field="retinanet")

session.view = dataset.view()

# Mover modelos para o device
model_yolo_raw.to(device).eval()
model_faster_raw.to(device).eval()
model_retinanet_raw.to(device).eval()
print("Modelos movidos para o device e em modo .eval()")

# --- CONFIGURAÇÃO DE MODELOS PARA HEATMAPS ---
MODEL_CONFIGS = [
    {
        "name": "Faster R-CNN",
        "raw_model": model_faster_raw,
        "label_field": "faster_rcnn",
        "wrapper": model_faster_wrapper,
        "dir": "heatmaps/faster_rcnn",
        "target_layers": [model_faster_raw.backbone.body.layer4] 
    },
    {
        "name": "RetinaNet",
        "raw_model": model_retinanet_raw,
        "label_field": "retinanet",
        "wrapper": model_retinanet_wrapper,
        "dir": "heatmaps/retinanet",
        "target_layers": [model_retinanet_raw.backbone.body.layer4] 
    },
    {
        "name": "YOLOv5",
        "raw_model": model_yolo_raw,
        "label_field": "yolo_v5",
        "wrapper": model_yolo_wrapper,
        "dir": "heatmaps/yolov5",
        # CORREÇÃO mantida: Acessa a lista de módulos aninhada para o YOLOv5
        "target_layers": [model_yolo_raw.model.model[9]] 
    }
]

# Mantenha o YOLO_IMG_SIZE = 640 no topo do arquivo.

def process_model_heatmaps(model_config):
    """Gera heatmaps para todas as detecções de um modelo no dataset."""
    model_name = model_config["name"]
    raw_model = model_config["raw_model"]
    label_field = model_config["label_field"]
    wrapper = model_config["wrapper"]
    output_dir = model_config["dir"]
    target_layers = model_config["target_layers"]

    print(f"\n--- Iniciando Geração de Heatmaps para: {model_name} ---")
    
    # 1. Cria a pasta de saída
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Pasta criada: {output_dir}")

    # 2. Configura o EigenCAM (apenas uma vez por modelo)
    cam = EigenCAM(
        model=raw_model,
        target_layers=target_layers,
    )
    
    # 3. Itera sobre todas as amostras
    for i, sample in enumerate(dataset):
        if i >= 50: # Limite de 50 heatmaps para evitar execuções muito longas em datasets maiores
            break
        
        # 4. Verifica se a amostra tem detecções
        if sample[label_field] and len(sample[label_field].detections) > 0:
            try:
                # Pega a detecção com maior confiança
                detections = sorted(sample[label_field].detections, key=lambda x: x.confidence, reverse=True)
                target_detection = detections[0]
                
                print(f"  Processando amostra {i+1}. Alvo: {target_detection.label} ({target_detection.confidence:.2f})")
                
                image_path = sample.filepath
                img_pil = Image.open(image_path).convert("RGB")
                
                # Armazena o tamanho original
                w_img_orig, h_img_orig = img_pil.size 
                
                # --- LÓGICA DE REDIMENSIONAMENTO CONDICIONAL (CORREÇÃO) ---
                is_yolo = (model_name == "YOLOv5")

                if is_yolo:
                    # 1. Redimensiona a imagem PIL para o padrão YOLO
                    img_pil_resized = img_pil.resize((YOLO_IMG_SIZE, YOLO_IMG_SIZE))
                    
                    # 2. Converte a imagem REDIMENSIONADA para float para visualização
                    rgb_img_display = np.float32(img_pil_resized) / 255
                    
                    # 3. Cria o tensor de entrada a partir da imagem REDIMENSIONADA
                    transform = T.Compose([T.ToTensor()])
                    input_tensor = transform(img_pil_resized).to(device)
                    
                    # 4. ESCALA AS COORDENADAS DO TARGET para o novo tamanho (640x640)
                    rel_bbox = target_detection.bounding_box 
                    x1 = rel_bbox[0] * w_img_orig
                    y1 = rel_bbox[1] * h_img_orig
                    x2 = (rel_bbox[0] + rel_bbox[2]) * w_img_orig
                    y2 = (rel_bbox[1] + rel_bbox[3]) * h_img_orig
                    
                    x1_scaled = x1 * (YOLO_IMG_SIZE / w_img_orig)
                    y1_scaled = y1 * (YOLO_IMG_SIZE / h_img_orig)
                    x2_scaled = x2 * (YOLO_IMG_SIZE / w_img_orig)
                    y2_scaled = y2 * (YOLO_IMG_SIZE / h_img_orig)
                    
                    target_boxes_list = [[x1_scaled, y1_scaled, x2_scaled, y2_scaled]]

                else:
                    # Faster R-CNN e RetinaNet: Usam a imagem original
                    rgb_img_display = np.float32(img_pil) / 255
                    
                    transform = T.Compose([T.ToTensor()])
                    input_tensor = transform(img_pil).to(device)
                    
                    # Coordenadas originais em pixel
                    rel_bbox = target_detection.bounding_box 
                    x1 = rel_bbox[0] * w_img_orig
                    y1 = rel_bbox[1] * h_img_orig
                    x2 = (rel_bbox[0] + rel_bbox[2]) * w_img_orig
                    y2 = (rel_bbox[1] + rel_bbox[3]) * h_img_orig
                    
                    target_boxes_list = [[x1, y1, x2, y2]]
                
                target_boxes_tensor = torch.tensor(target_boxes_list, dtype=torch.float).to(device)
                
                # Mapeamento de Classes
                class_list = wrapper.classes
                if target_detection.label in class_list:
                     class_index = class_list.index(target_detection.label) + 1
                else:
                    class_index = 1 
                target_labels_tensor = torch.tensor([class_index], dtype=torch.long).to(device)

                # Configurar Alvo
                targets = [FasterRCNNBoxScoreTarget(
                    labels=target_labels_tensor, 
                    bounding_boxes=target_boxes_tensor 
                )]
                
                # Calcular Grad-CAM
                print("    Calculando EigenCAM...")
                grayscale_cam = cam(
                    input_tensor=input_tensor.unsqueeze(0),
                    targets=targets,
                )
                
                # Visualizar e Salvar
                # A CORREÇÃO é usar 'rgb_img_display', que está no mesmo tamanho do heatmap do CAM
                visualization = show_cam_on_image(rgb_img_display, grayscale_cam[0, :], use_rgb=True)
                
                heatmap_filename = os.path.join(output_dir, f"heatmap_{sample.id}.png")
                viz_pil = Image.fromarray(visualization)
                viz_pil.save(heatmap_filename)
                print(f"    SUCESSO: Heatmap salvo em '{heatmap_filename}'")

            except Exception as e:
                print(f"  ERRO CRÍTICO no Grad-CAM para amostra {sample.id} ({model_name}): {e}")
                traceback.print_exc()
        else:
            print(f"  Amostra {i+1} ignorada: Nenhuma detecção em '{label_field}'.")

# ==============================================================================
# --- EXECUÇÃO DOS HEATMAPS PARA TODOS OS MODELOS ---
# ==============================================================================
print("\n--- INICIANDO GERAÇÃO DE HEATMAPS PARA TODOS OS MODELOS ---")
for config in MODEL_CONFIGS:
    process_model_heatmaps(config)
print("\n--- Geração de Heatmaps concluída. ---")

# ------------------------------------------------------------------------------

### PLOT DO NÍVEL DE CONFIANÇA
print("\n--- GERANDO GRÁFICO DE CONFIANÇA ---")
# ... (Código do Gráfico de Confiança aqui) ...
confidence_scores = {
    "YOLOv5": [],
    "Faster R-CNN": [],
    "RetinaNet": [] 
}
for sample in dataset.view():
    if sample.yolo_v5:
        for detection in sample.yolo_v5.detections:
            confidence_scores["YOLOv5"].append(detection.confidence)
    if sample.faster_rcnn:
        for detection in sample.faster_rcnn.detections:
            confidence_scores["Faster R-CNN"].append(detection.confidence)
    if sample.retinanet:
        for detection in sample.retinanet.detections:
            confidence_scores["RetinaNet"].append(detection.confidence)

df_list = []
for model, scores in confidence_scores.items():
    for score in scores:
        df_list.append({"Modelo": model, "Confiança": score})

if df_list: 
    df = pd.DataFrame(df_list)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Modelo", y="Confiança", data=df)
    plt.title("Distribuição da Confiança por Modelo")
    plt.ylabel("Nível de Confiança")
    plt.xlabel("Modelo")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("grafico_confianca.png")
    print("Gráfico 'grafico_confianca.png' salvo.")
    plt.clf() 
else:
    print("Nenhum dado de confiança encontrado para plotar.")


# ------------------------------------------------------------------------------

# --- BLOCO DE LIMPEZA E AVALIAÇÃO ---
print("\n--- AVALIAÇÃO DE PERFORMANCE ---")
print("Limpando avaliações antigas...")
model_fields = ["yolo_v5", "faster_rcnn", "retinanet"] 
for model_field in model_fields:
    eval_key = f"eval_{model_field}"
    if eval_key in dataset.list_evaluations():
        dataset.delete_evaluation(eval_key)

evaluation_results = {} 
map_scores_summary = {} 
plot_data_ap = []       
all_classes = set()     
model_name_map = {
    "yolo_v5": "YOLOv5",
    "faster_rcnn": "Faster R-CNN",
    "retinanet": "RetinaNet"
}

# A avaliação deve ser feita usando o nome do campo no dataset, não o modelo raw
for model_field in ["yolo_v5", "faster_rcnn", "retinanet"]:
    print(f"Avaliando {model_field}...")
    results = dataset.evaluate_detections(
        model_field,
        gt_field="ground_truth", 
        eval_key=f"eval_{model_field}", 
        method="open-images"
    )
    evaluation_results[model_field] = results

# --- RELATÓRIO DE PERFORMANCE ---
print("\n--- RELATÓRIO DE PERFORMANCE (BASELINE) ---")
for model_field, results in evaluation_results.items():
    model_name = model_name_map.get(model_field, model_field) 
    print(f"\nResultados para: {model_name}")
    map_score = results.mAP() 
    if map_score is not None:
        print(f"mAP (método open-images, IoU=0.5): {map_score:.4f}") 
        map_scores_summary[model_name] = map_score
    else:
        print(f"mAP: N/A")
        map_scores_summary[model_name] = 0.0 
    
    # Extrair AP por classe
    ap_dict = results._classwise_AP 
    if ap_dict: 
        for class_name, ap_score in ap_dict.items():
            if class_name == "mAP": continue 
            if ap_score is None: continue
            plot_data_ap.append({
                "Modelo": model_name,
                "Classe": class_name,
                "Average Precision (AP)": ap_score
            })
            all_classes.add(class_name)

print("-------------------------------------------")
print("--- RESUMO DO mAP ---")
print(pd.Series(map_scores_summary).sort_values(ascending=False))
print("-------------------------------------------\n")

# --- EXTRAÇÃO DE DADOS (P/R/F1) ---
print("Extraindo dados de Precision/Recall/F1...")
plot_metrics_data = [] 

classes_de_interesse = [
    "traffic light", "stop sign", "remote", "fork", "clock",
    "person", "cat", "dog", "chair", "bicycle", "bird", "car", "bus"
]

for model_field, results in evaluation_results.items():
    model_name = model_name_map.get(model_field, model_field)
    report_dict = results.report()
    if not report_dict: continue
    for class_name, metrics_dict in report_dict.items():
        if class_name not in classes_de_interesse: continue
        if isinstance(metrics_dict, dict):
            precision = metrics_dict.get("precision")
            recall = metrics_dict.get("recall")
            f1_score = metrics_dict.get("f1-score")
            if precision is not None:
                plot_metrics_data.append({"Modelo": model_name, "Classe": class_name, "Métrica": "Precision", "Valor": precision})
            if recall is not None:
                plot_metrics_data.append({"Modelo": model_name, "Classe": class_name, "Métrica": "Recall", "Valor": recall})
            if f1_score is not None:
                plot_metrics_data.append({"Modelo": model_name, "Classe": class_name, "Métrica": "F1-Score", "Valor": f1_score})


# --- GRÁFICOS DE AP ---
if plot_data_ap:
    print("Gerando gráfico de AP para todas as classes...")
    df_ap = pd.DataFrame(plot_data_ap)
    df_ap = df_ap.sort_values(by="Average Precision (AP)", ascending=False)
    num_classes = len(all_classes)
    chart_width = max(12, num_classes * 0.5)
    plt.figure(figsize=(chart_width, 7))
    sns.barplot(data=df_ap, x="Classe", y="Average Precision (AP)", hue="Modelo")
    plt.title("Comparação de Average Precision (AP)")
    plt.xticks(rotation=90, ha="right") 
    plt.tight_layout() 
    plt.savefig("grafico_ap_todas_as_classes.png")
    plt.clf() 
    
    # Gráfico filtrado
    print("Gerando gráfico de AP para classes selecionadas...")
    df_ap_filtered = df_ap[df_ap['Classe'].isin(classes_de_interesse)]
    if not df_ap_filtered.empty:
        plt.figure(figsize=(12, 6)) 
        sns.barplot(data=df_ap_filtered, x="Classe", y="Average Precision (AP)", hue="Modelo")
        plt.title("Comparação de AP (Classes de Interesse)")
        plt.xticks(rotation=45, ha="right") 
        plt.tight_layout() 
        plt.savefig("grafico_ap_classes_selecionadas.png")
        plt.clf()

# --- GRÁFICO DE MÉTRICAS DETALHADAS ---
if plot_metrics_data:
    print("Gerando gráfico de Métricas Detalhadas...")
    df_metrics = pd.DataFrame(plot_metrics_data)
    class_order = sorted([c for c in classes_de_interesse if c in df_metrics['Classe'].unique()]) 
    
    g = sns.catplot(
        data=df_metrics,
        x="Métrica", y="Valor", hue="Modelo",
        col="Classe", col_order=class_order,
        kind="bar", col_wrap=4,
        height=3, aspect=1,
        legend=True
    )
    g.figure.suptitle("Comparação de Métricas por Classe e Modelo", y=1.02) 
    g.set(ylim=(0, 1.1)) 
    plt.savefig("grafico_metricas_detalhadas.png", bbox_inches='tight') 
    plt.clf()

# ------------------------------------------------------------------------------

# ==============================================================================
# --- RELATÓRIO DE IoU ---
# ==============================================================================
print("\n--- GERANDO GRÁFICO DE IoU (TP) ---")
iou_scores = {
    "YOLOv5": [],
    "Faster R-CNN": [],
    "RetinaNet": [] 
}

for sample in dataset.view():
    # Helper para extrair IoU
    def extract_iou(model_key, eval_key):
        detections = sample[model_key].detections if sample[model_key] else []
        tp_dets = [d for d in detections if eval_key in d and d[eval_key] == "tp"]
        scores = []
        iou_field = f"{eval_key}_iou"
        for d in tp_dets:
            if iou_field in d and d[iou_field] is not None:
                scores.append(d[iou_field])
        return scores

    iou_scores["YOLOv5"].extend(extract_iou("yolo_v5", "eval_yolo_v5"))
    iou_scores["Faster R-CNN"].extend(extract_iou("faster_rcnn", "eval_faster_rcnn"))
    iou_scores["RetinaNet"].extend(extract_iou("retinanet", "eval_retinanet"))

df_list_iou = []
for model, scores in iou_scores.items():
    for score in scores:
        df_list_iou.append({"Modelo": model, "IoU": score})

if df_list_iou:
    df_iou = pd.DataFrame(df_list_iou)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Modelo", y="IoU", data=df_iou)
    plt.title("Distribuição do IoU para True Positives")
    plt.ylabel("IoU")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("grafico_iou_final.png")
    print("Gráfico 'grafico_iou_final.png' salvo.")
else:
    print("Nenhum IoU encontrado (talvez nenhuma detecção correta nas amostras carregadas).")

print("\n--- SCRIPT CONCLUÍDO ---")
print("Pressione ENTER para encerrar.")
input()