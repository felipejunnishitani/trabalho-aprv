import matplotlib
matplotlib.use('Agg') 

import fiftyone as fo
import fiftyone.zoo as foz
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torchvision
import numpy as np
import cv2
from ultralytics import YOLO

print("Carregando dataset...") 
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples=100, 
    label_types=["detections"],
    shuffle=True
)

# interface visual do FiftyOne para explorar as imagens
print("Iniciando interface do FiftyOne...")
session = fo.launch_app(dataset)

# informações da sessão
print("---")
print("Abra esta URL no seu navegador:")
print(session)
print("---")

# --- CARREGAMENTO DOS MODELOS ---
print("Carregando e aplicando YOLOv5...")
model_yolo = foz.load_zoo_model("yolov5s-coco-torch")
dataset.apply_model(model_yolo, label_field="yolo_v5")

print("Carregando e aplicando Faster R-CNN...")
model_faster = foz.load_zoo_model("faster-rcnn-resnet50-fpn-coco-torch")
dataset.apply_model(model_faster, label_field="faster_rcnn")

print("Carregando e aplicando RetinaNet...")
model_retinanet = foz.load_zoo_model("retinanet-resnet50-fpn-coco-torch")
dataset.apply_model(model_retinanet, label_field="retinanet")

session.view = dataset.view()

### PLOT DO NÍVEL DE CONFIANÇA
print("Gerando gráfico de confiança...")
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


# --- BLOCO DE LIMPEZA ---
print("Limpando avaliações antigas (se existirem)...")
model_fields = ["yolo_v5", "faster_rcnn", "retinanet"] 
for model_field in model_fields:
    eval_key = f"eval_{model_field}"
    if eval_key in dataset.list_evaluations():
        dataset.delete_evaluation(eval_key)

# --- ANÁLISE DE PERFORMANCE ---
print("Executando avaliação de performance (mAP e IoU)...")
evaluation_results = {} 
map_scores_summary = {} 
plot_data_ap = []       
all_classes = set()     
model_name_map = {
    "yolo_v5": "YOLOv5",
    "faster_rcnn": "Faster R-CNN",
    "retinanet": "RetinaNet"
}
for model_field in model_fields:
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
        print(f"mAP (método open-images, IoU=0.5): N/A (cálculo falhou)")
        map_scores_summary[model_name] = 0.0 
    
    # imprime o relatório(precision/recall/f1/support)
    print("\n--- Relatório detalhado (Precision/Recall/F1-Score/Support) ---")
    results.print_report()
    print("-------------------------------------------------")
    
    ap_dict = results._classwise_AP 
    
    if ap_dict: 
        for class_name, ap_score in ap_dict.items():
            
            if class_name == "mAP":
                continue 
            
            if ap_score is None:
                continue

            plot_data_ap.append({
                "Modelo": model_name,
                "Classe": class_name,
                "Average Precision (AP)": ap_score
            })
            all_classes.add(class_name)
    else:
        print(f"Nenhum AP por classe encontrado para {model_name}.")

print("-------------------------------------------")
print("--- RESUMO DO mAP ---")
print(pd.Series(map_scores_summary).sort_values(ascending=False))
print("-------------------------------------------\n")


# --- GRÁFICO DE AP POR CLASSE ---
if plot_data_ap:
    print("Gerando gráfico de AP por classe...")
    df_ap = pd.DataFrame(plot_data_ap)
    df_ap = df_ap.sort_values(by="Average Precision (AP)", ascending=False)
    num_classes = len(all_classes)
    plt.figure(figsize=(max(12, num_classes * 0.8), 7)) 
    sns.barplot(
        data=df_ap,
        x="Classe",
        y="Average Precision (AP)",
        hue="Modelo" 
    )
    plt.title("Comparação de Average Precision (AP) por Classe (mAP@0.5)")
    plt.ylabel("Average Precision (AP)")
    plt.xlabel("Classe do Objeto")
    plt.xticks(rotation=45, ha="right") 
    plt.legend(title="Modelo", loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.tight_layout() 
    plt.savefig("grafico_ap_por_classe.png")
    print("Gráfico 'grafico_ap_por_classe.png' salvo.")
    plt.clf() 
else:
    print("Nenhum dado de AP por classe encontrado para plotar.")

    
# --- PLOT IoU ---
print("Extraindo IoUs dos True Positives...")
iou_scores = {
    "YOLOv5": [],
    "Faster R-CNN": [],
    "RetinaNet": [] 
}
for sample in dataset.view():
    # YOLO
    if sample.yolo_v5:
        eval_key = "eval_yolo_v5"
        iou_key = eval_key + "_iou" 
        tp_detections = [d for d in sample.yolo_v5.detections if eval_key in d and d[eval_key] == "tp"]
        for d in tp_detections:
            if iou_key in d:
                iou_val = d[iou_key]
                if iou_val is not None:
                    iou_scores["YOLOv5"].append(iou_val) 

    # Faster R-CNN
    if sample.faster_rcnn:
        eval_key = "eval_faster_rcnn"
        iou_key = eval_key + "_iou" 
        tp_detections = [d for d in sample.faster_rcnn.detections if eval_key in d and d[eval_key] == "tp"]
        for d in tp_detections:
            if iou_key in d:
                iou_val = d[iou_key]
                if iou_val is not None:
                    iou_scores["Faster R-CNN"].append(iou_val)

    # RetinaNet
    if sample.retinanet:
        eval_key = "eval_retinanet"
        iou_key = eval_key + "_iou" 
        tp_detections = [d for d in sample.retinanet.detections if eval_key in d and d[eval_key] == "tp"]
        for d in tp_detections:
            if iou_key in d:
                iou_val = d[iou_key]
                if iou_val is not None:
                    iou_scores["RetinaNet"].append(iou_val)
df_list_iou = []
for model, scores in iou_scores.items():
    for score in scores:
        df_list_iou.append({"Modelo": model, "IoU": score})
if df_list_iou:
    df_iou = pd.DataFrame(df_list_iou)
    print("Salvando gráfico de IoU...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Modelo", y="IoU", data=df_iou)
    plt.title("Distribuição do IoU para Detecções Corretas (True Positives)")
    plt.ylabel("Intersection over Union (IoU)")
    plt.xlabel("Modelo")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("grafico_iou_baseline.png")
    plt.clf() 
    print("Gráfico 'grafico_iou_baseline.png' salvo.")
else:
    print("Nenhuma detecção 'True Positive' encontrada para gerar gráfico de IoU. (Aumente max_samples)")


# --- MANTER SESSÃO ABERTA ---
print("\nScript concluído. Sessão do FiftyOne está ativa no navegador.")
print("---")
print("A SESSÃO ESTÁ ATIVA. Pressione ENTER neste terminal para encerrar o script.")
print("---")
input()