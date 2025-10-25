import matplotlib
matplotlib.use('Agg') 

import fiftyone as fo
import fiftyone.zoo as foz
# REMOVIDO: import fiftyone.brain as fob 
from fiftyone import ViewField as L
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torchvision
import torchvision.transforms.functional as TF # Para pré-processamento
from torchvision import transforms # Para pré-processamento
import numpy as np
import cv2
from ultralytics import YOLO

# ==========================================================
# --- NOVOS IMPORTS para Grad-CAM Manual ---
# ==========================================================
try:
    from pytorch_grad_cam import GradCAM
    # Usaremos um target genérico, pode precisar de ajuste fino
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget 
    # Para visualização (opcional, mas útil para debug)
    from pytorch_grad_cam.utils.image import show_cam_on_image 
    # Para pegar a imagem como numpy
    from PIL import Image
    grad_cam_installed = True
except ImportError:
    print("\nAVISO: Biblioteca 'grad-cam' não instalada. Heatmaps não serão calculados.")
    print("Instale com: pip install grad-cam ttach\n")
    grad_cam_installed = False
# ==========================================================


print("Carregando dataset...") 
# Use shuffle=False para consistência
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples=100, # <<-- MANTIDO EM 10 PARA TESTE RÁPIDO. AUMENTE PARA 100 PARA RESULTADOS REAIS.
    label_types=["detections"],
    shuffle=False 
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

# --- Mapeamento de Classes COCO (String -> Int ID) ---
coco_classes = dataset.default_classes 
coco_label_to_id = {label: i for i, label in enumerate(coco_classes)}


# ==========================================================
# --- FUNÇÕES AUXILIARES E TRANSFORMAÇÕES para Grad-CAM ---
# ==========================================================
# Pré-processamento padrão para modelos TorchVision (Faster R-CNN, RetinaNet)
preprocess_tv = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Pré-processamento mais simples para YOLOv5 (pode precisar de ajuste)
def preprocess_yolo(img_np_bgr):
    img_rgb = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb / 255.0
    tensor = torch.from_numpy(img_normalized.transpose((2, 0, 1))).float().unsqueeze(0)
    return tensor

# Função para redimensionar heatmap para o tamanho da detecção
def resize_heatmap(heatmap, target_w, target_h):
    heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).float()
    resized_heatmap = torch.nn.functional.interpolate(
        heatmap_tensor,
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    )
    return resized_heatmap.squeeze().numpy()
# ==========================================================


### PLOT DO NÍVEL DE CONFIANÇA
# ... (bloco sem alterações) ...
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
# ... (bloco sem alterações) ...
print("Limpando avaliações antigas (se existirem)...")
model_fields = ["yolo_v5", "faster_rcnn", "retinanet"] 
for model_field in model_fields:
    eval_key = f"eval_{model_field}"
    if eval_key in dataset.list_evaluations():
        dataset.delete_evaluation(eval_key)

# --- ANÁLISE DE PERFORMANCE ---
# ... (bloco sem alterações) ...
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
models_to_evaluate = [
    ("yolo_v5", model_yolo_raw),
    ("faster_rcnn", model_faster_raw),
    ("retinanet", model_retinanet_raw)
]
for model_field, model_raw in models_to_evaluate:
    print(f"Avaliando {model_field}...")
    results = dataset.evaluate_detections(
        model_field, gt_field="ground_truth", eval_key=f"eval_{model_field}", method="open-images"
    )
    evaluation_results[model_field] = results

# --- RELATÓRIO DE PERFORMANCE ---
# ... (bloco sem alterações) ...
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
    print("\n--- Relatório detalhado (Precision/Recall/F1-Score/Support) ---")
    results.print_report()
    print("-------------------------------------------------")
    ap_dict = results.report() # Usando report() para AP
    if ap_dict: 
        for class_name, metrics_dict in ap_dict.items():
            if class_name == "mAP": continue 
            ap_score = None 
            if isinstance(metrics_dict, dict):
                ap_score = metrics_dict.get("ap") 
            elif isinstance(metrics_dict, (float, int)):
                 ap_score = metrics_dict
            if ap_score is None: continue
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


# --- Lista de Classes de Interesse ---
classes_de_interesse = [
    "traffic light", "stop sign", "remote", "fork", "clock",
    "person", "cat", "dog", "chair", "bicycle", "bird"
]

# --- BLOCO DE EXTRAÇÃO DE DADOS (P/R/F1) ---
# ... (bloco sem alterações) ...
print("Extraindo dados de Precision/Recall/F1 para classes de interesse...")
plot_metrics_data = [] 
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
                plot_metrics_data.append({
                    "Modelo": model_name, "Classe": class_name,
                    "Métrica": "Precision", "Valor": precision
                })
            if recall is not None:
                plot_metrics_data.append({
                    "Modelo": model_name, "Classe": class_name,
                    "Métrica": "Recall", "Valor": recall
                })
            if f1_score is not None:
                plot_metrics_data.append({
                    "Modelo": model_name, "Classe": class_name,
                    "Métrica": "F1-Score", "Valor": f1_score
                })


# --- GRÁFICO 1: AP PARA TODAS AS CLASSES ---
# ... (bloco sem alterações) ...
if plot_data_ap:
    print("Gerando gráfico de AP para todas as classes...")
    df_ap = pd.DataFrame(plot_data_ap)
    df_ap = df_ap.sort_values(by="Average Precision (AP)", ascending=False)
    num_classes = len(all_classes)
    chart_width = max(12, num_classes * 0.5)
    plt.figure(figsize=(chart_width, 7))
    sns.barplot(
        data=df_ap, x="Classe", y="Average Precision (AP)", hue="Modelo" 
    )
    plt.title("Comparação de Average Precision (AP) para todas as classes")
    plt.ylabel("Average Precision (AP)")
    plt.xlabel("Classe do Objeto")
    plt.xticks(rotation=90, ha="right") 
    plt.legend(title="Modelo", loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.tight_layout() 
    plt.savefig("grafico_ap_todas_as_classes.png")
    print("Gráfico 'grafico_ap_todas_as_classes.png' salvo.")
    plt.clf() 
else:
    print("Nenhum dado de AP por classe encontrado para plotar.")


# --- GRÁFICO 2: AP PARA CLASSES SELECIONADAS ---
# ... (bloco sem alterações) ...
if plot_data_ap:
    print(f"Gerando gráfico de AP para classes selecionadas...")
    df_ap_full = pd.DataFrame(plot_data_ap) 
    df_ap_filtered = df_ap_full[df_ap_full['Classe'].isin(classes_de_interesse)]
    if not df_ap_filtered.empty:
        df_ap_filtered = df_ap_filtered.sort_values(by="Average Precision (AP)", ascending=False)
        num_filtered_classes = len(df_ap_filtered['Classe'].unique())
        plt.figure(figsize=(max(10, num_filtered_classes * 1.2), 6)) 
        sns.barplot(
            data=df_ap_filtered, x="Classe", y="Average Precision (AP)", hue="Modelo" 
        )
        plt.title("Comparação de AP por Classe (Classes Selecionadas)")
        plt.ylabel("Average Precision (AP)")
        plt.xlabel("Classe do Objeto")
        plt.xticks(rotation=45, ha="right") 
        plt.legend(title="Modelo", loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6, axis='y')
        plt.tight_layout() 
        plt.savefig("grafico_ap_classes_selecionadas.png")
        print("Gráfico 'grafico_ap_classes_selecionadas.png' salvo.")
        plt.clf()
    else:
        print("Nenhuma das classes de interesse foi encontrada nos resultados.") 


# --- GRÁFICO 3: (Precision, Recall, F1) ---
# ... (bloco sem alterações) ...
if plot_metrics_data:
    print("Gerando gráfico de Métricas (Precision/Recall/F1)...")
    df_metrics = pd.DataFrame(plot_metrics_data)
    class_order = sorted([c for c in classes_de_interesse if c in df_metrics['Classe'].unique()]) 
    
    num_classes_plot = len(class_order)
    num_rows = -(-num_classes_plot // 4)
    
    g = sns.catplot(
        data=df_metrics,
        x="Métrica", y="Valor", hue="Modelo",
        col="Classe", col_order=class_order,
        kind="bar", col_wrap=4,
        height=3,
        aspect=1,
        legend=False
    )
    
    g.figure.suptitle("Comparação de Métricas (P, R, F1) por Classe e Modelo", y=1.03) 
    
    g.set_axis_labels("", "Valor da Métrica")
    g.set_titles("{col_name}") 
    g.set(ylim=(0, 1.1)) 
    
    g.add_legend(
        bbox_to_anchor=(1.02, 0.5), 
        loc='center left'          
    )
    
    g.figure.subplots_adjust(top=0.90, hspace=0.45, wspace=0.15, right=0.88) 
    
    plt.savefig("grafico_metricas_detalhadas.png", bbox_inches='tight') 
    print("Gráfico 'grafico_metricas_detalhadas.png' salvo.")
    plt.clf()
else:
    print("Nenhum dado de P/R/F1 encontrado para plotar.")

# ==========================================================
# --- BLOCO HEATMAPS (GRAD-CAM MANUAL) ---
# ==========================================================
print("\nCalculando heatmaps (Grad-CAM MANUALMENTE)... (Isso pode demorar)")

# Filtra o dataset (igual antes)
view = dataset.filter_labels("ground_truth", L("label").is_in(classes_de_interesse))
print(f"Calculando heatmaps para {len(view)} amostras relevantes...")

if len(view) > 0 and grad_cam_installed: # Adiciona checagem se grad-cam foi importado
    # Mapeamento de nome do modelo para a camada alvo (CHUTES!)
    target_layers_map = {
        "faster_rcnn": lambda m: [m.backbone.body.layer4[-1]],
        "retinanet": lambda m: [m.backbone.layer4[-1]],
        "yolo_v5": lambda m: [m.model.model[9].cv3.conv] # Exemplo MUITO específico, PODE FALHAR.
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    for model_field, model_raw in models_to_evaluate:
        print(f"Calculando Grad-CAM para {model_field}...")
        heatmap_field = f"gradcam_{model_field}"
        model_raw.to(device).eval()

        get_target_layers_func = target_layers_map.get(model_field)
        if not get_target_layers_func:
            print(f"Função para obter camada alvo não definida para {model_field}. Pulando.")
            continue

        try:
            target_layers = get_target_layers_func(model_raw)
            if not target_layers or target_layers[0] is None:
                 print(f"  > Camada alvo não encontrada para {model_field}. Pulando.")
                 continue
            print(f"  > Camada(s) alvo: {[layer.__class__.__name__ for layer in target_layers]}")

            cam = GradCAM(model=model_raw, target_layers=target_layers)

            for sample in view.select_fields(model_field).iter_samples(autosave=False, progress=True):
                img_bgr = cv2.imread(sample.filepath)
                if img_bgr is None: continue
                img_rgb_pil = Image.open(sample.filepath).convert('RGB')
                img_h, img_w = img_bgr.shape[:2]

                if model_field == "yolo_v5":
                    input_tensor = preprocess_yolo(img_bgr).to(device)
                else:
                    input_tensor = preprocess_tv(img_rgb_pil).unsqueeze(0).to(device)

                detections_to_process = sample[model_field].detections if sample[model_field] else []
                if not detections_to_process: continue

                processed_detections = []
                for detection in detections_to_process:
                    label_str = detection.label

                    # ==========================================================
                    # --- OTIMIZAÇÃO: Calcular CAM apenas para classes de interesse ---
                    # ==========================================================
                    if label_str not in classes_de_interesse:
                        processed_detections.append(detection) # Mantém a detecção, mas sem heatmap
                        continue # Pula para a próxima detecção
                    # ==========================================================

                    class_id = coco_label_to_id.get(label_str, -1)
                    if class_id == -1:
                        processed_detections.append(detection)
                        continue

                    targets = [ClassifierOutputTarget(class_id)]

                    try:
                        print(f"    -> Calculando CAM para detecção {detection.id} ({label_str})...") # DEBUG PRINT 1
                        grayscale_cam = cam(input_tensor=input_tensor,
                                            targets=targets,
                                            eigen_smooth=True,
                                            aug_smooth=True)

                        heatmap_np = grayscale_cam[0, :]
                        heatmap_np_to_save = heatmap_np

                        print(f"    -> Heatmap gerado! Shape: {heatmap_np_to_save.shape}, Min: {heatmap_np_to_save.min():.2f}, Max: {heatmap_np_to_save.max():.2f}") # DEBUG PRINT 2

                        fo_heatmap = fo.Heatmap(map=heatmap_np_to_save.astype(np.float32))
                        detection[heatmap_field] = fo_heatmap
                        processed_detections.append(detection)
                        print(f"    -> Heatmap ADICIONADO à detecção {detection.id}.") # DEBUG PRINT 3

                    except Exception as cam_err:
                        # DESCOMENTE PARA VER ERROS INTERNOS:
                        print(f"    -> ERRO no cálculo CAM para detecção {detection.id} ({label_str}): {cam_err}")
                        processed_detections.append(detection) # Adiciona mesmo se falhar

                # Atualiza apenas se houve detecções
                if sample[model_field]:
                    sample[model_field].detections = processed_detections
                
                print(f"  --> Salvando sample {sample.id}...") # DEBUG PRINT 4
                sample.save()
                print(f"  --> Sample salvo.") # DEBUG PRINT 5

        except Exception as e:
            print(f"ERRO GERAL ao calcular Grad-CAM para {model_field}: {e}")
            print("Verifique a definição da camada alvo e pré-processamento.")
            print("Pulando este modelo...")

    print("Cálculo de heatmaps (manual) concluído.")
    print("Atualizando a sessão do navegador para mostrar os heatmaps...")
    session.view = view # Garante que a sessão está mostrando a view correta

elif not grad_cam_installed:
    print("Biblioteca 'grad-cam' não encontrada. Pulando cálculo de heatmaps.")
else:
    print("Nenhuma amostra encontrada com as classes de interesse para calcular heatmaps.")
# --- FIM DO BLOCO HEATMAPS MANUAL ---


# --- PLOT IoU ---
# ... (bloco sem alterações) ...
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