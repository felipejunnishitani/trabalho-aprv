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
import numpy as np
import cv2
from ultralytics import YOLO

def main():
    print("Carregando dataset...") 
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        max_samples=1000, 
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
    models_to_evaluate = [
        ("yolo_v5", model_yolo_raw),
        ("faster_rcnn", model_faster_raw),
        ("retinanet", model_retinanet_raw)
    ]
    for model_field, model_raw in models_to_evaluate:
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
        print("\n--- Relatório detalhado (Precision/Recall/F1-Score/Support) ---")
        results.print_report()
        print("-------------------------------------------------")
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
        else:
            print(f"Nenhum AP por classe encontrado para {model_name}.")

    print("-------------------------------------------")
    print("--- RESUMO DO mAP ---")
    print(pd.Series(map_scores_summary).sort_values(ascending=False))
    print("-------------------------------------------\n")


    # --- EXTRAÇÃO DE DADOS (P/R/F1) ---
    print("Extraindo dados de Precision/Recall/F1 para classes de interesse...")
    plot_metrics_data = [] 

    classes_de_interesse = [
        # classes dependentes do contexto:
        "traffic light", "stop sign", "airplane", "skis", "tennis racket"

        # classes independentes do contexto:
        "person", "cat", "dog", "chair", "bicycle", "bird"
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
        
        # 1. Título principal
        g.figure.suptitle("Comparação de Métricas (P, R, F1) por Classe e Modelo", y=1.03) 
        
        # 2. Rótulos dos eixos e títulos dos subplots
        g.set_axis_labels("", "Valor da Métrica")
        g.set_titles("{col_name}") 
        g.set(ylim=(0, 1.1)) 
        
        # 3. Adicionar legenda à direita, SEM o título "Modelo"
        g.add_legend(
            # title="Modelo", # REMOVIDO ou title=None
            bbox_to_anchor=(1.02, 0.5), # Posição (X > 1 = direita, Y = 0.5 = centro vertical)
            loc='center left'          # Alinhamento da legenda em relação ao ponto de ancoragem
        )
        
        # 4. Ajustar espaçamento geral
        #    right=0.88 ou 0.90 -> Deixa um pouco mais de espaço para a legenda (sem título ocupa menos)
        g.figure.subplots_adjust(top=0.90, hspace=0.45, wspace=0.15, right=0.88) 
        
        # ==========================================================
        
        plt.savefig("grafico_metricas_detalhadas.png", bbox_inches='tight') 
        print("Gráfico 'grafico_metricas_detalhadas.png' salvo.")
        plt.clf()
    else:
        print("Nenhum dado de P/R/F1 encontrado para plotar.")

    ### CORRIGIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIR AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    # NAO AGUENTO MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIS
    # --- PLOT HEATMAPS (GRAD-CAM) ---
    print("\nCalculando heatmaps (Grad-CAM)... (Isso pode demorar)")

    # Filtra o dataset para conter APENAS amostras que tenham pelo menos um objeto das nossas classes de interesse.
    view = dataset.filter_labels("ground_truth", L("label").is_in(classes_de_interesse))
    print(f"Calculando heatmaps para {len(view)} amostras relevantes...")

    if len(view) > 0:
        for model_field, model_raw in models_to_evaluate:
            print(f"Calculando Grad-CAM para {model_field}...")
            
            patches_field = model_field 
            heatmap_field = f"gradcam_{model_field}" 

            try:
                # deu ruim -> n existe na versao 1.9 
                fob.compute_saliency_masks(
                    view,
                    model=model_raw,
                    patches_field=patches_field,
                    saliency_field=heatmap_field, #
                    method="grad-cam",
                    use_logits=True,
                )
                print(f"Heatmaps para {model_field} salvos no campo '{heatmap_field}'.")
            
            except Exception as e:
                print(f"ERRO ao calcular Grad-CAM para {model_field}: {e}")
                print("Alguns modelos (como YOLO) podem precisar de configuração manual da camada.")
                print("Pulando este modelo...")

        # Força a atualização da sessão no navegador
        print("Atualizando a sessão do navegador para mostrar os heatmaps...")
    else:
        print("Nenhuma amostra encontrada com as classes de interesse para calcular heatmaps.")
        
        
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


if __name__ == "__main__":
    main()
