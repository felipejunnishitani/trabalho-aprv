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

PASTA_ORIGINAL_LOCAL = "dataset_original"

PASTAS_IMAGENS = {
    "Neutro": "dataset_neutro",
    "Atipico": "dataset_atipico"
}

PASTAS_SAIDA = {
    "COCO_Original": "grafico_original",
    "Neutro": "grafico_neutro",
    "Atipico": "grafico_atipico"
}

CLASSES_DE_INTERESSE = [
    "stop sign", "airplane", "skis", "tennis racket",
    "person", "cat", "banana", "cup"
]

# Criação das pastas de saída
for pasta in PASTAS_SAIDA.values():
    if not os.path.exists(pasta): os.makedirs(pasta)

def limpar_datasets_antigos():
    print("Verificando datasets antigos...")
    for nome in ["COCO_Original", "Neutro", "Atipico"]:
        if nome in fo.list_datasets():
            print(f"   -> Deletando {nome}...")
            fo.delete_dataset(nome)

def baixar_json_coco():
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    pasta_anotacoes = "coco_annotations"
    arquivo_json = os.path.join(pasta_anotacoes, "annotations", "instances_val2017.json")
    
    if os.path.exists(arquivo_json):
        return arquivo_json
    
    print("   -> Baixando anotações do COCO (~240MB)...")
    if not os.path.exists(pasta_anotacoes): os.makedirs(pasta_anotacoes)
    caminho_zip = os.path.join(pasta_anotacoes, "annotations.zip")
    
    try:
        with requests.get(url, stream=True) as r:
            with open(caminho_zip, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
            zip_ref.extractall(pasta_anotacoes)
    finally:
        if os.path.exists(caminho_zip): os.remove(caminho_zip)
    
    return arquivo_json

def criar_json_filtrado(json_path, pasta_imagens):
    print(f"   -> Filtrando JSON para a pasta: {pasta_imagens}")
    
    arquivos_locais = os.listdir(pasta_imagens)
    print(f"      Arquivos locais encontrados: {len(arquivos_locais)}")

    mapa_locais = {}
    for f in arquivos_locais:
        nome_sem_ext = os.path.splitext(f)[0]
        nome_sem_zeros = nome_sem_ext.lstrip("0")
        if not nome_sem_zeros: nome_sem_zeros = "0"

        mapa_locais[f] = f
        mapa_locais[nome_sem_ext] = f
        mapa_locais[nome_sem_zeros] = f

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    images_mantidas = []
    ids_mantidos = set()
    count_matches = 0
    
    for img in data['images']:
        nome_json = img['file_name']
        nome_json_sem_ext = os.path.splitext(nome_json)[0]
        nome_json_sem_zeros = nome_json_sem_ext.lstrip("0")
        if not nome_json_sem_zeros: nome_json_sem_zeros = "0"

        nome_real = None
        if nome_json in mapa_locais:
            nome_real = mapa_locais[nome_json]
        elif nome_json_sem_ext in mapa_locais:
            nome_real = mapa_locais[nome_json_sem_ext]
        elif nome_json_sem_zeros in mapa_locais:
            nome_real = mapa_locais[nome_json_sem_zeros]
            
        if nome_real:
            img['file_name'] = nome_real
            images_mantidas.append(img)
            ids_mantidos.add(img['id'])
            count_matches += 1
            
    print(f"      Correspondências encontradas: {count_matches}")
    
    if count_matches == 0:
        print("      !!! ERRO CRÍTICO: Nenhuma imagem do JSON bateu com a pasta local. !!!")

    annotations_mantidas = [
        ann for ann in data['annotations']
        if ann['image_id'] in ids_mantidos
    ]
    
    novo_data = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": data.get("categories", []),
        "images": images_mantidas,
        "annotations": annotations_mantidas
    }
    
    novo_json_path = os.path.join("coco_annotations", "instances_filtered.json")
    with open(novo_json_path, 'w') as f:
        json.dump(novo_data, f)
        
    print(f"   -> JSON Filtrado: {len(images_mantidas)} imagens | {len(annotations_mantidas)} anotações")
    return novo_json_path

def carregar_dataset_modificado(nome_dataset, pasta_imagens, mapa_original):
    if not os.path.exists(pasta_imagens) or not os.listdir(pasta_imagens):
        print(f"   AVISO: Pasta {pasta_imagens} vazia.")
        return None

    print(f"\n--- Carregando {nome_dataset} ---")
    
    if nome_dataset in fo.list_datasets():
        ds = fo.load_dataset(nome_dataset)
        if "ground_truth" not in ds.get_field_schema():
            print(f"   Dataset {nome_dataset} existe mas está corrompido. Recriando...")
            fo.delete_dataset(nome_dataset)
            ds = fo.Dataset.from_dir(pasta_imagens, dataset_type=fo.types.ImageDirectory, name=nome_dataset)
    else:
        ds = fo.Dataset.from_dir(
            dataset_dir=pasta_imagens,
            dataset_type=fo.types.ImageDirectory,
            name=nome_dataset
        )
    
    match_count = 0
    print("   -> Sincronizando Ground Truth...")
    
    for sample in ds:
        nome_atual = os.path.basename(sample.filepath)
        nome_limpo = nome_atual.replace("neutro_", "").replace("atipico_", "").replace("mod_", "")
        stem_busca = os.path.splitext(nome_limpo)[0]
        
        amostra_ref = None
        if stem_busca in mapa_original:
            amostra_ref = mapa_original[stem_busca]
        
        if amostra_ref:
            gt = getattr(amostra_ref, "ground_truth", None)
            
            if gt:
                sample["ground_truth"] = gt.copy()
                sample.save()
                match_count += 1
            
    print(f"   -> {match_count} anotações copiadas com sucesso.")
    return ds

def gerar_analise_completa(dataset, nome_contexto):
    pasta_destino = PASTAS_SAIDA.get(nome_contexto, f"grafico_{nome_contexto}")
    print(f"   -> Gerando gráficos em: {pasta_destino}")

    dados_confianca, dados_iou = [], []
    mapa_modelos = {"yolo_v5": "YOLOv5", "faster_rcnn": "Faster R-CNN", "retinanet": "RetinaNet"}

    for sample in dataset:
        for campo, nome_modelo in mapa_modelos.items():
            if campo in sample:
                preds = sample[campo]
                if not preds or not preds.detections: continue
                
                eval_key = f"eval_{campo}"
                
                for det in preds.detections:
                    if det.confidence:
                        dados_confianca.append({"Modelo": nome_modelo, "Confiança": det.confidence})
                    
                    if det.has_field(eval_key) and det[eval_key] == "tp":
                        iou_key = f"{eval_key}_iou"
                        val_iou = getattr(det, iou_key, None)
                        
                        if val_iou is None:
                            val_iou = getattr(det, "iou", None)

                        if val_iou is not None:
                            dados_iou.append({"Modelo": nome_modelo, "IoU": val_iou})

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
    
    dados_ap, dados_met = [], []
    texto_relatorio = f"=== RELATÓRIO: {nome_contexto.upper()} ===\n\n"

    for campo, nome_modelo in mapa_modelos.items():
        eval_key = f"eval_{campo}"
        if eval_key in dataset.list_evaluations():
            try:
                results = dataset.load_evaluation_results(eval_key)
                if not results: continue

                rep = results.report()
                if not isinstance(rep, dict):
                    texto_relatorio += f"--- {nome_modelo} ---\n{str(rep)}\n\n"
                else:
                    texto_relatorio += f"--- {nome_modelo} ---\n{json.dumps(rep, indent=4)}\n\n"
                    
                    for cls in CLASSES_DE_INTERESSE:
                        if cls in rep:
                            for k in ["precision", "recall", "f1-score"]:
                                if k in rep[cls]:
                                    dados_met.append({
                                        "Modelo": nome_modelo, 
                                        "Classe": cls, 
                                        "Métrica": k.capitalize(), 
                                        "Valor": rep[cls][k]
                                    })
                
                if hasattr(results, '_classwise_AP'):
                    ap_data = results._classwise_AP
                    
                    if isinstance(ap_data, dict):
                        for cls, ap in ap_data.items():
                            if cls in CLASSES_DE_INTERESSE:
                                dados_ap.append({"Modelo": nome_modelo, "Classe": cls, "AP": ap})
                                
                    elif isinstance(ap_data, (np.ndarray, list)):
                        nomes_classes = getattr(results, 'classes', [])
                        
                        if not nomes_classes and hasattr(dataset, 'classes'):
                             nomes_classes = dataset.classes
                        
                        if len(nomes_classes) == len(ap_data):
                            for cls, ap in zip(nomes_classes, ap_data):
                                if cls in CLASSES_DE_INTERESSE:
                                    dados_ap.append({"Modelo": nome_modelo, "Classe": cls, "AP": ap})
            
            except Exception as e:
                print(f"Erro ao processar métricas de {nome_modelo}: {e}")
                traceback.print_exc()
        else:
            texto_relatorio += f"--- {nome_modelo} ---\n(Sem avaliação)\n\n"

    with open(os.path.join(pasta_destino, "relatorio_detalhado.txt"), "w") as f: f.write(texto_relatorio)

    if dados_ap:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=pd.DataFrame(dados_ap), x="Classe", y="AP", hue="Modelo", palette="viridis")
        plt.title(f"AP por Classe - {nome_contexto}"); plt.ylim(0, 1.1)
        plt.savefig(os.path.join(pasta_destino, "grafico_ap_classes_selecionadas.png")); plt.close()

    if dados_met:
        try:
            g = sns.catplot(data=pd.DataFrame(dados_met), x="Métrica", y="Valor", hue="Modelo", col="Classe", col_wrap=4, kind="bar", height=3, aspect=1, palette="muted")
            g.fig.subplots_adjust(top=0.9); g.fig.suptitle(f"Métricas Detalhadas - {nome_contexto}")
            plt.savefig(os.path.join(pasta_destino, "grafico_metricas_detalhadas.png")); plt.close()
        except: pass

class YoloModelWrapper(torch.nn.Module):
    def __init__(self, ultralytics_model):
        super().__init__()
        self.model = ultralytics_model.model
        
    def forward(self, x):
        result = self.model(x)
        if isinstance(result, tuple):
            return result[0]
        return result

def get_target_layers(model_wrapper, model_type):
    raw = model_wrapper._model
    
    if "yolo" in model_type:
        try:
            return [raw.model.model[9]]
        except (AttributeError, IndexError):
            try:
                return [raw.model.model[-2]]
            except:
                 return []
            
    elif "faster" in model_type or "retina" in model_type:
        return [raw.backbone.body.layer4[-1]]
        
    return []

def process_model_heatmaps(dataset, model_config, device, classes_de_interesse, lista_filenames=None):
    model_name = model_config["name"]
    raw_model = model_config["raw_model"]
    label_field = model_config["label_field"]
    target_layers = model_config["target_layers"]
    class_list = model_config["classes"]
    
    YOLO_IMG_SIZE = 640
    
    base_output_dir = f"heatmaps_{dataset.name}"
    output_dir = os.path.join(base_output_dir, model_name)

    print(f"\n   -> Gerando Heatmaps para {model_name} em {dataset.name}...")
    
    view_para_processar = dataset
    
    if lista_filenames is not None and len(lista_filenames) > 0:
        
        ids_encontrados = []
        
        for sample in dataset.select_fields(["filepath"]).iter_samples():
            nome_arquivo = os.path.basename(sample.filepath)
            if nome_arquivo in lista_filenames:
                ids_encontrados.append(sample.id)
        
        if len(ids_encontrados) > 0:
            view_para_processar = dataset.select(ids_encontrados)
        else:
            view_para_processar = dataset.select([]) 
            
    else:
        print("      Filtro padrão: Processando primeiras 20 amostras.")
        view_para_processar = dataset.limit(20)

    if len(view_para_processar) == 0:
        print("      AVISO: Nenhuma imagem encontrada com os nomes fornecidos neste dataset.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not target_layers:
        print(f"      AVISO: Camadas alvo não encontradas para {model_name}. Pulando.")
        return
        
    is_yolo = "yolo" in model_name.lower()
    if is_yolo:
        model_para_cam = YoloModelWrapper(raw_model)
    else:
        model_para_cam = raw_model
        model_para_cam.eval()

    try:
        cam = EigenCAM(model=model_para_cam, target_layers=target_layers)
    except Exception as e:
        print(f"      ERRO ao inicializar EigenCAM: {e}")
        return

    transform = T.Compose([T.ToTensor()])
    
    for i, sample in enumerate(view_para_processar):
        if not sample.has_field(label_field) or not sample[label_field] or not sample[label_field].detections:
            continue
        
        try:
            detections_filtradas = [
                det for det in sample[label_field].detections 
                if det.label in classes_de_interesse
            ]
            
            if not detections_filtradas:
                continue

            target_detection = max(
                detections_filtradas, 
                key=lambda x: x.confidence if x.confidence is not None else -1
            )
            
            image_path = sample.filepath
            img_pil = Image.open(image_path).convert("RGB")
            w_img_orig, h_img_orig = img_pil.size
            
            targets = None
            
            rel_bbox = target_detection.bounding_box
            x1_orig = rel_bbox[0] * w_img_orig
            y1_orig = rel_bbox[1] * h_img_orig
            x2_orig = (rel_bbox[0] + rel_bbox[2]) * w_img_orig
            y2_orig = (rel_bbox[1] + rel_bbox[3]) * h_img_orig
            
            target_boxes_list = [[x1_orig, y1_orig, x2_orig, y2_orig]]
            target_boxes_tensor = torch.tensor(target_boxes_list, dtype=torch.float).to(device)
            
            class_index = class_list.index(target_detection.label) if target_detection.label in class_list else 1
            target_labels_tensor = torch.tensor([class_index], dtype=torch.long).to(device)

            targets = [FasterRCNNBoxScoreTarget(
                labels=target_labels_tensor,
                bounding_boxes=target_boxes_tensor
            )]
            
            if is_yolo:
                img_pil_resized = img_pil.resize((YOLO_IMG_SIZE, YOLO_IMG_SIZE))
                rgb_img_display = np.array(img_pil_resized, dtype=np.float32) / 255.0 
                input_tensor = transform(img_pil_resized).to(device)
            else:
                rgb_img_display = np.array(img_pil, dtype=np.float32) / 255.0
                input_tensor = transform(img_pil).to(device)
                
            grayscale_cam = cam(
                input_tensor=input_tensor.unsqueeze(0),
                targets=targets,
            )
            
            visualization = show_cam_on_image(rgb_img_display, grayscale_cam[0, :], use_rgb=True)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            heatmap_filename = os.path.join(output_dir, f"{dataset.name}_{base_name}_{target_detection.label}.png")
            Image.fromarray(visualization).save(heatmap_filename)

        except Exception as e:
            print(f"      Erro na amostra {sample.id} ({model_name}): {e}")
            pass

if __name__ == "__main__":
    multiprocessing.freeze_support()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- CONFIGURAÇÃO: Usando device {device} ---")
    
    limpar_datasets_antigos()

    print("\n[1/6] Montando Dataset Original...")
    
    json_full = baixar_json_coco()
    if not json_full: exit()

    json_filtrado = criar_json_filtrado(json_full, PASTA_ORIGINAL_LOCAL)

    print("   -> Importando dataset COCO...")
    try:
        dataset_original = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=PASTA_ORIGINAL_LOCAL,
            labels_path=json_filtrado,
            name="COCO_Original",
            include_id=True
        )
        
        schema = dataset_original.get_field_schema()
        if "detections" in schema:
            dataset_original.rename_sample_field("detections", "ground_truth")
        elif "ground_truth" not in schema:
            dataset_original.add_sample_field("ground_truth", fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections)

    except Exception as e:
        print(f"   ERRO FATAL ao carregar dataset: {e}")
        exit()

    dataset_original.persistent = True
    dataset_original.save()
    
    mapa_original = {os.path.splitext(os.path.basename(s.filepath))[0]: s for s in dataset_original}

    datasets_para_avaliar = [dataset_original]
    for nome, pasta in PASTAS_IMAGENS.items():
        ds_mod = carregar_dataset_modificado(nome, pasta, mapa_original)
        if ds_mod:
            ds_mod.persistent = True
            ds_mod.save()
            datasets_para_avaliar.append(ds_mod)

    print("\n[2/6] Carregando Modelos...")
    yolo = foz.load_zoo_model("yolov5s-coco-torch")
    faster = foz.load_zoo_model("faster-rcnn-resnet50-fpn-coco-torch")
    retina = foz.load_zoo_model("retinanet-resnet50-fpn-coco-torch")

    IDS_BASE_NUMERICOS = [724, 785, 1532, 1675, 1761, 2473, 2532, 2587, 2592, 4495, 5060, 5477, 5529, 10363, 13348, 15440, 15497, 22396, 106384]

    print("\n[3/6] Avaliando e Gerando Heatmaps...")
    
    for ds in datasets_para_avaliar:
        print(f"\n>>> Dataset: {ds.name} ({len(ds)} imgs) <<<")
        if len(ds) == 0: continue
        
        filenames_para_filtro = []
        prefixo = ""
        
        if ds.name == "Neutro":
            prefixo = "neutro_"
        elif ds.name == "Atipico":
            prefixo = "atipico_"
        
        for id_num in IDS_BASE_NUMERICOS:
            nome_arquivo = f"{prefixo}{id_num:012}.jpg" 
            filenames_para_filtro.append(nome_arquivo)
            

        configs_heatmap = []
        
        model_setups = [
            (yolo, "yolo_v5", "yolo"),
            (faster, "faster_rcnn", "faster"),
            (retina, "retinanet", "retina")
        ]

        for model_obj, label_field, type_str in model_setups:
            print(f"   -> Aplicando {label_field}...")
            try:
                ds.apply_model(model_obj, label_field=label_field)
            except Exception as e:
                print(f"Erro ao aplicar {label_field}: {e}")

            if label_field in ds.get_field_schema():
                print(f"   -> Avaliando {label_field}...")
                try:
                    ds.evaluate_detections(
                        label_field,
                        gt_field="ground_truth",
                        eval_key=f"eval_{label_field}",
                        classes=CLASSES_DE_INTERESSE,
                        compute_mAP=True
                    )
                except Exception as e: print(f"Erro eval: {e}")

                try:
                    cfg = {
                        "name": label_field,
                        "raw_model": model_obj._model,
                        "label_field": label_field,
                        "target_layers": get_target_layers(model_obj, type_str),
                        "classes": model_obj.classes
                    }
                    configs_heatmap.append(cfg)
                except Exception as e:
                    print(f"Erro ao preparar config heatmap: {e}")

        print(f"   -> Gerando Heatmaps para dataset: {ds.name}...")
        for config in configs_heatmap:
            process_model_heatmaps(
                ds, 
                config, 
                device, 
                CLASSES_DE_INTERESSE, 
                lista_filenames=filenames_para_filtro 
            )

    print("\n[4/6] Gerando relatórios...")
    for ds in datasets_para_avaliar:
        if any(f"eval_{m}" in ds.list_evaluations() for m in ["yolo_v5", "faster_rcnn", "retinanet"]):
            gerar_analise_completa(ds, ds.name)
        else:
            print(f"   -> Sem dados para {ds.name}")

    print("\n[5/6] Concluído! Abrindo App...")
    session = fo.launch_app()
    print("Abra o navegador se não abrir automaticamente.")
    session.wait()
        
    print("\n[6/6] Fim.")