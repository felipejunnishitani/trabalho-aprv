import matplotlib
matplotlib.use('Agg')

import fiftyone as fo
import fiftyone.zoo as foz
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Baixa 200 imagens do split de validação do COCO 2017
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples=200,
    label_types=["detections"],
    shuffle=True
)

# Inicia a interface visual do FiftyOne para explorar as imagens
session = fo.launch_app(dataset, remote=True)

### Carrega modelos pré-treinados
# Carrega o modelo YOLOv5 pré-treinado em COCO
model = foz.load_zoo_model("yolov5s-coco-torch")

# Adiciona as predições do YOLOv5 ao seu dataset
# O 'yolo_v5' é o nome do campo onde as detecções serão salvas
dataset.apply_model(model, label_field="yolo_v5")

# Carrega o modelo Faster R-CNN pré-treinado em COCO
#oo_model("faster-rcnn-resnet50-fpn-coco-torch")

# Adiciona as predições do Faster R-CNN
#dataset.apply_model(model, label_field="faster_rcnn")

# Carrega o modelo EfficientDet-D0 pré-treinado em COCO
#model = foz.load_zoo_model("efficientdet-d0-coco-tf1")

# Adiciona as predições do EfficientDet
#dataset.apply_model(model, label_field="efficientdet_d0")
### 

# Atualize a sessão para ver as novas detecções no visualizador
session.view = dataset.view()

### PLOT DO NÍVEL DE CONFIANÇA
# 1. Extraia as pontuações de confiança para cada modelo
confidence_scores = {
    "YOLOv5": [],
    "Faster R-CNN": [],
    #"EfficientDet": []
}

# Itera por cada amostra no dataset
for sample in dataset.view():
    if sample.yolo_v5:
        for detection in sample.yolo_v5.detections:
            confidence_scores["YOLOv5"].append(detection.confidence)
    if sample.faster_rcnn:
        for detection in sample.faster_rcnn.detections:
            confidence_scores["Faster R-CNN"].append(detection.confidence)
    #if sample.efficientdet_d0:
    #    for detection in sample.efficientdet_d0.detections:
    #        confidence_scores["EfficientDet"].append(detection.confidence)

# 2. Prepara os dados para o plot com Pandas
df_list = []
for model, scores in confidence_scores.items():
    for score in scores:
        df_list.append({"Modelo": model, "Confiança": score})

df = pd.DataFrame(df_list)

# 3. Cria e exibe o gráfico (boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(x="Modelo", y="Confiança", data=df)
plt.title("Distribuição da Confiança por Modelo")
plt.ylabel("Nível de Confiança")
plt.xlabel("Modelo")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("grafico_confianca.png")
plt.show()

session.wait()
