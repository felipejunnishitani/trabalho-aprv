Visão geral

O arquivo main.py implementa um pipeline completo para avaliação de modelos de detecção de objetos em diferentes variações de um mesmo conjunto de imagens. O script:

Prepara datasets no FiftyOne a partir de imagens locais e anotações no formato COCO;

Aplica múltiplos modelos de detecção pré-treinados;

Avalia o desempenho dos modelos (mAP, precisão, recall, F1-score e IoU);

Gera gráficos comparativos e relatórios textuais;

Produz heatmaps de atenção (EigenCAM) para interpretação visual das detecções;

Abre a interface interativa do FiftyOne para inspeção dos resultados.

Funcionalidades principais

📂 Criação e gerenciamento automático de datasets no FiftyOne

🤖 Avaliação de múltiplos modelos de detecção

📊 Geração de gráficos estatísticos por dataset e modelo

🧠 Interpretação visual com Grad-CAM / EigenCAM

📝 Relatórios detalhados em texto

🖥️ Visualização interativa via FiftyOne App

Estrutura esperada de diretórios
project/
│
├── main.py
├── dataset_original/      # Imagens originais
├── dataset_neutro/        # Imagens modificadas (neutro)
├── dataset_atipico/       # Imagens modificadas (atípico)
│
├── grafico_original/      # Saídas do dataset original
├── grafico_neutro/        # Saídas do dataset neutro
├── grafico_atipico/       # Saídas do dataset atípico
│
├── heatmaps_COCO_Original/
├── heatmaps_Neutro/
├── heatmaps_Atipico/
└── coco_annotations/      # Anotações COCO baixadas automaticamente

As pastas de saída são criadas automaticamente, caso não existam.

Configurações principais

No início do arquivo main.py, encontram-se variáveis que podem ser ajustadas conforme o experimento:

PASTA_ORIGINAL_LOCAL = "dataset_original"


PASTAS_IMAGENS = {
    "Neutro": "dataset_neutro",
    "Atipico": "dataset_atipico",
}


PASTAS_SAIDA = {
    "COCO_Original": "grafico_original",
    "Neutro": "grafico_neutro",
    "Atipico": "grafico_atipico",
}


CLASSES_DE_INTERESSE = [
    "stop sign", "airplane", "skis",
    "tennis racket", "person",
    "cat", "banana", "cup"
]
Dependências

Recomenda-se o uso de um ambiente virtual.

Instalação via pip
pip install fiftyone torch torchvision matplotlib seaborn pandas numpy pillow requests pymongo pytorch-grad-cam

⚠️ Certifique-se de instalar uma versão do PyTorch compatível com sua GPU e CUDA, se aplicável.

Como executar

Organize as imagens nas pastas dataset_original, dataset_neutro e dataset_atipico.

Execute o script:

python main.py

Durante a execução:

As anotações COCO serão baixadas automaticamente, se necessário;

Os modelos serão carregados via FiftyOne Zoo;

As métricas, gráficos e heatmaps serão gerados;

A interface do FiftyOne será aberta ao final.

Resultados gerados

Para cada dataset analisado, o script gera:

📄 relatorio_detalhado.txt

📊 Gráficos:

grafico_confianca.png

grafico_iou_final.png

grafico_ap_classes_selecionadas.png

grafico_metricas_detalhadas.png

🔥 Heatmaps salvos em subpastas organizadas por modelo

Observações importantes

A execução pode ser demorada dependendo da quantidade de imagens e do hardware disponível.

Para execução em servidores sem interface gráfica, recomenda-se comentar as linhas finais responsáveis por abrir o FiftyOne App:

# session = fo.launch_app()
# session.wait()

Caso ocorram erros relacionados a nomes de arquivos, verifique se os nomes das imagens correspondem aos IDs do COCO.

Possíveis extensões

Exportação dos resultados em CSV ou JSON

Integração com pipelines de ML (ex.: MLflow)

Avaliação de modelos customizados

Análise estatística entre contextos (original × neutro × atípico)

Autores: Anne Mari Suenaga Sakai, Felipe Jun Nishitani e Lucas Pereira Goes
Contexto: Avaliação e interpretabilidade de modelos de detecção de objetos