## 🤖 Avaliação e Interpretabilidade de Modelos de Detecção de Objetos

Este repositório contém um pipeline completo implementado em `main.py` para avaliar o desempenho e a interpretabilidade visual de modelos pré-treinados de detecção de objetos em diferentes variações (original, neutro e atípico) de um mesmo conjunto de imagens.

-----

## 🌟 Visão Geral

O script **`main.py`** orquestra uma série de etapas cruciais para a avaliação robusta de modelos, utilizando a poderosa biblioteca **FiftyOne** para gerenciamento de dados e visualização.

### 🎯 Funcionalidades Principais

  * 📂 **Criação e Gerenciamento de Datasets:** Prepara e gerencia automaticamente datasets no FiftyOne a partir de imagens locais e anotações no formato COCO.
  * 🤖 **Avaliação Multimodelo:** Aplica e avalia múltiplos modelos de detecção pré-treinados (via FiftyOne Zoo).
  * 📊 **Geração de Métricas e Gráficos:** Calcula métricas de desempenho essenciais ($\text{mAP}$, Precisão, Recall, $\text{F1-score}$, $\text{IoU}$) e gera gráficos comparativos.
  * 🧠 **Interpretabilidade Visual (EigenCAM):** Produz mapas de calor (heatmaps) de atenção (via EigenCAM) para justificar visualmente as detecções e auxiliar na análise de falhas.
  * 📝 **Relatórios Detalhados:** Gera relatórios textuais e estatísticos por dataset e modelo.
  * 🖥️ **Visualização Interativa:** Abre a interface do FiftyOne App para inspeção visual e depuração dos resultados da avaliação.

-----

## ⚙️ Configuração do Projeto

### 📂 Estrutura Esperada de Diretórios

O script espera uma estrutura inicial de diretórios e cria automaticamente as pastas de saída.

```
project/
│
├── main.py
├── dataset_original/        # 🖼️ Imagens originais para o dataset base
├── dataset_neutro/          # 🖼️ Imagens modificadas (variação 'neutro')
├── dataset_atipico/         # 🖼️ Imagens modificadas (variação 'atípico')
│
├── grafico_original/        # 📈 Saídas (gráficos, relatórios) do dataset original
├── grafico_neutro/          # 📈 Saídas (gráficos, relatórios) do dataset neutro
├── grafico_atipico/         # 📈 Saídas (gráficos, relatórios) do dataset atípico
│
├── heatmaps_COCO_Original/  # 🔥 Heatmaps gerados (EigenCAM)
├── heatmaps_Neutro/         # 🔥 Heatmaps gerados (EigenCAM)
├── heatmaps_Atipico/        # 🔥 Heatmaps gerados (EigenCAM)
└── coco_annotations/        # 💾 Anotações COCO baixadas automaticamente
```

### 🛠️ Ajustes de Configuração

As variáveis no início do arquivo `main.py` podem ser ajustadas conforme a necessidade do seu experimento:

> ```python
> PASTA_ORIGINAL_LOCAL = "dataset_original"
> ```

> PASTAS\_IMAGENS = {
> "Neutro": "dataset\_neutro",
> "Atipico": "dataset\_atipico",
> }

> PASTAS\_SAIDA = {
> "COCO\_Original": "grafico\_original",
> "Neutro": "grafico\_neutro",
> "Atipico": "grafico\_atipico",
> }

> CLASSES\_DE\_INTERESSE = [
> "stop sign", "airplane", "skis",
> \# ... adicione ou remova classes conforme seu foco
> ]
>
> ```
> ```

-----

## 📥 Dependências

Recomenda-se fortemente o uso de um ambiente virtual (ex: `venv` ou `conda`).

### Instalação via pip

```bash
pip install fiftyone torch torchvision matplotlib seaborn pandas numpy pillow requests pymongo pytorch-grad-cam
```

> ⚠️ **Importante:** Certifique-se de instalar uma versão do **PyTorch** (`torch` e `torchvision`) compatível com sua **GPU e CUDA**, caso deseje aproveitar a aceleração de hardware.

-----

## 🚀 Como Executar

### 1\. Preparação

Organize suas imagens nas respectivas pastas de entrada: `dataset_original`, `dataset_neutro` e `dataset_atipico`.

### 2\. Execução

Execute o script diretamente:

```bash
python main.py
```

### Fluxo de Execução

1.  As anotações COCO (para as classes de interesse) serão baixadas automaticamente, se necessário.
2.  Os modelos de detecção serão carregados via FiftyOne Zoo.
3.  As métricas de desempenho, gráficos e heatmaps serão gerados e salvos nas pastas de saída.
4.  A interface do **FiftyOne App** será aberta ao final para exploração visual.

-----

## 📈 Resultados Gerados

Para cada dataset analisado, o script salva os seguintes arquivos na pasta de saída correspondente (ex: `grafico_original/`):

### 📄 Relatório Textual

  * `relatorio_detalhado.txt`

### 📊 Gráficos (PNG)

  * `grafico_confianca.png`
  * `grafico_iou_final.png`
  * `grafico_ap_classes_selecionadas.png`
  * `grafico_metricas_detalhadas.png`

### 🔥 Mapas de Calor (Heatmaps)

Subpastas contendo imagens com os heatmaps de atenção (EigenCAM) aplicados, organizadas por nome do modelo.

-----

## 📝 Observações

### Desempenho

A execução pode ser **demorada** dependendo da quantidade de imagens e do hardware disponível, especialmente durante a geração dos heatmaps (EigenCAM).

### Execução em Servidores (Headless)

Se estiver executando em um servidor **sem interface gráfica** (headless), **comente** as linhas finais no `main.py` que iniciam o FiftyOne App, para evitar erros:

```python
# session = fo.launch_app()
# session.wait()
```

### Erros de Anotação

Caso ocorram erros relacionados a nomes de arquivos ou IDs de anotações, verifique se os nomes dos arquivos de imagem locais correspondem aos IDs esperados no COCO.

-----

## ✒️ Autores

  * Anne Mari Suenaga Sakai
  * Felipe Jun Nishitani
  * Lucas Pereira Goes

**Contexto:** Avaliação e interpretabilidade de modelos de detecção de objetos.

-----

