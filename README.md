# unisul-machine-learning

Coleção de scripts utilizados para a matéria Aprendizado de Máquina na UNISUL.

## 📜 Scripts

### Algoritmo Classificador Supervisionado

Treina o reconhecimento dos personagens Bart e Homer de Os Simpsons utilizando um algoritmo classificador simples de Aprendizado de Máquina Supervisionado.

O script gera um arquivo no formato ARFF para ser usado no aplicativo WEKA.

**Executar:**

```sh
python exercicies/supervised_classifier_bart_homer.py
```

### Avaliação 1

**Enunciado:**

-   Desenvolva um programa para realizar extração de características de imagens (conforme apresentado em aula).
-   O programa deve ser capaz de analisar um dataset de imagens e criar um arquivo **\*.arff** com as características de todas as imagens contidas no dataset.
-   Para geração do arquivo e demais etapas do trabalho, você deverá utilizar o dataset dos **[personagens de Os Simpsons](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset)**.
-   Escolha dois personagens distintos deste dataset para utilizar em seu trabalho. Mantenha um diretório apenas com os personagens escolhidos por você.
-   As características que serão extraídas de cada personagem devem ser definidas por você.
-   O programa deve possibilitar selecionar uma imagem qualquer, e exibir as características da imagem selecionada.
-   O programa deve permitir que seja selecionada uma imagem para realizar a inferência das probabilidades de a imagem selecionada de ser um determinado personagem ou a probabilidade de ser o outro personagem.
-   Compactar em um arquivo com o seu nome:
    -   Documento contendo quais personagens foram utilizados e quais características de cada personagem foram escolhidas para a etapa de extração de características.
    -   Arquivo **\*.arff** com as características extraídas.
    -   Matriz de confusão gerada pelo algoritmo Naive Bayes.
    -   Código-fonte (pode ser um link para o github).

**Decisões:**

Para essa avaliação foram escolhidos os personagens Margie Simpsons e Kent Brockman.

## 🚀 Como usar

**Requerimentos:**

-   Python 3.8

Criar um ambiente virtual:

```sh
python -m venv venv
```

Carregar as variáveis de ambiente:

```sh
# bash
venv/Scripts/activate

# cmd
venv\Scripts\activate.bat

# powershell
venv/Scripts/Activate.ps1
```

Instalar as dependências do projeto:

```sh
pip install -r requirements.txt
```

Executar um script:

```sh
python script_path.py
```

## 🔑 Licença

Este projeto está sob a [licença MIT](LICENSE.md).
