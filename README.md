# unisul-machine-learning

Coleção de scripts utilizados para a matéria Aprendizado de Máquina na UNISUL.

## 📜 Scripts

### Avaliação 1

#### Enunciado

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

#### Decisões

Para essa avaliação foram escolhidos os personagens Marge Simpson e Diretor Skinner.

Como características foram escolhidas para cada personagem:

-   **Marge Simpson:** o cabelo azul e o vestido verde
-   **Diretor Skinner:** o cabelo cinza e o terno azul

#### Matriz de confusão

![Confusion Matrix](/docs/test_1_confusion_matrix.png)

#### Executar

```sh
python test_1.py
```

### Classificação Marge Simpson e Diretor Skinner

Avaliação 1 com adição do algoritmo da Árvore de Decisão.

#### Executar

```sh
python marge_skinner.py
```

### Avaliação 2

#### Enunciado

- Desenvolver um programa.
- O programa deve realizar a extração de características de sons.
- O programa deve criar um arquivo *\*.arff* contendo as características extraídas.
- O programa deve treinar uma rede neural perceptron multicamadas com as características extraídas.
- O programa deve permitir o usuário escolher um arquivo de som (*.wav*) e informar a pontuação do som obtida na rede neural treinada.

##### Entregar

- O código-fonte do programa.
- Descrição das características.
- Configurações da rede neural.
- Arquivo de características *\*.arff*.

##### Detalhes do dataset

O dataset [Audio Cats and Dogs](https://www.kaggle.com/mmoreaux/audio-cats-and-dogs) consiste de:

- 164 arquivos WAV de miados de gato correspondendo a um total de 1323 segundos de audio;
- 113 arquivos WAV de latidos de cachorro correspondendo a um total de 598 segundos de audio.

Todos os arquivos WAV possuem frequência de 16KHz e duração variável.

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
