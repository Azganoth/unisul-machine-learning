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

#### Descrição das características

`chroma_stft_mean` `chroma_stft_var`

Média e mediana dos valores do cromagrama de cada quadro do audio.

> Um cromagrama é a projeção de um espectro de quadro de audio em 12 caixas que representam os 12 semitons distintos (ou croma) da oitava musical (intervalo entre uma nota musical e outra com a metade ou o dobro de sua frequência).

`rms_mean` `rms_var`

Média e mediana dos valores da raiz do valor quadrático médio de cada quadro do audio.

`spectral_centroid_mean` `spectral_centroid_var`

Média e mediana dos valores da centróide espectral de cada quadro do audio.

> O Centróide espectral indica onde o centro da massa de um audio está localizada e calcula a média ponderada das frequências presentes no audio.

`spectral_bandwidth_mean` `spectral_bandwidth_var`

Média e mediana dos valores da largura de banda espectral de cada quadro do audio.

#### Configurações da rede neural

- As camadas ocultas consistem de **2** *(duas)* camadas com **5** *(cinco)* neurônios em cada.
- A função tangente hiperbólica mostrou o melhor resultado como função de ativação.
- A taxa de aprendizagem `0.2` se mostrou a mais eficiente.
- O momentum `0.15` mostrou um bom resultado.
- Apesar do treinamento ser concluído por volta de 500 iterações, foi dado um máximo de 1000 iterações.

#### Executar

```sh
python test_2.py
```

### Avaliação 3

#### Enunciado

Descobrir “informações” que não estão visíveis no dataset, como por exemplo:

- Existe associação entre vendas?
- Percebe-se mudança de perfil das vendas a medida que o tempo passa?
- É possível fazer algum agrupamento baseando-se em vendas?
- É possível descobrir algum perfil de jogador com base no local da venda?
- O nome do jogo está associado ao gênero?
- É possível prever se alguma editora está em queda ou melhorando as vendas a medida que o tempo passa?
- Existe associações entre gêneros e plataformas? Ou entre gêneros e vendas?
- Outras descobertas.

##### Entregar

- Descrição de todas as técnicas, algoritmos e parâmetros utilizados como teste do dataset, mesmo as que não descobriram absolutamente nada como resultado final.
- Descrição da técnica, algoritmo e parâmetros que geraram alguma descoberta.
- Descrição da(s) descoberta(s) obtida(s).
- Informação de como foram feitos os testes (desenvolvimento de aplicação ou uso de alguma aplicação como o WEKA).
- No caso de desenvolvimento disponibilização do códigofonte.
- Parágrafo conclusivo relacionando o trabalho com os aspectos abordados na Unidade de Aprendizagem.

##### Detalhes do dataset

O dataset [Venda de jogos](/samples/vendas_de_jogos.csv) consiste de:

- 16.598 entradas contendo informações de venda de jogos.
- Cada entrada contém as seguintes informações sobre uma venda:
  - **Ranking:** posição no *ranking* de vendas;
  - **Nome:** nome do jogo;
  - **Plataforma:** plataforma em que o jogo foi liberado (PC, PS4, XBOX, etc);
  - **Ano:** ano de lançamento do jogo;
  - **Gênero:** gênero do jogo;
  - **Editora:** empresa que publicou o jogo;
  - **Vendas América do Norte:** vendas na América do Norte (em milhões de dólares);
  - **Vendas EUA:** vendas na Europa (em milhões de dólares);
  - **Vendas Japão:** vendas no Japão (em milhões de dólares);
  - **Vendas em outros paises:** vendas no restante do mundo (em milhões de dólares);
  - **Vendas totais:** total de vendas no mundo inteiro (em milhões de dólares).

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
