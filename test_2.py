import numpy as np
import librosa
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from tkinter.messagebox import askyesnocancel, showwarning
from tkinter.filedialog import askopenfilename
from typing import List, Literal, Tuple

from unisul_machine_learning.weka import *
from unisul_machine_learning.tkinter import *


Features = Tuple[float, float, float, float, float, float, float, float]
Instance = Tuple[float, float, float, float, float, float, float, float, Literal['Cat', 'Dog']]
Instances = List[Instance]

# caminhos principais
root_path = Path(__file__).parent.resolve()

cat_meows_samples_path = (root_path / 'samples/cat_meows').resolve()
dog_woofs_samples_path = (root_path / 'samples/dog_woofs').resolve()

samples_paths: List[Path] = (list(cat_meows_samples_path.glob('*.wav')) +
                             list(dog_woofs_samples_path.glob('*.wav')))

# meta dados do dataset
dataset_file_path = (root_path / 'test_2.arff')
dataset_name = 'cats_meows_and_dog_woofs_features'
dataset_attributes = (
    ('chroma_stft_mean', 'numeric'),
    ('chroma_stft_var', 'real'),
    ('rms_mean', 'real'),
    ('rms_var', 'real'),
    ('spectral_centroid_mean', 'real'),
    ('spectral_centroid_var', 'real'),
    ('spectral_bandwidth_mean', 'real'),
    ('spectral_bandwidth_var', 'real'),
    ('class', '{Cat,Dog}'),
)
dataset_instances: Instances = []

# rede neural perceptron multicamadas
mlp_classifier = MLPClassifier(hidden_layer_sizes=(5, 2), activation='tanh', solver='sgd',
                               learning_rate_init=0.2, momentum=0.15, max_iter=1000, random_state=1)

# normalizador
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))


def measure_features(sample_path: Path) -> Features:
    """Avalia características prédefinidas de uma amostra.

    Parameters
    ----------
    sample_path : Path
        O caminho da amostra.

    Returns
    -------
    Features
        A pontuação das características extraídas da amostra.
    """
    # carregar o audio
    audio, sr = librosa.load(sample_path, None)

    # remover silêncio do começo e final do audio
    audio, _ = librosa.effects.trim(audio)

    # extrair as características
    chroma_stft = librosa.feature.chroma_stft(audio, sr)
    rms = librosa.feature.rms(audio)
    spectral_centroid = librosa.feature.spectral_centroid(audio, sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(audio, sr)

    # calcular a média e mediana de cada característica
    return (np.mean(chroma_stft), np.median(chroma_stft),
            np.mean(rms), np.median(rms),
            np.mean(spectral_centroid), np.median(spectral_centroid),
            np.mean(spectral_bandwidth), np.median(spectral_bandwidth))


def extract_samples_instances():
    """Avalia as características de cada amostra, gerando e retornando uma lista de instâncias.

    Returns
    -------
    Instances
        A lista de instâncias contêndo a pontuação das características e classe de cada amostra.
    """
    # mostrar as categorias das informações que serão mostradas sobre cada amostra
    print('classe:arquivo -> chroma_stft_mean chroma_stft_var rms_mean rms_var spectral_centroid_mean spectral_centroid_var spectral_bandwidth_mean spectral_bandwidth_var', end='\n\n')

    samples_instances: Instances = []
    for sample_path in samples_paths:
        sample_instance = (*measure_features(sample_path),
                           'Cat' if sample_path.parent == cat_meows_samples_path else 'Dog')

        # mostrar a pontuação de cada característica e a classe da amostra
        print(f'{sample_instance[8]}:{sample_path.stem}',
              '->',
              f'{sample_instance[0]:.{3}f}',
              f'{sample_instance[1]:.{3}f}',
              f'{sample_instance[2]:.{3}f}',
              f'{sample_instance[3]:.{3}f}',
              f'{sample_instance[4]:.{3}f}',
              f'{sample_instance[5]:.{3}f}',
              f'{sample_instance[6]:.{3}f}',
              f'{sample_instance[7]:.{3}f}')

        samples_instances.append(sample_instance)

    return samples_instances


def fit():
    """Treina a rede neural.

    Returns:
        float: A acurácia da rede neural.
    """
    global dataset_instances

    # recuperar as entradas das instâncias do dataset
    X = np.array([instance[:8] for instance in dataset_instances])
    y = np.array([instance[8] for instance in dataset_instances])

    # normalizar as entradas numéricas para a escala de -1 a 1, para melhorar os resultados
    X = min_max_scaler.fit_transform(X)

    # treinar a rede neural com as entradas normalizadas
    mlp_classifier.fit(X, y)

    # calcular a acurácia da rede neural recém treinada
    score = mlp_classifier.score(X, y)

    return score


def predict(sample_path: Path):
    """Avalia caracteristicas prédefinidas de uma amostra e
    faz uma predição com base no conjunto de amostras treinados.

    Parameters
    ----------
    sample_path : Path
        O caminho da amostra.

    Returns
    -------
    tuple
        As pontuações da amostra em cada característica e a predição da rede neural.
    """
    features = measure_features(sample_path)

    # transformar as caracteristicas da amostra em uma numpy array
    X = np.array([features])

    # normalizar a entrada para a escala de -1 a 1
    X = min_max_scaler.transform(X)

    # calcular a predição em porcentagem da amostra na rede neural
    proba = tuple(mlp_classifier.predict_proba(X)[0])

    return features, proba


# GUI
root = create_parent('Avaliação 2 - Aprendizado de Máquina (UNISUL)', resizable=(False, False))
root.columnconfigure(0, minsize=500)

# quadros
root_frame = frame(root, sticky='nsew')

neural_network_frame = responsive_named_frame(root_frame, 'Rede Neural', padx=5, pady=5)
classification_frame = responsive_named_frame(root_frame, 'Classificação', padx=5, pady=5)

# variáveis
neural_network_score_var = str_var(root)
number_of_trained_instances_var = int_var(root, 0)

classification_file_path_var = str_var(root)

feature_var_1 = str_var(root)
feature_var_2 = str_var(root)
feature_var_3 = str_var(root)
feature_var_4 = str_var(root)
feature_var_5 = str_var(root)
feature_var_6 = str_var(root)
feature_var_7 = str_var(root)
feature_var_8 = str_var(root)

cat_meow_proba_var = str_var(root)
dog_woof_proba_var = str_var(root)


# ações
def train():
    """Carrega ou extrai as instâncias do conjunto de amostras e treina a rede neural."""
    global dataset_instances
    load_dataset = False
    if dataset_file_path.is_file():
        load_dataset = askyesnocancel(
            'Conjunto de amostras treinadas encontrado',
            'Um conjunto de amostras treinadas foi encontrado, deseja carregá-lo?'
            ' Caso não, um novo conjunto será treinado e o substituirá.')

    if load_dataset is None:
        return
    elif load_dataset:
        dataset_instances = [(float(instance[0]),
                              float(instance[1]),
                              float(instance[2]),
                              float(instance[3]),
                              float(instance[4]),
                              float(instance[5]),
                              float(instance[6]),
                              float(instance[7]),
                              str(instance[8]))
                             for instance in load_arff(dataset_file_path)[2]]
    else:
        dataset_instances = extract_samples_instances()
        save_arff(dataset_file_path, dataset_name, dataset_attributes, dataset_instances)

    score = fit()

    neural_network_score_var.set(f'{score * 100:.{2}f}%')
    number_of_trained_instances_var.set(len(dataset_instances))


def classify():
    """Classifica uma amostra."""
    if not dataset_instances:
        showwarning('Nenhuma amostra treinada',
                    'Não é possível classificar uma amostra sem amostras treinadas.'
                    ' Treine um conjunto de amostras antes de classificar uma amostra.')
        return

    sample_path = askopenfilename(title='Selecione uma amostra',
                                  filetypes=[('Arquivos de audio WAV', '*.wav')],
                                  initialdir=cat_meows_samples_path)

    if sample_path:
        classification_file_path_var.set(sample_path)
        features, proba = predict(Path(sample_path))

        feature_var_1.set(f'{features[0]:.{3}f}')
        feature_var_2.set(f'{features[1]:.{3}f}')
        feature_var_3.set(f'{features[2]:.{3}f}')
        feature_var_4.set(f'{features[3]:.{3}f}')
        feature_var_5.set(f'{features[4]:.{3}f}')
        feature_var_6.set(f'{features[5]:.{3}f}')
        feature_var_7.set(f'{features[6]:.{3}f}')
        feature_var_8.set(f'{features[7]:.{3}f}')

        cat_meow_proba_var.set(f'{proba[0] * 100:.{2}f}%')
        dog_woof_proba_var.set(f'{proba[1] * 100:.{2}f}%')
    else:
        classification_file_path_var.set('')

        feature_var_1.set('')
        feature_var_2.set('')
        feature_var_3.set('')
        feature_var_4.set('')
        feature_var_5.set('')
        feature_var_6.set('')
        feature_var_7.set('')
        feature_var_8.set('')

        cat_meow_proba_var.set('')
        dog_woof_proba_var.set('')


responsive_button(neural_network_frame, 'Treinar', train, padx=2, pady=2)
responsive_button(classification_frame, 'Classificar', classify, padx=2, pady=2)

# info
responsive_item_label(neural_network_frame, 'Acurácia:', neural_network_score_var, padx=5, pady=2)
responsive_item_label(neural_network_frame, 'Instâncias treinadas', number_of_trained_instances_var,
                      padx=5, pady=2)


responsive_variable_label(classification_frame, classification_file_path_var, padx=5, pady=2)

responsive_text_label(classification_frame, 'Características', padx=5, pady=10)
responsive_item_label(classification_frame, 'Chroma STFT mean', feature_var_1, padx=5, pady=2)
responsive_item_label(classification_frame, 'Chroma STFT variance', feature_var_2, padx=5, pady=2)
responsive_item_label(classification_frame, 'RMS mean', feature_var_3, padx=5, pady=2)
responsive_item_label(classification_frame, 'RMS variance', feature_var_4, padx=5, pady=2)
responsive_item_label(classification_frame, 'Spectral Centroid mean', feature_var_5,
                      padx=5, pady=2)
responsive_item_label(classification_frame, 'Spectral Centroid variance', feature_var_6,
                      padx=5, pady=2)
responsive_item_label(classification_frame, 'Spectral Bandwidth mean', feature_var_7,
                      padx=5, pady=2)
responsive_item_label(classification_frame, 'Spectral Bandwidth variance', feature_var_8,
                      padx=5, pady=2)

responsive_text_label(classification_frame, 'Predição', padx=5, pady=10)
responsive_item_label(classification_frame, 'Gato', cat_meow_proba_var, padx=5, pady=2)
responsive_item_label(classification_frame, 'Cachorro', dog_woof_proba_var, padx=5, pady=2)

root.mainloop()
