import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from itertools import islice
from tkinter.messagebox import askyesnocancel, showwarning
from tkinter.filedialog import askopenfilename
from typing import Callable, List, Literal, Tuple

from PIL import Image, ImageTk

from unisul_machine_learning.weka import load_arff, save_arff


Features = Tuple[float, float, float, float]
Instance = Tuple[float, float, float, float, Literal['Marge', 'Skinner']]
Instances = List[Instance]

# caminhos principais
root_path = Path(__file__).parent.resolve()

marge_samples_path = (root_path / 'samples/marge_simpson').resolve()
skinner_samples_path = (root_path / 'samples/principal_skinner').resolve()

accepted_image_types = ('jpg', 'jpeg', 'png', 'bmp')

# meta dados do dataset
dataset_file_path = (root_path / 'marge_skinner_features.arff')
dataset_name = 'marge_skinner_features'
dataset_attributes = (
    ('marge_hair', 'real'),
    ('marge_dress', 'real'),
    ('skinner_hair', 'real'),
    ('skinner_suit', 'real'),
    ('class', '{Marge,Skinner}'),
)
dataset_instances: Instances = []

# algoritmos classificadores de aprendizado de máquina
naive_bayes_classifier = GaussianNB()
decision_tree_classifier = DecisionTreeClassifier()


def measure_features(image: Image.Image) -> Features:
    """Avalia caracteristicas prédefinidas de uma imagem.

    As características avaliadas a são: o cabelo azul da Marge Simpson,
    o vestido verde da Marge Simpson, o cabelo cinza do Diretor Skinner e
    o terno azul do Diretor Skinner.

    Parameters
    ----------
    image : Image.Image
        A imagem.

    Returns
    -------
    Features
        As pontuações da imagem em cada característica avaliada.
    """
    # Cabelo da Marge (azul)
    # rgb(36,50,237) rgb(42,56,243) rgb(53,63,174) rgb(57,73,169)
    # rgb(37,53,149) rgb(68,83,164) rgb(69,50,165) rgb(71,57,171)
    marge_hair_feature_score = 0.0

    # Vestido da Marge (verde)
    # rgb(147,193,85) rgb(166,206,83) rgb(103,155,10) rgb(114,167,15)
    # rgb(175,199,111) rgb(176,209,105) rgb(166,164,103) rgb(159,171,97)
    marge_dress_feature_score = 0.0

    # Cabelo do Skinner (cinza)
    # rgb(132,115,89) rgb(135,119,122) rgb(167,167,157) rgb(162,163,157)
    # rgb(111,96,91) rgb(116,95,90) rgb(137,134,125) rgb(140,139,137)
    skinner_hair_feature_score = 0.0

    # Terno do Skinner (azul)
    # rgb(56,103,129) rgb(56,106,131) rgb(48,103,142) rgb(47,106,146)
    # rgb(48,93,150) rgb(52,95,146) rgb(42,106,116) rgb(43,111,120)
    skinner_suit_feature_score = 0.0

    width, height = image.width, image.height
    half_height = (height - 1) / 2
    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))

            # verificar cabelo da Marge
            if y < half_height and 35 <= r <= 72 and 50 <= g <= 85 and 145 <= b <= 245:
                marge_hair_feature_score += 1.0

            # verificar vestido da Marge
            if y >= half_height and 100 <= r <= 178 and 150 <= g <= 210 and 10 <= b <= 110:
                marge_dress_feature_score += 1.0

            # verificar cabelo do Skinner
            if y < half_height and 110 <= r <= 170 and 95 <= g <= 170 and 85 <= b <= 160:
                skinner_hair_feature_score += 1.0

            # verificar terno do Skinner
            if y >= half_height and 40 <= r <= 60 and 90 <= g <= 115 and 115 <= b <= 155:
                skinner_suit_feature_score += 1.0

    # normalizar a pontuação de cada característica de acordo com o número de pixels na imagem
    marge_hair_feature_score = (marge_hair_feature_score / (width * height)) * 100
    marge_dress_feature_score = (marge_dress_feature_score / (width * height)) * 100
    skinner_hair_feature_score = (skinner_hair_feature_score / (width * height)) * 100
    skinner_suit_feature_score = (skinner_suit_feature_score / (width * height)) * 100

    return (marge_hair_feature_score, marge_dress_feature_score,
            skinner_hair_feature_score, skinner_suit_feature_score)


def extract_samples_instances():
    """Avalia as características de cada amostra e retorna uma lista de instâncias.

    Returns
    -------
    Instances
        A lista de instâncias contêndo a pontuação e classe de cada amostra.
    """
    # mostrar cabeçalho das informações que serão mostradas de cada amostra
    print('classe:arquivo | marge_cabelo | marge_vestido | skinner_cabelo | skinner_terno')
    print('------------------------------------------------------------------------------')

    samples_instances: Instances = []
    for sample_path in [
            sample_path
            for accepted_image_type in accepted_image_types
            for samples_path in (marge_samples_path, skinner_samples_path)
            for sample_path in islice(samples_path.glob(f'*.{accepted_image_type}'), 600)]:
        sample_instance = (*measure_features(Image.open(sample_path)),
                           'Marge' if sample_path.parent == marge_samples_path else 'Skinner')

        # mostrar a pontuação de cada característica e a classe da amostra no console
        print(f'{sample_instance[4]}:{sample_path.stem}',
              '->',
              f'{sample_instance[0]:.{3}f}',
              f'{sample_instance[1]:.{3}f}',
              f'{sample_instance[2]:.{3}f}',
              f'{sample_instance[3]:.{3}f}')

        samples_instances.append(sample_instance)

    return samples_instances


def predict(image: Image.Image):
    """Avalia caracteristicas prédefinidas de uma imagem e
    faz uma predição com base no conjunto de amostras treinados.

    Parameters
    ----------
    image : Image.Image
        A imagem.

    Returns
    -------
    tuple
        As pontuações da imagem em cada característica e a predição de cada classificador.
    """
    features = measure_features(image)

    return (features,
            tuple(naive_bayes_classifier.predict_proba([features])[0]),
            tuple(decision_tree_classifier.predict_proba([features])[0]))


# GUI
root = tk.Tk()
root.title('Aprendizado de Máquina - UNISUL')

root.resizable(False, False)


# helpers
def frame(parent: tk.Misc, column: int, row: int, column_span: int = 1, row_span: int = 1,
          padx: int = 5, pady: int = 5, sticky: str = ''):
    frame_widget = ttk.Frame(parent)
    frame_widget.grid(column=column, row=row, columnspan=column_span, rowspan=row_span,
                      padx=padx, pady=pady, sticky=sticky)
    return frame_widget


def responsive_named_frame(parent: tk.Misc, name: str, orientation: Literal["x", "y", "both"] = 'x',
                           padx: int = 5, pady: int = 5):
    frame_widget = ttk.LabelFrame(parent, text=name)
    frame_widget.pack(fill=orientation, padx=padx, pady=pady)
    return frame_widget


def responsive_text_label(parent: tk.Misc, text: str, padx: int = 5, pady: int = 5):
    label_widget = ttk.Label(parent, text=text)
    label_widget.pack(padx=padx, pady=pady)


def responsive_variable_label(parent: tk.Misc, variable: tk.Variable, padx: int = 5, pady: int = 5):
    label_widget = ttk.Label(parent, textvariable=variable)
    label_widget.pack(padx=padx, pady=pady)


def responsive_item_label(parent: tk.Misc, text: str, variable: tk.Variable,
                          padx: int = 5, pady: int = 5):
    child_frame = ttk.Frame(parent)
    child_frame.pack(fill='x', padx=padx, pady=pady)
    label_widget = ttk.Label(child_frame, text=text)
    label_widget.pack(side='left')
    value_widget = ttk.Label(child_frame, textvariable=variable)
    value_widget.pack(side='right')


def responsive_image_label(parent: tk.Misc, padx: int = 5, pady: int = 5):
    label_widget = ttk.Label(parent)
    label_widget.pack(fill='both', padx=padx, pady=pady)

    def set_image_label(image: Image.Image):
        image_widget = ImageTk.PhotoImage(image)
        label_widget.configure(image=image_widget)
        label_widget.image = image_widget

    return set_image_label


def responsive_button(parent: tk.Misc, text: str, action: Callable, padx: int = 5, pady: int = 5):
    button_widget = ttk.Button(parent, text=text, command=action)
    button_widget.pack(fill='x', padx=padx, pady=pady)


# variáveis
feature_var_1 = tk.StringVar(value='--')
feature_var_2 = tk.StringVar(value='--')
feature_var_3 = tk.StringVar(value='--')
feature_var_4 = tk.StringVar(value='--')

naive_bayes_marge_proba_var = tk.StringVar(value='--')
naive_bayes_skinner_proba_var = tk.StringVar(value='--')
decision_tree_marge_proba_var = tk.StringVar(value='--')
decision_tree_skinner_proba_var = tk.StringVar(value='--')

instances_status_var = tk.StringVar(value='Nenhuma instância treinada')

# quadros
root_frame = frame(root, 0, 0, padx=0, pady=0)
root_frame.columnconfigure(0, minsize=200)
root_frame.columnconfigure(1, minsize=200)
preview_frame = frame(root_frame, 0, 0, sticky='nsew')
main_frame = frame(root_frame, 1, 0, sticky='nsew')
status_frame = frame(root_frame, 0, 1, 2, sticky='nsew')
status_frame.configure(relief='sunken')

target_frame = responsive_named_frame(preview_frame, 'Alvo')
actions_frame = responsive_named_frame(main_frame, 'Ações')
features_frame = responsive_named_frame(main_frame, 'Características')
predictions_frame = responsive_named_frame(main_frame, 'Predições')

# widgets
set_target_image = responsive_image_label(target_frame)

responsive_item_label(features_frame, 'Cabelo da Marge', feature_var_1)
responsive_item_label(features_frame, 'Vestido da Marge', feature_var_2)
responsive_item_label(features_frame, 'Cabelo do Skinner', feature_var_3)
responsive_item_label(features_frame, 'Terno do Skinner', feature_var_4)

responsive_text_label(predictions_frame, 'Naive Bayes', pady=10)
responsive_item_label(predictions_frame, 'Marge', naive_bayes_marge_proba_var)
responsive_item_label(predictions_frame, 'Skinner', naive_bayes_skinner_proba_var)

responsive_text_label(predictions_frame, 'Árvore de Decisão', pady=10)
responsive_item_label(predictions_frame, 'Marge', decision_tree_marge_proba_var)
responsive_item_label(predictions_frame, 'Skinner', decision_tree_skinner_proba_var)

responsive_variable_label(status_frame, instances_status_var, 2, 2)


# comandos
def train():
    """Treina os algoritmos com um conjunto de amostras."""
    global dataset_instances
    load_dataset = False
    if dataset_file_path.is_file():
        load_dataset = askyesnocancel(
            'Um conjunto de amostras treinadas foi encontrado',
            'Um conjunto de amostras treinadas foi encontrado, deseja carregá-lo?'
            ' Caso não, um novo conjunto será treinado e o substituirá.')

    if load_dataset is None:
        return
    elif load_dataset:
        dataset_instances.clear()
        dataset_instances.extend([(float(instance[0]),
                                   float(instance[1]),
                                   float(instance[2]),
                                   float(instance[3]),
                                   str(instance[4]))
                                  for instance in load_arff(dataset_file_path)[2]])
    else:
        dataset_instances.clear()
        dataset_instances.extend(extract_samples_instances())
        save_arff(dataset_file_path, dataset_name, dataset_attributes, dataset_instances)

    instances_status_var.set(f'Instâncias treinadas: {len(dataset_instances)}')

    X = np.array([list(instance[:4]) for instance in dataset_instances])
    y = np.array([instance[4] for instance in dataset_instances])

    naive_bayes_classifier.fit(X, y)
    decision_tree_classifier.fit(X, y)


def classify():
    """Classifica um imagem."""
    if not dataset_instances:
        showwarning('Nenhuma amostra treinada',
                    'Não é possível classificar uma imagem sem amostras treinadas.'
                    ' Treine um conjunto de amostras antes de classificar uma imagem.')
        return

    image_path = askopenfilename(title='Selecione uma imagem',
                                 filetypes=(
                                     (f'{accepted_image_type} images', f'*.{accepted_image_type}')
                                     for accepted_image_type in accepted_image_types),
                                 initialdir=root_path)

    if image_path:
        target_image = Image.open(image_path)

        set_target_image(target_image)

        features, naive_bayes_proba, decision_tree_proba = predict(target_image)

        feature_var_1.set(f'{features[0]:.{3}f}')
        feature_var_2.set(f'{features[1]:.{3}f}')
        feature_var_3.set(f'{features[2]:.{3}f}')
        feature_var_4.set(f'{features[3]:.{3}f}')

        naive_bayes_marge_proba_var.set(f'{naive_bayes_proba[0] * 100:.{3}f}%')
        naive_bayes_skinner_proba_var.set(f'{naive_bayes_proba[1] * 100:.{3}f}%')
        decision_tree_marge_proba_var.set(f'{decision_tree_proba[0] * 100:.{3}f}%')
        decision_tree_skinner_proba_var.set(f'{decision_tree_proba[1] * 100:.{3}f}%')


# ações
responsive_button(actions_frame, 'Treinar', train)
responsive_button(actions_frame, 'Classificar', classify)

root.mainloop()
