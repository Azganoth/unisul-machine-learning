import numpy as np
import tkinter as tk
from colorsys import rgb_to_hls, hls_to_rgb
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from tkinter import filedialog
from typing import List, Literal, Tuple

from PIL import Image, ImageTk


# lol, o python troca a ordem das coordenadas de "HSL" para "HLS"...
def rgb_to_hsl(r: float, g: float, b: float) -> Tuple[float, float, float]:
    '''Convert the color from RGB coordinates to HSL coordinates.'''
    h, l, s = rgb_to_hls(r, g, b)
    return h, s, l


def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
    '''Convert the color from HSL coordinates to RGB coordinates.'''
    return hls_to_rgb(h, l, s)


root_path = (Path(__file__).parent / '..').resolve()

marge_samples_path = (root_path / 'samples/test_1/marge_simpson').resolve()
skinner_samples_path = (root_path / 'samples/test_1/principal_skinner').resolve()
weka_file_path = (root_path / 'test_1.arff')

image_globs = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

samples_characteristics: List[Tuple[float, float, float, float, Literal['Marge', 'Skinner']]] = []

weka_header = '''
@relation caracteristicas

@attribute marge_cabelo real
@attribute marge_vestido real
@attribute skinner_cabelo real
@attribute skinner_terno real
@attribute classe {Marge, Skinner}

@data
'''


def extract_characteristics(image: Image.Image):
    # Cabelo da Marge (azul)
    # Tons:
    # - pic0000 -> (rgb(36,50,237), hsl(235.8,0.848,0.535)), (rgb(42,56,243), hsl(235.8,0.893,0.559))
    # - pic0001 -> (rgb(53,63,174), hsl(235,0.533,0.445)), (rgb(57,73,169), hsl(231.4,0.496,0.443))
    # - pic0002 -> (rgb(37,53,149), hsl(231.4,0.602,0.365)), (rgb(68,83,164), hsl(230.6,0.414,0.455))
    # - pic0023 -> (rgb(69,50,165), hsl(249.9,0.535,0.422)), (rgb(71,57,171), hsl(247.4,0.5,0.447))
    characteristic_marge_hair = 0.0

    # Vestido da Marge (verde)
    # Tons:
    # - pic0000 -> (rgb(147,193,85), hsl(85.6,0.466,0.545)), (rgb(166,206,83), hsl(79.5,0.557,0.567))
    # - pic0001 -> (rgb(103,155,10), hsl(81.5,0.879,0.324)), (rgb(114,167,15), hsl(80.9,0.835,0.357))
    # - pic0002 -> (rgb(175,199,111), hsl(76.4,0.44,0.608)), (rgb(176,209,105), hsl(79,0.531,0.616))
    # - pic0023 -> (rgb(166,164,103), hsl(58.1,0.261,0.527)), (rgb(159,171,97), hsl(69.7,0.306,0.525))
    characteristic_marge_dress = 0.0

    # Cabelo do Skinner (cinza)
    # Tons:
    # - pic0000 -> (rgb(132,115,89), hsl(36.3,0.195,0.433)), (rgb(135,119,122), hsl(348.8,0.63,0.498))
    # - pic0002 -> (rgb(167,167,157), hsl(60,0.54,0.635)), (rgb(162,163,157), hsl(70,0.32,0.627))
    # - pic0004 -> (rgb(111,96,91), hsl(15,0.99,0.396)), (rgb(116,95,90), hsl(11.5,0.126,0.404))
    # - pic0034 -> (rgb(137,134,125), hsl(45,0.48,0.514)), (rgb(140,139,137), hsl(40,0.13,0.543))
    characteristic_skinner_hair = 0.0

    # Terno do Skinner (azul)
    # Tons:
    # - pic0000 -> (rgb(56,103,129), hsl(201.4,0.395,0.363)), (rgb(56,106,131), hsl(200,0.401,0.367))
    # - pic0002 -> (rgb(48,103,142), hsl(204.9,0.495,0.373)), (rgb(47,106,146), hsl(204.2,0.513,0.378))
    # - pic0004 -> (rgb(48,93,150), hsl(213.5,0.515,0.388)), (rgb(52,95,146), hsl(212.6,0.475,0.388))
    # - pic0034 -> (rgb(42,106,116), hsl(188.1,0.468,0.31)), (rgb(43,111,120), hsl(187,0.472,0.32))
    characteristic_skinner_suit = 0.0

    width, height = image.width, image.height
    half_height = (height - 1) / 2
    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))
            # h, s, l = rgb_to_hsl(r / 255, g / 255, b / 255)

            # Verificar cabelo da Marge
            # if y < half_height and 0.63 <= h <= 0.71 and 0.4 <= s <= 0.9 and 0.3 <= l <= 0.6:
            if y < half_height and 35 <= r <= 72 and 50 <= g <= 85 and 145 <= b <= 245:
                characteristic_marge_hair += 1.0
                # print(h, s, l)

            # Verificar vestido da Marge
            # if y >= half_height and 0.15 <= h <= 0.23 and 0.25 <= s <= 0.75 and 0.3 <= l <= 0.625:
            if y >= half_height and 100 <= r <= 178 and 150 <= g <= 210 and 10 <= b <= 110:
                characteristic_marge_dress += 1.0

            # Verificar cabelo do Skinner
            # if y < half_height and 0 <= h <= 0.20 and 0.025 <= s <= 1 and 0.3 <= l <= 0.65:
            if y < half_height and 110 <= r <= 170 and 95 <= g <= 170 and 85 <= b <= 160:
                characteristic_skinner_hair += 1.0

            # Verificar terno do Skinner
            # if y >= half_height and 0.5 <= h <= 0.6 and 0.37 <= s <= 0.9 and 0.25 <= l <= 0.4:
            if y >= half_height and 40 <= r <= 60 and 90 <= g <= 115 and 115 <= b <= 155:
                characteristic_skinner_suit += 1.0

    # Normalizar as características pelo número de pixels totais da imagem para porcentagem
    characteristic_marge_hair = (characteristic_marge_hair / (width * height)) * 100
    characteristic_marge_dress = (characteristic_marge_dress / (width * height)) * 100
    characteristic_skinner_hair = (characteristic_skinner_hair / (width * height)) * 100
    characteristic_skinner_suit = (characteristic_skinner_suit / (width * height)) * 100

    return (characteristic_marge_hair, characteristic_marge_dress,
            characteristic_skinner_hair, characteristic_skinner_suit)


# GUI
root = tk.Tk()
root.title('Avaliação 1 - Marge Simpson e Diretor Skinner')

# variáveis
samples_label_var = tk.IntVar()

marge_hair_label_var = tk.StringVar(value='---')
marge_dress_label_var = tk.StringVar(value='---')
skinner_hair_label_var = tk.StringVar(value='---')
skinner_suit_label_var = tk.StringVar(value='---')

marge_prediction_label_var = tk.StringVar(value='---')
skinner_prediction_label_var = tk.StringVar(value='---')


# comandos
def train():
    samples_characteristics.clear()
    for sample_path in [sample_path
                        for image_glob in image_globs
                        for samples_path in [marge_samples_path, skinner_samples_path]
                        for sample_path in list(samples_path.glob(image_glob))[:300]]:
        sample_characteristics = (
            *extract_characteristics(Image.open(sample_path)),
            'Marge' if sample_path.parent == marge_samples_path else 'Skinner')

        print((f'{sample_characteristics[4]}:{sample_path.stem} -> '
               f'{sample_characteristics[0]:.{3}f} '
               f'{sample_characteristics[1]:.{3}f} '
               f'{sample_characteristics[2]:.{3}f} '
               f'{sample_characteristics[3]:.{3}f}'))

        samples_characteristics.append(sample_characteristics)

    weka_content = weka_header + '\n'.join(
        (','.join(map(str, characteristic)) for characteristic in samples_characteristics))

    with open(weka_file_path, 'w') as file:
        print(weka_content.strip(), file=file)

    samples_label_var.set(len(samples_characteristics))


def load():
    if weka_file_path.is_file():
        with open(weka_file_path) as weka_file:
            samples_characteristics.clear()
            for weka_data in weka_file.read().replace(weka_header.strip(), '').splitlines():
                characteristics = tuple(filter(bool, weka_data.split(',')))
                if len(characteristics) == 5:
                    samples_characteristics.append((float(characteristics[0]),
                                                    float(characteristics[1]),
                                                    float(characteristics[2]),
                                                    float(characteristics[3]),
                                                    characteristics[4]))

        samples_label_var.set(len(samples_characteristics))
    else:
        pass


def predict():
    selected_image_path = filedialog.askopenfilename(
        parent=root, initialdir=root_path, title='Selecione uma imagem',
        filetypes=((f'{image_glob[2:]} images', image_glob) for image_glob in image_globs))

    if selected_image_path:
        selected_image = Image.open(selected_image_path)

        selected_image_widget = ImageTk.PhotoImage(selected_image)
        image_label.configure(image=selected_image_widget)
        image_label.image = selected_image_widget

        selected_image_characteristics = extract_characteristics(selected_image)
        marge_hair_label_var.set(f'{selected_image_characteristics[0]:.{3}f}')
        marge_dress_label_var.set(f'{selected_image_characteristics[1]:.{3}f}')
        skinner_hair_label_var.set(f'{selected_image_characteristics[2]:.{3}f}')
        skinner_suit_label_var.set(f'{selected_image_characteristics[3]:.{3}f}')

        clf = GaussianNB()

        X = np.array([list(characteristics[:4]) for characteristics in samples_characteristics])
        y = np.array([[characteristics[4]] for characteristics in samples_characteristics])

        clf.fit(X, y)

        prediction = clf.predict([list(selected_image_characteristics)])

        result_marge = 1 if prediction == ['Marge'] else 0
        result_skinner = 1 if prediction == ['Skinner'] else 0

        marge_prediction_label_var.set(f'{result_marge * 100:.{1}f}%')
        skinner_prediction_label_var.set(f'{result_skinner * 100:.{1}f}%')


# quadros
main_frame = tk.Frame(root)
side_frame = tk.Frame(root)

actions_frame = tk.LabelFrame(side_frame, text='Ações')
samples_frame = tk.LabelFrame(side_frame, text='Amostras')
characteristics_frame = tk.LabelFrame(side_frame, text='Características')
prediction_frame = tk.LabelFrame(side_frame, text='Predição')

characteristic_frame_1 = tk.Frame(characteristics_frame)
characteristic_frame_2 = tk.Frame(characteristics_frame)
characteristic_frame_3 = tk.Frame(characteristics_frame)
characteristic_frame_4 = tk.Frame(characteristics_frame)

predict_frame_1 = tk.Frame(prediction_frame)
predict_frame_2 = tk.Frame(prediction_frame)

# widgets
# TODO: don't load an image
# marge_image = ImageTk.PhotoImage(Image.open((marge_samples_path / 'pic_0002.jpg')))
# image_label = tk.Label(main_frame, image=marge_image)
image_label = tk.Label(main_frame)

extract_button = tk.Button(actions_frame, text='Extrair características', command=train)
load_button = tk.Button(actions_frame, text='Carregar características', command=load)
predict_button = tk.Button(actions_frame, text='Selecionar imagem', command=predict)

samples_label_val = tk.Label(samples_frame, textvariable=samples_label_var)

marge_hair_label_text = tk.Label(characteristic_frame_1, text='Cabelo da Marge:')
marge_hair_label_val = tk.Label(characteristic_frame_1, textvariable=marge_hair_label_var)
marge_dress_label_text = tk.Label(characteristic_frame_2, text='Vestido da Marge:')
marge_dress_label_val = tk.Label(characteristic_frame_2, textvariable=marge_dress_label_var)
skinner_hair_label_text = tk.Label(characteristic_frame_3, text='Cabelo do Skinner:')
skinner_hair_label_val = tk.Label(characteristic_frame_3, textvariable=skinner_hair_label_var)
skinner_suit_label_text = tk.Label(characteristic_frame_4, text='Terno do Skinner:')
skinner_suit_label_val = tk.Label(characteristic_frame_4, textvariable=skinner_suit_label_var)

marge_prediction_label_text = tk.Label(predict_frame_1, text='Marge:')
marge_prediction_label_val = tk.Label(predict_frame_1, textvariable=marge_prediction_label_var)
skinner_prediction_label_text = tk.Label(predict_frame_2, text='Skinner:')
skinner_prediction_label_val = tk.Label(predict_frame_2, textvariable=skinner_prediction_label_var)

# geometria
root.resizable(False, False)
root.configure(padx=5, pady=5)

main_frame.grid(column=0, row=0, padx=5, pady=5, sticky=tk.NSEW)
main_frame.columnconfigure(0, minsize=300)
main_frame.rowconfigure(0, minsize=300)
side_frame.grid(column=1, row=0, sticky=tk.NSEW)

actions_frame.pack(fill=tk.X, padx=5, pady=5)
samples_frame.pack(fill=tk.X, padx=5, pady=5)
characteristics_frame.pack(fill=tk.X, padx=5, pady=5)
prediction_frame.pack(fill=tk.X, padx=5, pady=5)

characteristic_frame_1.pack(fill=tk.X, padx=5, pady=5)
characteristic_frame_2.pack(fill=tk.X, padx=5, pady=5)
characteristic_frame_3.pack(fill=tk.X, padx=5, pady=5)
characteristic_frame_4.pack(fill=tk.X, padx=5, pady=5)

predict_frame_1.pack(fill=tk.X, padx=5, pady=5)
predict_frame_2.pack(fill=tk.X, padx=5, pady=5)

image_label.pack(padx=5, pady=5)

extract_button.pack(fill=tk.X, padx=5, pady=5)
load_button.pack(fill=tk.X, padx=5, pady=5)
predict_button.pack(fill=tk.X, padx=5, pady=5)

samples_label_val.pack(padx=5, pady=5)

marge_hair_label_text.pack(padx=5, pady=5, side=tk.LEFT)
marge_hair_label_val.pack(padx=5, pady=5, side=tk.RIGHT)
marge_dress_label_text.pack(padx=5, pady=5, side=tk.LEFT)
marge_dress_label_val.pack(padx=5, pady=5, side=tk.RIGHT)
skinner_hair_label_text.pack(padx=5, pady=5, side=tk.LEFT)
skinner_hair_label_val.pack(padx=5, pady=5, side=tk.RIGHT)
skinner_suit_label_text.pack(padx=5, pady=5, side=tk.LEFT)
skinner_suit_label_val.pack(padx=5, pady=5, side=tk.RIGHT)

marge_prediction_label_text.pack(padx=5, pady=5, side=tk.LEFT)
marge_prediction_label_val.pack(padx=5, pady=5, side=tk.RIGHT)
skinner_prediction_label_text.pack(padx=5, pady=5, side=tk.LEFT)
skinner_prediction_label_val.pack(padx=5, pady=5, side=tk.RIGHT)

root.mainloop()
