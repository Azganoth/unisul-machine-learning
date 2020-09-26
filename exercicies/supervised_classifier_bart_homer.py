from pathlib import Path

from PIL import Image

weka_data = '''
@relation caracteristicas

@attribute laranja_camisa_bart real
@attribute azul_calcao_bart real
@attribute azul_sapato_bart real
@attribute marrom_boca_homer real
@attribute azul_calca_homer real
@attribute cinza_sapato_homer real
@attribute classe {Bart, Homer}

@data
'''

root_path = (Path(__file__).parent / '..').resolve()

samples_path = (root_path / 'samples/bart-homer').resolve()
samples_globs = ['*.jpg', '*.png', '*.bmp', '*.gif']

for sample_path in [sample_path
                    for samples_glob in samples_globs for sample_path in samples_path.glob(
                        samples_glob)]:
    sample = Image.open(sample_path)

    # Extrair caracteristicas
    characteristics_bart_shirt = 0  # camisa do bart (laranja)
    characteristics_bart_pants = 0  # calça do bart (azul)
    characteristics_bart_shoes = 0  # sapatos do bart (azul)
    characteristics_homer_beard = 0  # barba do homer (marrom)
    characteristics_homer_pants = 0  # calça do homer (azul)
    characteristics_homer_shoes = 0  # sapatos do homer (cinza)

    width, height = sample.width, sample.height
    for x in range(width):
        for y in range(height):
            r, g, b = sample.getpixel((x, y))

            # Verificar camisa do bart
            if 200 <= r <= 255 and 70 <= g <= 105 and 7 <= b <= 90:
                characteristics_bart_shirt += 1

            # Verificar calça do bart
            if y > height / 2 and 0 <= r <= 20 and 5 <= g <= 125 and 125 <= b <= 170:
                characteristics_bart_pants += 1

            # Verificar sapatos do bart
            if y > height / 2 + height / 3 and 5 <= r <= 20 and 3 <= g <= 12 and 125 <= b <= 140:
                characteristics_bart_shoes += 1

            # Verificar barba do homer
            if y < height / 2 + height / 3 and 175 <= r <= 200 and 160 <= g <= 185 and 95 <= b <= 140:
                characteristics_homer_beard += 1

            # Verificar calça do homer
            if 0 <= r <= 90 and 98 <= g <= 120 and 150 <= b <= 180:
                characteristics_homer_pants += 1

            # Verificar sapatos do homer
            if y > height / 2 + height / 3 and 25 <= r <= 45 and 25 <= g <= 45 and 25 <= b <= 45:
                characteristics_homer_shoes += 1

    # Normalizar as características pelo número de pixels totais da imagem para porcentagem
    characteristics_bart_shirt = (characteristics_bart_shirt / (width * height)) * 100
    characteristics_bart_pants = (characteristics_bart_pants / (width * height)) * 100
    characteristics_bart_shoes = (characteristics_bart_shoes / (width * height)) * 100
    characteristics_homer_beard = (characteristics_homer_beard / (width * height)) * 100
    characteristics_homer_pants = (characteristics_homer_pants / (width * height)) * 100
    characteristics_homer_shoes = (characteristics_homer_shoes / (width * height)) * 100

    # APRENDIZADO SUPERVISIONADO - JÁ SABEMOS QUAL A CLASSE NAS IMAGENS DE TREINAMENTO
    characteristics_class = 'Bart' if sample_path.stem.startswith('bart') else 'Homer'

    print((f'{sample_path.stem} -> '
           f'{characteristics_bart_shirt} '
           f'{characteristics_bart_pants} '
           f'{characteristics_bart_shoes} '
           f'{characteristics_homer_beard} '
           f'{characteristics_homer_pants} '
           f'{characteristics_homer_shoes} '
           f'{characteristics_class}'))

    weka_data += (f'{characteristics_bart_shirt},'
                  f'{characteristics_bart_pants},'
                  f'{characteristics_bart_shoes},'
                  f'{characteristics_homer_beard},'
                  f'{characteristics_homer_pants},'
                  f'{characteristics_homer_shoes},'
                  f'{characteristics_class}\n')

with open((root_path / 'supervised_classifier_bart_homer_characteristics.arff'), 'w') as file:
    print(weka_data.strip(), file=file)
