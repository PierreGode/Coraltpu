from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from PIL import Image

# Ladda modellen
interpreter = make_interpreter('mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite')
interpreter.allocate_tensors()

# Ladda bilden
image = Image.open('parrot.jpg').convert('RGB').resize(common.input_size(interpreter))
common.set_input(interpreter, image)

# Kör inferens
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1)

# Läs etiketter
labels = read_label_file('inat_bird_labels.txt')
for c in classes:
    print(f'ID: {c.id}, Score: {c.score:.5f}, Label: {labels.get(c.id, "unknown")}')
