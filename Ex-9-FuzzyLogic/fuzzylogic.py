import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.filters import sobel
from skimage.transform import resize
from skfuzzy import control as ctrl
import skfuzzy as fuzz

# Import RGB Image and Convert to Grayscale
Irgb = io.imread('peppers.png')  # Replace with your image path
Igray = color.rgb2gray(Irgb)

# Resize the image to a standard size (optional)
Igray = resize(Igray, (384, 512))

# Plot the Grayscale Image
plt.figure()
plt.imshow(Igray, cmap='gray')
plt.title('Input Image in Grayscale')
plt.axis('off')
plt.show()

# Define fuzzy input variables (Gradients in X and Y directions)
Ix = ctrl.Antecedent(np.linspace(-1, 1, 100), 'Ix')
Iy = ctrl.Antecedent(np.linspace(-1, 1, 100), 'Iy')

# Define fuzzy output variable (Edge intensity)
Iout = ctrl.Consequent(np.linspace(0, 1, 100), 'Iout')

# Define membership functions for the inputs
Ix['zero'] = fuzz.gaussmf(Ix.universe, 0, 0.1)
Iy['zero'] = fuzz.gaussmf(Iy.universe, 0, 0.1)

# Define membership functions for the output
Iout['white'] = fuzz.trimf(Iout.universe, [0.1, 1, 1])
Iout['black'] = fuzz.trimf(Iout.universe, [0, 0, 0.7])

# Create fuzzy rules
rule1 = ctrl.Rule(Ix['zero'] & Iy['zero'], Iout['white'])
rule2 = ctrl.Rule(Ix['zero'].invert() | Iy['zero'].invert(), Iout['black'])

# Create and simulate the fuzzy inference system
edge_detection_ctrl = ctrl.ControlSystem([rule1, rule2])
edge_detection = ctrl.ControlSystemSimulation(edge_detection_ctrl)

# Compute the gradients (Ix and Iy) using Sobel filters
Ix_grad = sobel(Igray, axis=1)
Iy_grad = sobel(Igray, axis=0)

# Initialize an array to store the edge detection results
Ieval = np.zeros_like(Igray)

# Evaluate the fuzzy system for each pixel
for i in range(Igray.shape[0]):
    for j in range(Igray.shape[1]):
        edge_detection.input['Ix'] = Ix_grad[i, j]
        edge_detection.input['Iy'] = Iy_grad[i, j]
        edge_detection.compute()
        Ieval[i, j] = edge_detection.output['Iout']

# Plot the Original Grayscale Image
plt.figure()
plt.imshow(Igray, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')
plt.show()

# Plot the Detected Edges
plt.figure()
plt.imshow(Ieval, cmap='gray')
plt.title('Edge Detection Using Fuzzy Logic')
plt.axis('off')
plt.show()
