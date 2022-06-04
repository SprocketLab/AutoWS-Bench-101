fashion_mnist_labels = ["t-shirt or top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", 
                        "sneaker", "bag", "ankle boot"]
cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]

classes_ = {
    "mnist": [f"a photo of the number: \"{i}\"." for i in range(10)],
    "spherical_mnist": [f"{i}" for i in range(10)],
    "cifar10": [f'a photo of a {classname}.' for classname in cifar10_labels],
    "fashion_mnist": [templates[0].format(classname) for classname in fashion_mnist_labels],
    "spherical_mnist": [f"a photo of the number: \"{i}\" stereographically projected onto a sphere and back to the plane." for i in range(10)],
    "permuted_mnist": [f"a photo of the number: \"{i}\" permuted by a bit reversal permutation." for i in range(10)],
    "navier_stokes": ["The Navier-Stokes equation in the turbulent regime with low viscosity: 1e-5.", "The Navier-Stokes equation with high viscosity: 1e-3."],
    # [f"a photo of {label}" for label in fashion_mnist_labels],
}