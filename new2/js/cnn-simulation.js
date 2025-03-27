// CNN Simulation module

// Apply convolution to the input data
function applyConvolution(inputData, kernel, stride = 1, paddingType = 'valid') {
    const kernelSize = kernel.length;
    const inputHeight = inputData.length;
    const inputWidth = inputData[0].length;

    // Create padded input if needed
    let paddedInput = inputData;
    let paddingSize = 0;

    if (paddingType === 'same') {
        // Calculate padding to maintain the same output dimensions
        paddingSize = Math.floor(kernelSize / 2);
        paddedInput = padArray(inputData, paddingSize);
    }

    // Calculate output dimensions
    const outputHeight = Math.floor((paddedInput.length - kernelSize) / stride) + 1;
    const outputWidth = Math.floor((paddedInput[0].length - kernelSize) / stride) + 1;

    // Create output array
    const output = new Array(outputHeight);
    for (let i = 0; i < outputHeight; i++) {
        output[i] = new Array(outputWidth).fill(0);
    }

    // Apply convolution
    for (let y = 0; y < outputHeight; y++) {
        for (let x = 0; x < outputWidth; x++) {
            // Get the region of interest in the padded input
            let sum = 0;

            for (let ky = 0; ky < kernelSize; ky++) {
                for (let kx = 0; kx < kernelSize; kx++) {
                    const inputY = y * stride + ky;
                    const inputX = x * stride + kx;

                    sum += paddedInput[inputY][inputX] * kernel[ky][kx];
                }
            }

            output[y][x] = sum;
        }
    }

    return { convOutput: output, paddedInput };
}

// Add padding to an array
function padArray(array, padSize) {
    const height = array.length;
    const width = array[0].length;

    // Create new array with padding
    const padded = new Array(height + 2 * padSize);

    for (let i = 0; i < padded.length; i++) {
        padded[i] = new Array(width + 2 * padSize).fill(0);
    }

    // Copy the original array to the center
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            padded[y + padSize][x + padSize] = array[y][x];
        }
    }

    return padded;
}

// Apply ReLU activation function to the data
function applyReLU(data) {
    const height = data.length;
    const width = data[0].length;

    const output = new Array(height);

    for (let y = 0; y < height; y++) {
        output[y] = new Array(width);
        for (let x = 0; x < width; x++) {
            // ReLU: max(0, x)
            output[y][x] = Math.max(0, data[y][x]);
        }
    }

    return output;
}

// Apply max pooling to the data
function applyMaxPooling(data, poolSize = 2, stride = 2) {
    const height = data.length;
    const width = data[0].length;

    // Calculate output dimensions
    const outputHeight = Math.floor((height - poolSize) / stride) + 1;
    const outputWidth = Math.floor((width - poolSize) / stride) + 1;

    // Create output array
    const output = new Array(outputHeight);
    for (let i = 0; i < outputHeight; i++) {
        output[i] = new Array(outputWidth).fill(0);
    }

    // Apply max pooling
    for (let y = 0; y < outputHeight; y++) {
        for (let x = 0; x < outputWidth; x++) {
            // Find the maximum value in the pooling window
            let maxVal = -Infinity;

            for (let py = 0; py < poolSize; py++) {
                for (let px = 0; px < poolSize; px++) {
                    const inputY = y * stride + py;
                    const inputX = x * stride + px;

                    if (inputY < height && inputX < width) {
                        maxVal = Math.max(maxVal, data[inputY][inputX]);
                    }
                }
            }

            output[y][x] = maxVal;
        }
    }

    return output;
}

// Create a simple CNN model using TensorFlow.js (demonstration only)
function createCNNModel() {
    // Create a simple CNN model for demonstration
    const model = tf.sequential();

    // Add a convolutional layer
    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 3,
        filters: 16,
        activation: 'relu',
        padding: 'same'
    }));

    // Add a max pooling layer
    model.add(tf.layers.maxPooling2d({
        poolSize: 2,
        strides: 2
    }));

    // Add another convolutional layer
    model.add(tf.layers.conv2d({
        kernelSize: 3,
        filters: 32,
        activation: 'relu',
        padding: 'same'
    }));

    // Add another max pooling layer
    model.add(tf.layers.maxPooling2d({
        poolSize: 2,
        strides: 2
    }));

    // Flatten the output for the dense layers
    model.add(tf.layers.flatten());

    // Add a dense layer
    model.add(tf.layers.dense({
        units: 128,
        activation: 'relu'
    }));

    // Add output layer
    model.add(tf.layers.dense({
        units: 10,
        activation: 'softmax'
    }));

    return model;
}

// Function to run a CNN on a sample image (for demonstration)
async function runCNNOnSample() {
    try {
        // Create the model
        const model = createCNNModel();

        // Get the image data from the input canvas
        const inputCanvas = document.getElementById('input-canvas');
        const ctx = inputCanvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, 28, 28); // Assuming 28x28 canvas or resize

        // Prepare the data for the model
        const inputTensor = tf.browser.fromPixels(imageData, 1)
            .reshape([1, 28, 28, 1])
            .toFloat()
            .div(255.0);

        // Run the model to get the prediction
        const prediction = model.predict(inputTensor);
        const probabilities = prediction.dataSync();

        // Find the class with the highest probability
        let maxProb = 0;
        let predictedClass = 0;

        for (let i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                predictedClass = i;
            }
        }

        console.log(`CNN Prediction: Class ${predictedClass} with probability ${maxProb.toFixed(4)}`);

        // Clean up tensors
        inputTensor.dispose();
        prediction.dispose();

        return { predictedClass, probability: maxProb };
    } catch (error) {
        console.error('Error running CNN prediction:', error);
        return { error: error.message };
    }
}
