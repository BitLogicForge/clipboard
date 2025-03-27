// CNN Visualization module

// Canvas references
let inputCanvas, inputCtx;
let convolutionCanvas, convolutionCtx;
let convOutputCanvas, convOutputCtx;
let reluOutputCanvas, reluOutputCtx;
let poolingOutputCanvas, poolingOutputCtx;
let architectureCanvas, architectureCtx;

// Drawing state
let isDrawing = false;
let drawMode = true;

// Kernel definitions
const kernels = {
    edge: [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ],
    blur: [
        [0.1, 0.1, 0.1],
        [0.1, 0.2, 0.1],
        [0.1, 0.1, 0.1]
    ],
    sharpen: [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ],
    custom: [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
};

// Animation state
let animationId = null;
let animationStep = 0;

// Initialize canvases and event listeners
function initCNNVisualization() {
    // Get canvas references
    inputCanvas = document.getElementById('input-canvas');
    inputCtx = inputCanvas.getContext('2d');

    convolutionCanvas = document.getElementById('convolution-canvas');
    convolutionCtx = convolutionCanvas.getContext('2d');

    convOutputCanvas = document.getElementById('conv-output');
    convOutputCtx = convOutputCanvas.getContext('2d');

    reluOutputCanvas = document.getElementById('relu-output');
    reluOutputCtx = reluOutputCanvas.getContext('2d');

    poolingOutputCanvas = document.getElementById('pooling-output');
    poolingOutputCtx = poolingOutputCanvas.getContext('2d');

    architectureCanvas = document.getElementById('cnn-architecture');
    architectureCtx = architectureCanvas.getContext('2d');

    // Initialize canvases with white backgrounds
    clearCanvas(inputCtx, inputCanvas.width, inputCanvas.height);
    clearCanvas(convolutionCtx, convolutionCanvas.width, convolutionCanvas.height);
    clearCanvas(convOutputCtx, convOutputCanvas.width, convOutputCanvas.height);
    clearCanvas(reluOutputCtx, reluOutputCanvas.width, reluOutputCanvas.height);
    clearCanvas(poolingOutputCtx, poolingOutputCanvas.width, poolingOutputCanvas.height);

    // Set up event listeners for the input canvas
    inputCanvas.addEventListener('mousedown', startDrawing);
    inputCanvas.addEventListener('mousemove', draw);
    inputCanvas.addEventListener('mouseup', stopDrawing);
    inputCanvas.addEventListener('mouseout', stopDrawing);

    // Set up button event listeners
    document.getElementById('draw-btn').addEventListener('click', toggleDrawMode);
    document.getElementById('clear-btn').addEventListener('click', clearInputCanvas);
    document.getElementById('sample-btn').addEventListener('click', loadSampleImage);
    document.getElementById('apply-filter-btn').addEventListener('click', applyFilter);
    document.getElementById('animate-btn').addEventListener('click', toggleConvolutionAnimation);

    // Kernel selection listeners
    document.querySelectorAll('input[name="kernel"]').forEach(radio => {
        radio.addEventListener('change', handleKernelSelection);
    });

    // Custom kernel input change
    document.querySelectorAll('#custom-kernel-container input').forEach(input => {
        input.addEventListener('change', updateCustomKernel);
    });

    // Draw CNN architecture
    drawCNNArchitecture();
}

// Clear canvas with white background
function clearCanvas(ctx, width, height) {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);
}

// Start drawing on mouse down
function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

// Draw on the canvas
function draw(e) {
    if (!isDrawing) return;

    const rect = inputCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    inputCtx.lineWidth = 10;
    inputCtx.lineCap = 'round';
    inputCtx.strokeStyle = drawMode ? 'black' : 'white';

    inputCtx.lineTo(x, y);
    inputCtx.stroke();
    inputCtx.beginPath();
    inputCtx.moveTo(x, y);
}

// Stop drawing
function stopDrawing() {
    isDrawing = false;
    inputCtx.beginPath();
}

// Toggle between draw and erase modes
function toggleDrawMode() {
    drawMode = !drawMode;
    document.getElementById('draw-btn').textContent = drawMode ? 'Draw Mode' : 'Erase Mode';
}

// Clear the input canvas
function clearInputCanvas() {
    clearCanvas(inputCtx, inputCanvas.width, inputCanvas.height);
    clearCanvas(convOutputCtx, convOutputCanvas.width, convOutputCanvas.height);
    clearCanvas(reluOutputCtx, reluOutputCanvas.width, reluOutputCanvas.height);
    clearCanvas(poolingOutputCtx, poolingOutputCanvas.width, poolingOutputCanvas.height);
    stopConvolutionAnimation();
}

// Load a sample image
function loadSampleImage() {
    const img = new Image();
    img.onload = function () {
        clearCanvas(inputCtx, inputCanvas.width, inputCanvas.height);
        // Scale image to fit the canvas while maintaining aspect ratio
        const scale = Math.min(inputCanvas.width / img.width, inputCanvas.height / img.height);
        const x = (inputCanvas.width - img.width * scale) / 2;
        const y = (inputCanvas.height - img.height * scale) / 2;

        inputCtx.drawImage(img, x, y, img.width * scale, img.height * scale);
    };
    img.src = 'images/sample-digit.png'; // Fallback to a simple shape if image doesn't exist

    // Draw a simple digit-like shape if the image fails to load
    img.onerror = function () {
        clearCanvas(inputCtx, inputCanvas.width, inputCanvas.height);

        // Draw a simple "8" shape
        inputCtx.strokeStyle = 'black';
        inputCtx.lineWidth = 20;
        inputCtx.beginPath();
        inputCtx.arc(100, 70, 40, 0, Math.PI * 2);
        inputCtx.stroke();

        inputCtx.beginPath();
        inputCtx.arc(100, 130, 40, 0, Math.PI * 2);
        inputCtx.stroke();
    };
}

// Handle kernel selection
function handleKernelSelection() {
    const selectedKernel = document.querySelector('input[name="kernel"]:checked').value;
    document.getElementById('custom-kernel-container').style.display =
        selectedKernel === 'custom' ? 'block' : 'none';
}

// Update custom kernel values
function updateCustomKernel() {
    const inputs = document.querySelectorAll('#custom-kernel-container input');
    for (let i = 0; i < inputs.length; i++) {
        const row = Math.floor(i / 3);
        const col = i % 3;
        kernels.custom[row][col] = parseFloat(inputs[i].value) || 0;
    }
}

// Apply selected filter to the input image
function applyFilter() {
    // Get the selected kernel
    const selectedKernelType = document.querySelector('input[name="kernel"]:checked').value;
    const selectedKernel = kernels[selectedKernelType];

    // Get the stride and padding settings
    const stride = parseInt(document.getElementById('stride-control').value);
    const paddingType = document.getElementById('padding-control').value;

    // Apply convolution
    const inputData = getImageData(inputCtx, inputCanvas.width, inputCanvas.height);

    // Perform convolution and update output canvases
    const { convOutput, paddedInput } = applyConvolution(inputData, selectedKernel, stride, paddingType);

    // Display convolution output
    displayOutput(convOutput, convOutputCtx, convOutputCanvas.width, convOutputCanvas.height);

    // Apply ReLU activation
    const reluOutput = applyReLU(convOutput);
    displayOutput(reluOutput, reluOutputCtx, reluOutputCanvas.width, reluOutputCanvas.height);

    // Apply max pooling 
    const poolSize = 2; // 2x2 pooling
    const pooledOutput = applyMaxPooling(reluOutput, poolSize);
    displayOutput(pooledOutput, poolingOutputCtx, poolingOutputCanvas.width, poolingOutputCanvas.height);

    // Save for animation
    animationInputData = paddedInput;
    animationKernel = selectedKernel;
    animationStride = stride;

    // Visualize the convolution process
    visualizeConvolutionProcess(paddedInput, selectedKernel, convOutput, stride);
}

// Get image data from canvas
function getImageData(ctx, width, height) {
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = new Array(height);

    for (let y = 0; y < height; y++) {
        data[y] = new Array(width);
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            // Convert to grayscale if it's not already
            data[y][x] = (imageData.data[idx] + imageData.data[idx + 1] + imageData.data[idx + 2]) / 3 / 255;
        }
    }

    return data;
}

// Display output data on canvas
function displayOutput(outputData, ctx, width, height) {
    const imageData = ctx.createImageData(width, height);

    // Find min and max for normalization
    let min = Infinity;
    let max = -Infinity;

    for (let y = 0; y < outputData.length; y++) {
        for (let x = 0; x < outputData[0].length; x++) {
            min = Math.min(min, outputData[y][x]);
            max = Math.max(max, outputData[y][x]);
        }
    }

    // Normalize and set the image data
    const range = max - min;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            // Scale the output to fit the canvas
            const outY = Math.min(Math.floor(y * outputData.length / height), outputData.length - 1);
            const outX = Math.min(Math.floor(x * outputData[0].length / width), outputData[0].length - 1);

            // Normalize the value (0-1)
            const normalizedValue = range === 0 ? 0.5 : (outputData[outY][outX] - min) / range;

            // Convert to pixel value (0-255)
            const pixelValue = Math.round(normalizedValue * 255);

            const idx = (y * width + x) * 4;
            imageData.data[idx] = pixelValue;     // R
            imageData.data[idx + 1] = pixelValue; // G
            imageData.data[idx + 2] = pixelValue; // B
            imageData.data[idx + 3] = 255;        // Alpha
        }
    }

    ctx.putImageData(imageData, 0, 0);
}

// Draw CNN architecture
function drawCNNArchitecture() {
    if (!architectureCtx) return;

    architectureCtx.clearRect(0, 0, architectureCanvas.width, architectureCanvas.height);

    const width = architectureCanvas.width;
    const height = architectureCanvas.height;
    const centerY = height / 2;

    // Draw layers
    const layers = [
        { label: "Input", color: "#3498db", width: 60, height: 60, x: 50 },
        { label: "Conv", color: "#e74c3c", width: 50, height: 50, x: 150 },
        { label: "ReLU", color: "#f39c12", width: 50, height: 50, x: 230 },
        { label: "Pool", color: "#2ecc71", width: 40, height: 40, x: 310 },
        { label: "Conv", color: "#e74c3c", width: 30, height: 30, x: 390 },
        { label: "ReLU", color: "#f39c12", width: 30, height: 30, x: 460 },
        { label: "Pool", color: "#2ecc71", width: 20, height: 20, x: 530 },
        { label: "FC", color: "#9b59b6", width: 20, height: 60, x: 600 },
        { label: "Softmax", color: "#34495e", width: 15, height: 50, x: 660 },
        { label: "Output", color: "#1abc9c", width: 10, height: 40, x: 720 }
    ];

    // Draw each layer and connecting lines
    for (let i = 0; i < layers.length; i++) {
        const layer = layers[i];

        // Draw the layer
        architectureCtx.fillStyle = layer.color;
        architectureCtx.fillRect(
            layer.x - layer.width / 2,
            centerY - layer.height / 2,
            layer.width,
            layer.height
        );

        // Add label
        architectureCtx.fillStyle = "black";
        architectureCtx.font = "10px Arial";
        architectureCtx.textAlign = "center";
        architectureCtx.fillText(layer.label, layer.x, centerY + layer.height / 2 + 15);

        // Draw connecting line to next layer
        if (i < layers.length - 1) {
            architectureCtx.strokeStyle = "#7f8c8d";
            architectureCtx.beginPath();
            architectureCtx.moveTo(layer.x + layer.width / 2, centerY);
            architectureCtx.lineTo(layers[i + 1].x - layers[i + 1].width / 2, centerY);
            architectureCtx.stroke();
        }
    }
}

// Visualize the convolution process
function visualizeConvolutionProcess(inputData, kernel, outputData, stride) {
    if (!convolutionCtx) return;

    convolutionCtx.clearRect(0, 0, convolutionCanvas.width, convolutionCanvas.height);

    const padding = 10;
    const cellSize = 16;
    const kernelSize = kernel.length;

    const inputWidth = inputData[0].length;
    const inputHeight = inputData.length;

    // Draw input matrix
    const inputStartX = padding;
    const inputStartY = padding;

    // Draw the grid and values for input
    drawDataGrid(convolutionCtx, inputData, inputStartX, inputStartY, cellSize, 'Input');

    // Draw kernel
    const kernelStartX = inputStartX + inputWidth * cellSize + 60;
    const kernelStartY = padding;

    drawDataGrid(convolutionCtx, kernel, kernelStartX, kernelStartY, cellSize, 'Kernel');

    // Draw output matrix
    const outputStartX = kernelStartX + kernelSize * cellSize + 60;
    const outputStartY = padding;
    const outputWidth = outputData[0].length;
    const outputHeight = outputData.length;

    drawDataGrid(convolutionCtx, outputData, outputStartX, outputStartY, cellSize, 'Output');

    // Draw calculation example
    const exampleStartX = inputStartX;
    const exampleStartY = inputStartY + inputHeight * cellSize + 50;

    convolutionCtx.fillStyle = 'black';
    convolutionCtx.font = '14px Arial';
    convolutionCtx.fillText('Convolution Calculation Example:', exampleStartX, exampleStartY);

    // Highlight a region in the input for the example calculation
    const highlightX = Math.min(2, inputWidth - kernelSize);
    const highlightY = Math.min(2, inputHeight - kernelSize);

    // Draw highlighted area in input
    convolutionCtx.strokeStyle = 'red';
    convolutionCtx.lineWidth = 2;
    convolutionCtx.strokeRect(
        inputStartX + highlightX * cellSize,
        inputStartY + highlightY * cellSize,
        kernelSize * cellSize,
        kernelSize * cellSize
    );

    // Draw the calculation
    let calculationText = 'Output[0,0] = ';
    let sum = 0;

    for (let ky = 0; ky < kernelSize; ky++) {
        for (let kx = 0; kx < kernelSize; kx++) {
            const inputValue = inputData[highlightY + ky][highlightX + kx];
            const kernelValue = kernel[ky][kx];
            const product = inputValue * kernelValue;

            calculationText += `(${inputValue.toFixed(1)} × ${kernelValue.toFixed(1)})`;

            if (kx < kernelSize - 1 || ky < kernelSize - 1) {
                calculationText += ' + ';
            }

            sum += product;
        }
    }

    calculationText += ` = ${sum.toFixed(2)}`;

    convolutionCtx.fillStyle = 'black';
    convolutionCtx.font = '12px Arial';
    convolutionCtx.fillText(calculationText, exampleStartX, exampleStartY + 30);

    // Highlight the corresponding output cell
    const outputHighlightX = Math.floor(highlightX / stride);
    const outputHighlightY = Math.floor(highlightY / stride);

    if (outputHighlightX < outputWidth && outputHighlightY < outputHeight) {
        convolutionCtx.strokeStyle = 'red';
        convolutionCtx.strokeRect(
            outputStartX + outputHighlightX * cellSize,
            outputStartY + outputHighlightY * cellSize,
            cellSize,
            cellSize
        );
    }
}

// Draw data grid with labels
function drawDataGrid(ctx, data, startX, startY, cellSize, label) {
    const height = data.length;
    const width = data[0].length;

    // Draw label
    ctx.fillStyle = 'black';
    ctx.font = '14px Arial';
    ctx.fillText(label, startX, startY - 5);

    // Draw grid
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 1;

    for (let y = 0; y <= height; y++) {
        ctx.beginPath();
        ctx.moveTo(startX, startY + y * cellSize);
        ctx.lineTo(startX + width * cellSize, startY + y * cellSize);
        ctx.stroke();
    }

    for (let x = 0; x <= width; x++) {
        ctx.beginPath();
        ctx.moveTo(startX + x * cellSize, startY);
        ctx.lineTo(startX + x * cellSize, startY + height * cellSize);
        ctx.stroke();
    }

    // Draw values
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const value = data[y][x];

            // Use color intensity for visualization
            const intensity = Math.max(0, Math.min(255, Math.round(value * 255)));
            ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
            ctx.fillRect(startX + x * cellSize, startY + y * cellSize, cellSize, cellSize);

            // Draw text in contrasting color
            ctx.fillStyle = intensity > 128 ? 'black' : 'white';
            ctx.fillText(
                value.toFixed(1),
                startX + x * cellSize + cellSize / 2,
                startY + y * cellSize + cellSize / 2
            );
        }
    }
}

// Variables for animation
let animationInputData;
let animationKernel;
let animationStride;
let animationCurrentX = 0;
let animationCurrentY = 0;

// Toggle convolution animation
function toggleConvolutionAnimation() {
    if (animationId) {
        stopConvolutionAnimation();
    } else {
        startConvolutionAnimation();
    }
}

// Start convolution animation
function startConvolutionAnimation() {
    if (!animationInputData || !animationKernel) {
        alert('Please apply a filter first');
        return;
    }

    // Reset animation position
    animationCurrentX = 0;
    animationCurrentY = 0;

    // Stop any existing animation
    stopConvolutionAnimation();

    // Update the button text
    document.getElementById('animate-btn').textContent = 'Stop Animation';

    // Start the animation
    animateConvolutionStep();
}

// Stop convolution animation
function stopConvolutionAnimation() {
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
        document.getElementById('animate-btn').textContent = 'Animate Convolution';
    }
}

// Animate a single convolution step
function animateConvolutionStep() {
    if (!convolutionCtx || !animationInputData || !animationKernel) return;

    const inputHeight = animationInputData.length;
    const inputWidth = animationInputData[0].length;
    const kernelSize = animationKernel.length;
    const stride = animationStride || 1;

    // Calculate valid convolution area
    const outputHeight = Math.floor((inputHeight - kernelSize) / stride) + 1;
    const outputWidth = Math.floor((inputWidth - kernelSize) / stride) + 1;

    // Clear the canvas
    convolutionCtx.clearRect(0, 0, convolutionCanvas.width, convolutionCanvas.height);

    const padding = 10;
    const cellSize = 16;

    // Draw input data
    const inputStartX = padding;
    const inputStartY = padding;
    drawDataGrid(convolutionCtx, animationInputData, inputStartX, inputStartY, cellSize, 'Input');

    // Draw kernel
    const kernelStartX = inputStartX + inputWidth * cellSize + 60;
    const kernelStartY = padding;
    drawDataGrid(convolutionCtx, animationKernel, kernelStartX, kernelStartY, cellSize, 'Kernel');

    // Highlight current position on input
    convolutionCtx.strokeStyle = 'red';
    convolutionCtx.lineWidth = 2;
    convolutionCtx.strokeRect(
        inputStartX + animationCurrentX * stride * cellSize,
        inputStartY + animationCurrentY * stride * cellSize,
        kernelSize * cellSize,
        kernelSize * cellSize
    );

    // Draw calculation
    const exampleStartX = inputStartX;
    const exampleStartY = inputStartY + inputHeight * cellSize + 50;

    // Calculate the output value for this position
    let sum = 0;
    let calculationText = `Output[${animationCurrentY},${animationCurrentX}] = `;

    for (let ky = 0; ky < kernelSize; ky++) {
        for (let kx = 0; kx < kernelSize; kx++) {
            const inputY = animationCurrentY * stride + ky;
            const inputX = animationCurrentX * stride + kx;

            if (inputY < inputHeight && inputX < inputWidth) {
                const inputValue = animationInputData[inputY][inputX];
                const kernelValue = animationKernel[ky][kx];
                const product = inputValue * kernelValue;

                calculationText += `(${inputValue.toFixed(1)} × ${kernelValue.toFixed(1)})`;

                if (kx < kernelSize - 1 || ky < kernelSize - 1) {
                    calculationText += ' + ';
                }

                sum += product;
            }
        }
    }

    calculationText += ` = ${sum.toFixed(2)}`;

    convolutionCtx.fillStyle = 'black';
    convolutionCtx.font = '12px Arial';
    convolutionCtx.fillText('Convolution Animation:', exampleStartX, exampleStartY);
    convolutionCtx.fillText(calculationText, exampleStartX, exampleStartY + 30);

    // Move to the next position
    animationCurrentX++;
    if (animationCurrentX >= outputWidth) {
        animationCurrentX = 0;
        animationCurrentY++;

        if (animationCurrentY >= outputHeight) {
            animationCurrentY = 0; // Loop the animation
        }
    }

    // Continue the animation
    animationId = requestAnimationFrame(animateConvolutionStep);
}

// Wait for DOM content to load
document.addEventListener('DOMContentLoaded', initCNNVisualization);
