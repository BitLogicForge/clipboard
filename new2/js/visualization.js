// Visualization module for perceptron demo

// Canvas references
let plotCanvas, plotCtx;
let diagramCanvas, diagramCtx;
let weightsCanvas, weightsCtx;

// Initialize canvases after DOM is loaded
function initCanvases() {
    // Plot canvas for main visualization
    plotCanvas = document.getElementById("plot-canvas");
    plotCtx = plotCanvas.getContext("2d");

    // Diagram canvas for perceptron explanation
    diagramCanvas = document.getElementById("perceptron-diagram");
    diagramCtx = diagramCanvas.getContext("2d");

    // Weights canvas for weights visualization
    weightsCanvas = document.getElementById("weights-canvas");
    weightsCtx = weightsCanvas.getContext("2d");
}

// Draw perceptron diagram
function drawPerceptronDiagram() {
    if (!diagramCtx) return;

    diagramCtx.clearRect(0, 0, diagramCanvas.width, diagramCanvas.height);

    // Draw inputs
    diagramCtx.fillStyle = "#3498db";
    diagramCtx.beginPath();
    diagramCtx.arc(50, 70, 20, 0, Math.PI * 2);
    diagramCtx.fill();
    diagramCtx.fillStyle = "white";
    diagramCtx.font = "16px Arial";
    diagramCtx.fillText("x₁", 45, 75);

    diagramCtx.fillStyle = "#3498db";
    diagramCtx.beginPath();
    diagramCtx.arc(50, 130, 20, 0, Math.PI * 2);
    diagramCtx.fill();
    diagramCtx.fillStyle = "white";
    diagramCtx.font = "16px Arial";
    diagramCtx.fillText("x₂", 45, 135);

    // Draw neuron
    diagramCtx.fillStyle = "#e74c3c";
    diagramCtx.beginPath();
    diagramCtx.arc(300, 100, 30, 0, Math.PI * 2);
    diagramCtx.fill();
    diagramCtx.fillStyle = "white";
    diagramCtx.font = "16px Arial";
    diagramCtx.fillText("Σ", 295, 105);

    // Draw output
    diagramCtx.fillStyle = "#2ecc71";
    diagramCtx.beginPath();
    diagramCtx.arc(450, 100, 20, 0, Math.PI * 2);
    diagramCtx.fill();
    diagramCtx.fillStyle = "white";
    diagramCtx.font = "14px Arial";
    diagramCtx.fillText("0/1", 443, 105);

    // Draw connections
    diagramCtx.strokeStyle = "#34495e";
    diagramCtx.lineWidth = 2;

    // Input 1 to Neuron
    diagramCtx.beginPath();
    diagramCtx.moveTo(70, 70);
    diagramCtx.lineTo(270, 90);
    diagramCtx.stroke();

    // Input 2 to Neuron
    diagramCtx.beginPath();
    diagramCtx.moveTo(70, 130);
    diagramCtx.lineTo(270, 110);
    diagramCtx.stroke();

    // Neuron to Output
    diagramCtx.beginPath();
    diagramCtx.moveTo(330, 100);
    diagramCtx.lineTo(430, 100);
    diagramCtx.stroke();

    // Add weight labels
    diagramCtx.fillStyle = "#34495e";
    diagramCtx.font = "14px Arial";
    diagramCtx.fillText("w₁", 150, 70);
    diagramCtx.fillText("w₂", 150, 130);

    // Add activation label
    diagramCtx.fillText("Activation", 350, 90);

    // Add bias
    diagramCtx.fillText("+ bias", 300, 150);
    diagramCtx.beginPath();
    diagramCtx.moveTo(300, 140);
    diagramCtx.lineTo(300, 120);
    diagramCtx.stroke();
}

// Draw weights visualization
function updateWeightsVisualization() {
    if (!weightsCtx) return;

    const w1 = parseFloat(document.getElementById("weight1").value);
    const w2 = parseFloat(document.getElementById("weight2").value);
    const b = parseFloat(document.getElementById("bias").value);

    document.getElementById("weight1-value").textContent = w1.toFixed(1);
    document.getElementById("weight2-value").textContent = w2.toFixed(1);
    document.getElementById("bias-value").textContent = b.toFixed(1);

    weightsCtx.clearRect(0, 0, weightsCanvas.width, weightsCanvas.height);

    const canvasWidth = weightsCanvas.width;
    const canvasHeight = weightsCanvas.height;

    // Calculate line equation: w1*x + w2*y + b = 0 -> y = (-w1/w2)x - b/w2
    let slope = -w1 / w2;
    let intercept = -b / w2;

    // Draw grid
    weightsCtx.strokeStyle = "#e0e0e0";
    weightsCtx.lineWidth = 1;

    // Draw horizontal grid lines
    for (let y = 0; y <= canvasHeight; y += 50) {
        weightsCtx.beginPath();
        weightsCtx.moveTo(0, y);
        weightsCtx.lineTo(canvasWidth, y);
        weightsCtx.stroke();
    }

    // Draw vertical grid lines
    for (let x = 0; x <= canvasWidth; x += 50) {
        weightsCtx.beginPath();
        weightsCtx.moveTo(x, 0);
        weightsCtx.lineTo(x, canvasHeight);
        weightsCtx.stroke();
    }

    // Draw axis
    weightsCtx.strokeStyle = "#000";
    weightsCtx.lineWidth = 2;

    // X-axis
    weightsCtx.beginPath();
    weightsCtx.moveTo(0, canvasHeight / 2);
    weightsCtx.lineTo(canvasWidth, canvasHeight / 2);
    weightsCtx.stroke();

    // Y-axis
    weightsCtx.beginPath();
    weightsCtx.moveTo(canvasWidth / 2, 0);
    weightsCtx.lineTo(canvasWidth / 2, canvasHeight);
    weightsCtx.stroke();

    // Draw decision boundary
    weightsCtx.strokeStyle = "#e74c3c";
    weightsCtx.lineWidth = 3;

    // Transform coordinates to make center of canvas the origin
    function transformX(x) {
        return (x * canvasWidth) / 4 + canvasWidth / 2;
    }

    function transformY(y) {
        return canvasHeight / 2 - (y * canvasHeight) / 4;
    }

    // Draw line
    weightsCtx.beginPath();

    if (Math.abs(w2) < 0.1) {
        // Handle nearly horizontal line (when w2 is close to 0)
        let x0 = -b / w1;
        weightsCtx.moveTo(transformX(x0), 0);
        weightsCtx.lineTo(transformX(x0), canvasHeight);
    } else {
        // Regular case
        let x1 = -2;
        let y1 = slope * x1 + intercept;
        let x2 = 2;
        let y2 = slope * x2 + intercept;

        weightsCtx.moveTo(transformX(x1), transformY(y1));
        weightsCtx.lineTo(transformX(x2), transformY(y2));
    }

    weightsCtx.stroke();

    // Add some sample points
    const samplePoints = [
        { x: 0.5, y: 0.7, label: 1 },
        { x: -0.8, y: 0.2, label: 0 },
        { x: 0.7, y: -0.5, label: 0 },
        { x: -0.3, y: -0.8, label: 0 },
        { x: -0.5, y: 0.6, label: 1 },
    ];

    for (let point of samplePoints) {
        // Predict using perceptron formula: w1*x + w2*y + b > 0 ? 1 : 0
        let prediction = w1 * point.x + w2 * point.y + b > 0 ? 1 : 0;

        weightsCtx.fillStyle = prediction === 1 ? "#e74c3c" : "#3498db";
        weightsCtx.beginPath();
        weightsCtx.arc(
            transformX(point.x),
            transformY(point.y),
            8,
            0,
            Math.PI * 2
        );
        weightsCtx.fill();

        // Draw a black border if misclassified
        if (prediction !== point.label) {
            weightsCtx.strokeStyle = "#000";
            weightsCtx.lineWidth = 2;
            weightsCtx.stroke();
        }
    }

    // Add legend
    weightsCtx.font = "12px Arial";
    weightsCtx.fillStyle = "#000";
    weightsCtx.fillText(
        "Decision Boundary: " +
        w1.toFixed(1) +
        "x + " +
        w2.toFixed(1) +
        "y + " +
        b.toFixed(1) +
        " = 0",
        10,
        20
    );
}

// Update progress bar
function updateProgressBar(percentage) {
    document.getElementById("train-progress").style.width = percentage + "%";
}

// Draw the data points on canvas
function drawData(trainingData, trainingLabels, slope, intercept) {
    if (!plotCtx) return;

    const width = plotCanvas.width;
    const height = plotCanvas.height;

    plotCtx.clearRect(0, 0, width, height);

    // Draw the target line
    plotCtx.beginPath();
    plotCtx.moveTo(0, intercept);
    plotCtx.lineTo(width, slope * width + intercept);
    plotCtx.strokeStyle = "rgba(100, 100, 100, 0.8)";
    plotCtx.lineWidth = 2;
    plotCtx.stroke();

    // Draw data points
    for (let i = 0; i < trainingData.length; i++) {
        const x = trainingData[i][0] * width;
        const y = trainingData[i][1] * height;
        const label = trainingLabels[i];

        plotCtx.beginPath();
        plotCtx.arc(x, y, 5, 0, Math.PI * 2);
        plotCtx.fillStyle = label === 0 ? "blue" : "red";
        plotCtx.fill();
    }
}

// Visualize the model's predictions
function visualizeModel(model, trainingData, trainingLabels, slope, intercept) {
    if (!plotCtx || !model) return;

    const width = plotCanvas.width;
    const height = plotCanvas.height;

    plotCtx.clearRect(0, 0, width, height);

    // Draw the target line
    plotCtx.beginPath();
    plotCtx.moveTo(0, intercept);
    plotCtx.lineTo(width, slope * width + intercept);
    plotCtx.strokeStyle = "rgba(100, 100, 100, 0.8)";
    plotCtx.lineWidth = 2;
    plotCtx.stroke();

    // Create prediction grid
    const resolution = 50;
    const gridSize = width / resolution;
    const predictions = [];
    const inputs = [];

    for (let x = 0; x < width; x += gridSize) {
        for (let y = 0; y < height; y += gridSize) {
            inputs.push([x / width, y / height]);
        }
    }

    // Make predictions
    const xs = tf.tensor2d(inputs);
    const preds = model.predict(xs);
    const values = preds.dataSync();

    // Draw prediction grid
    for (let i = 0; i < inputs.length; i++) {
        const x = inputs[i][0] * width;
        const y = inputs[i][1] * height;
        const prediction = values[i];

        plotCtx.fillStyle = `rgba(${prediction > 0.5 ? "255, 0, 0" : "0, 0, 255"
            }, 0.1)`;
        plotCtx.fillRect(x, y, gridSize, gridSize);
    }

    // Draw learned decision boundary
    try {
        // Extract weights and bias from the model
        const weights = model.layers[0].getWeights()[0].dataSync();
        const bias = model.layers[0].getWeights()[1].dataSync()[0];

        const w1 = weights[0];
        const w2 = weights[1];

        // Calculate learned decision boundary line: w1*x + w2*y + b = 0 -> y = (-w1/w2)x - b/w2
        if (Math.abs(w2) > 0.0001) {
            // Avoid division by zero
            const learnedSlope = -w1 / w2;
            const learnedIntercept = -bias / w2;

            // Convert to pixel coordinates from 0-1 range
            const pixelLeanedIntercept = learnedIntercept * height;

            // Draw the learned decision boundary (where sigmoid output = 0.5)
            plotCtx.beginPath();
            plotCtx.moveTo(0, pixelLeanedIntercept);
            plotCtx.lineTo(width, learnedSlope * width + pixelLeanedIntercept);
            plotCtx.strokeStyle = "rgba(255, 165, 0, 0.8)"; // Orange color
            plotCtx.lineWidth = 2;
            plotCtx.stroke();

            // Add a small label
            plotCtx.fillStyle = "rgba(255, 165, 0, 1)";
            plotCtx.font = "12px Arial";
            plotCtx.fillText("Learned Boundary", 10, 20);
        }
    } catch (error) {
        console.error("Error drawing learned boundary:", error);
    }

    // Draw data points on top
    for (let i = 0; i < trainingData.length; i++) {
        const x = trainingData[i][0] * width;
        const y = trainingData[i][1] * height;
        const label = trainingLabels[i];

        plotCtx.beginPath();
        plotCtx.arc(x, y, 5, 0, Math.PI * 2);
        plotCtx.fillStyle = label === 0 ? "blue" : "red";
        plotCtx.fill();
        plotCtx.strokeStyle = "black";
        plotCtx.lineWidth = 1;
        plotCtx.stroke();
    }

    xs.dispose();
    preds.dispose();
}
