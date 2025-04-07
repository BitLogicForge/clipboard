// Visualization module for perceptron demo with enhanced visuals using P5.js

// Canvas references
let plotCanvas, plotCtx;
let diagramCanvas, diagramCtx;
let weightsCanvas, weightsCtx;
let p5Instance;

// Color palette for modern look
const COLORS = {
    background: '#f8f9fa',
    primary: '#6c5ce7',
    secondary: '#00cec9',
    accent: '#fd79a8',
    light: '#dfe6e9',
    dark: '#2d3436',
    class0: '#0984e3',
    class1: '#e17055',
    correctLine: 'rgba(100, 100, 100, 0.8)',
    learnedLine: 'rgba(253, 121, 168, 0.9)',
    gridLines: 'rgba(236, 240, 241, 0.8)',
    neuron: '#6c5ce7',
    connections: '#a29bfe',
    text: '#2d3436'
};

// Initialize canvases after DOM is loaded
function initCanvases() {
    // Plot canvas for main visualization
    plotCanvas = document.getElementById("plot-canvas");
    if (plotCanvas) {
        plotCtx = plotCanvas.getContext("2d");
        // Set high-resolution canvas
        setupHighResCanvas(plotCanvas);
    }

    // Diagram canvas for perceptron explanation
    diagramCanvas = document.getElementById("perceptron-diagram");
    if (diagramCanvas) {
        diagramCtx = diagramCanvas.getContext("2d");
        // Set high-resolution canvas
        setupHighResCanvas(diagramCanvas);

        // Initialize P5.js sketch for the perceptron diagram
        initP5PerceptronDiagram();
    }

    // Weights canvas for weights visualization
    weightsCanvas = document.getElementById("weights-canvas");
    if (weightsCanvas) {
        weightsCtx = weightsCanvas.getContext("2d");
        // Set high-resolution canvas
        setupHighResCanvas(weightsCanvas);
    }
}

// Draw the perceptron diagram (fallback if P5.js isn't working)
function drawPerceptronDiagram() {
    // If P5 is working, we don't need this function to do anything
    if (p5Instance) {
        console.log("P5 instance is already managing the perceptron diagram");
        return;
    }

    // If P5 isn't working, we'll use this as a fallback
    if (!diagramCanvas || !diagramCtx) return;

    const width = diagramCanvas.width;
    const height = diagramCanvas.height;

    // Clear canvas
    diagramCtx.fillStyle = COLORS.background;
    diagramCtx.fillRect(0, 0, width, height);

    // Draw grid
    diagramCtx.strokeStyle = COLORS.gridLines;
    diagramCtx.lineWidth = 1;

    // Vertical lines
    for (let x = 0; x < width; x += 20) {
        diagramCtx.beginPath();
        diagramCtx.moveTo(x, 0);
        diagramCtx.lineTo(x, height);
        diagramCtx.stroke();
    }

    // Horizontal lines
    for (let y = 0; y < height; y += 20) {
        diagramCtx.beginPath();
        diagramCtx.moveTo(0, y);
        diagramCtx.lineTo(width, y);
        diagramCtx.stroke();
    }

    // Draw neurons
    const input1Pos = { x: 50, y: 70 };
    const input2Pos = { x: 50, y: 130 };
    const neuronPos = { x: 300, y: 100 };
    const outputPos = { x: 450, y: 100 };

    drawNeuron(diagramCtx, input1Pos.x, input1Pos.y, 25, COLORS.primary, "x₁");
    drawNeuron(diagramCtx, input2Pos.x, input2Pos.y, 25, COLORS.primary, "x₂");
    drawNeuron(diagramCtx, neuronPos.x, neuronPos.y, 35, COLORS.primary, "Σ");
    drawNeuron(diagramCtx, outputPos.x, outputPos.y, 25, COLORS.secondary, "ŷ");

    // Draw connections
    diagramCtx.strokeStyle = COLORS.connections;
    diagramCtx.lineWidth = 3;

    // Input 1 to neuron
    diagramCtx.beginPath();
    diagramCtx.moveTo(input1Pos.x + 25, input1Pos.y);
    diagramCtx.lineTo(neuronPos.x - 35, neuronPos.y - 10);
    diagramCtx.stroke();

    // Input 2 to neuron
    diagramCtx.beginPath();
    diagramCtx.moveTo(input2Pos.x + 25, input2Pos.y);
    diagramCtx.lineTo(neuronPos.x - 35, neuronPos.y + 10);
    diagramCtx.stroke();

    // Neuron to output
    diagramCtx.beginPath();
    diagramCtx.moveTo(neuronPos.x + 35, neuronPos.y);
    diagramCtx.lineTo(outputPos.x - 25, outputPos.y);
    diagramCtx.stroke();

    // Draw bias connection
    diagramCtx.setLineDash([5, 3]);
    diagramCtx.beginPath();
    diagramCtx.moveTo(neuronPos.x, neuronPos.y + 35);
    diagramCtx.lineTo(neuronPos.x, neuronPos.y + 20);
    diagramCtx.stroke();
    diagramCtx.setLineDash([]);

    // Add labels
    diagramCtx.fillStyle = COLORS.dark;
    diagramCtx.font = "16px Poppins, sans-serif";
    diagramCtx.textAlign = "center";

    // Weight labels
    diagramCtx.fillText("w₁", 150, 70);
    diagramCtx.fillText("w₂", 150, 130);

    // Bias label
    diagramCtx.fillText("bias", 300, 150);

    // Formula
    diagramCtx.textAlign = "left";
    diagramCtx.fillText("ŷ = σ(w₁x₁ + w₂x₂ + b)", 350, 50);
}

// Helper function to draw neurons for the fallback diagram
function drawNeuron(ctx, x, y, radius, color, label) {
    // Shadow
    ctx.fillStyle = "rgba(0, 0, 0, 0.2)";
    ctx.beginPath();
    ctx.arc(x + 5, y + 5, radius, 0, Math.PI * 2);
    ctx.fill();

    // Neuron body
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();

    // Highlight
    ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
    ctx.beginPath();
    ctx.arc(x - radius / 3, y - radius / 3, radius / 1.5, 0, Math.PI * 2);
    ctx.fill();

    // Label
    ctx.fillStyle = "white";
    ctx.font = radius + "px Poppins, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, x, y);
}

// Helper function to setup high resolution canvas
function setupHighResCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
}

// Initialize P5.js sketch for perceptron diagram
function initP5PerceptronDiagram() {
    // Create P5 sketch for the perceptron diagram
    const sketch = function (p) {
        // Neuron positions
        const input1Pos = { x: 50, y: 70 };
        const input2Pos = { x: 50, y: 130 };
        const neuronPos = { x: 300, y: 100 };
        const outputPos = { x: 450, y: 100 };

        // Particles for animation
        let particles = [];

        p.setup = function () {
            // Create canvas that exactly fits inside the diagramCanvas element
            const rect = diagramCanvas.getBoundingClientRect();
            const canvas = p.createCanvas(rect.width, rect.height);
            canvas.parent(diagramCanvas.parentElement);

            // Replace the canvas element with P5's canvas
            diagramCanvas.parentNode.replaceChild(canvas.elt, diagramCanvas);

            // Initialize particles
            createParticles();

            // Set frame rate to smooth animation
            p.frameRate(30);
        };

        p.draw = function () {
            p.background(248, 249, 250);

            // Draw grid
            drawGrid();

            // Draw connections
            drawConnections();

            // Update and draw particles
            updateParticles();

            // Draw neurons
            drawNeuron(input1Pos.x, input1Pos.y, 25, p.color(108, 92, 231), "x₁");
            drawNeuron(input2Pos.x, input2Pos.y, 25, p.color(108, 92, 231), "x₂");

            // Draw the main neuron with pulsing effect
            const pulseSize = 3 * Math.sin(p.frameCount * 0.05);
            drawNeuron(neuronPos.x, neuronPos.y, 35 + pulseSize, p.color(108, 92, 231), "Σ");

            // Draw output neuron
            drawNeuron(outputPos.x, outputPos.y, 25, p.color(0, 206, 201), "ŷ");

            // Add labels
            addLabels();
        };

        // Helper function to draw neurons
        function drawNeuron(x, y, radius, color, label) {
            // Shadow
            p.noStroke();
            p.fill(0, 20);
            p.ellipse(x + 5, y + 5, radius * 2, radius * 2);

            // Neuron body
            p.fill(color);
            p.ellipse(x, y, radius * 2, radius * 2);

            // Highlight
            p.fill(255, 80);
            p.ellipse(x - radius / 3, y - radius / 3, radius / 1.5, radius / 1.5);

            // Label
            p.fill(255);
            p.textSize(radius);
            p.textAlign(p.CENTER, p.CENTER);
            p.text(label, x, y);
        }

        // Draw grid background
        function drawGrid() {
            p.stroke(235, 240, 241);
            p.strokeWeight(1);

            // Vertical lines
            for (let x = 0; x < p.width; x += 20) {
                p.line(x, 0, x, p.height);
            }

            // Horizontal lines
            for (let y = 0; y < p.height; y += 20) {
                p.line(0, y, p.width, y);
            }
        }

        // Draw connections between neurons
        function drawConnections() {
            // Input 1 to neuron connection
            p.stroke(162, 155, 254);
            p.strokeWeight(3);
            p.line(input1Pos.x + 25, input1Pos.y, neuronPos.x - 35, neuronPos.y - 10);

            // Input 2 to neuron connection
            p.line(input2Pos.x + 25, input2Pos.y, neuronPos.x - 35, neuronPos.y + 10);

            // Neuron to output connection
            p.line(neuronPos.x + 35, neuronPos.y, outputPos.x - 25, outputPos.y);

            // Bias connection
            p.strokeWeight(2);
            p.drawingContext.setLineDash([5, 3]);
            p.line(neuronPos.x, neuronPos.y + 35, neuronPos.x, neuronPos.y + 20);
            p.drawingContext.setLineDash([]);
        }

        // Create animated particles
        function createParticles() {
            particles = [];

            // Create particles for input 1 to neuron
            for (let i = 0; i < 3; i++) {
                particles.push({
                    start: { x: input1Pos.x + 25, y: input1Pos.y },
                    end: { x: neuronPos.x - 35, y: neuronPos.y - 10 },
                    progress: i * 0.33,
                    speed: 0.01,
                    color: p.color(253, 121, 168)
                });
            }

            // Create particles for input 2 to neuron
            for (let i = 0; i < 3; i++) {
                particles.push({
                    start: { x: input2Pos.x + 25, y: input2Pos.y },
                    end: { x: neuronPos.x - 35, y: neuronPos.y + 10 },
                    progress: i * 0.33,
                    speed: 0.01,
                    color: p.color(253, 121, 168)
                });
            }

            // Create particles for neuron to output
            for (let i = 0; i < 3; i++) {
                particles.push({
                    start: { x: neuronPos.x + 35, y: neuronPos.y },
                    end: { x: outputPos.x - 25, y: outputPos.y },
                    progress: i * 0.33,
                    speed: 0.01,
                    color: p.color(253, 121, 168)
                });
            }
        }

        // Update and draw particles
        function updateParticles() {
            for (let particle of particles) {
                // Calculate current position
                const x = p.lerp(particle.start.x, particle.end.x, particle.progress);
                const y = p.lerp(particle.start.y, particle.end.y, particle.progress);

                // Draw particle
                p.noStroke();

                // Glow effect
                p.fill(255, 80);
                p.ellipse(x, y, 10, 10);

                // Core
                p.fill(particle.color);
                p.ellipse(x, y, 6, 6);

                // Update progress
                particle.progress += particle.speed;
                if (particle.progress > 1) {
                    particle.progress = 0;
                }
            }
        }

        // Add labels to the diagram
        function addLabels() {
            p.noStroke();
            p.fill(45, 52, 54);
            p.textSize(16);
            p.textAlign(p.CENTER, p.CENTER);

            // Weight labels
            p.text("w₁", 150, 70);
            p.text("w₂", 150, 130);

            // Bias label
            p.text("bias", 300, 150);

            // Formula
            p.textAlign(p.LEFT, p.CENTER);
            p.text("ŷ = σ(w₁x₁ + w₂x₂ + b)", 350, 50);

            // Draw activation arrow
            drawArrow(neuronPos.x + 45, neuronPos.y, outputPos.x - 35, outputPos.y);
        }

        // Draw an arrow
        function drawArrow(x1, y1, x2, y2) {
            p.push();
            p.stroke(253, 121, 168);
            p.strokeWeight(2);
            p.fill(253, 121, 168);

            const angle = p.atan2(y2 - y1, x2 - x1);
            const length = p.dist(x1, y1, x2, y1);

            p.translate(x1, y1);
            p.rotate(angle);
            p.line(0, 0, length - 10, 0);
            p.triangle(length - 2, 0, length - 12, -6, length - 12, 6);
            p.pop();
        }
    };

    try {
        // Create and save the p5 instance
        p5Instance = new p5(sketch);
        console.log("P5.js sketch initialized successfully");
    } catch (error) {
        console.error("Error initializing P5.js sketch:", error);
        // Fall back to basic Canvas API
        drawPerceptronDiagram();
    }
}

// Update weights visualization using Chart.js
function updateWeightsVisualization() {
    if (!weightsCanvas) return;

    const w1 = parseFloat(document.getElementById("weight1").value);
    const w2 = parseFloat(document.getElementById("weight2").value);
    const b = parseFloat(document.getElementById("bias").value);

    document.getElementById("weight1-value").textContent = w1.toFixed(1);
    document.getElementById("weight2-value").textContent = w2.toFixed(1);
    document.getElementById("bias-value").textContent = b.toFixed(1);

    // Get the existing chart instance if it exists
    let weightsChart = Chart.getChart(weightsCanvas);

    // If a chart already exists, destroy it to create a new one
    if (weightsChart) {
        weightsChart.destroy();
    }

    // Calculate decision boundary line
    const slope = -w1 / w2;
    const intercept = -b / w2;

    // Generate points for the decision boundary line
    const linePoints = [];
    for (let x = -2; x <= 2; x += 0.1) {
        if (Math.abs(w2) < 0.001) {
            // Handle near-vertical lines (when w2 is close to 0)
            const x0 = -b / w1;
            linePoints.push({
                x: x0,
                y: x
            });
        } else {
            // Regular case: y = slope * x + intercept
            linePoints.push({
                x: x,
                y: slope * x + intercept
            });
        }
    }

    // Generate sample points
    const samplePoints = [
        { x: 0.5, y: 0.7, label: 1 },
        { x: -0.8, y: 0.2, label: 0 },
        { x: 0.7, y: -0.5, label: 0 },
        { x: -0.3, y: -0.8, label: 0 },
        { x: -0.5, y: 0.6, label: 1 },
        { x: 1.2, y: 0.3, label: 1 },
        { x: -1.0, y: -0.2, label: 0 }
    ];

    // Split points by class and add classification prediction
    const class0Points = [];
    const class1Points = [];
    const misclassifiedPoints = [];

    for (const point of samplePoints) {
        const prediction = w1 * point.x + w2 * point.y + b > 0 ? 1 : 0;
        const isCorrect = prediction === point.label;

        if (!isCorrect) {
            misclassifiedPoints.push({
                x: point.x,
                y: point.y,
                label: point.label,
                prediction: prediction
            });
        } else if (point.label === 0) {
            class0Points.push({
                x: point.x,
                y: point.y
            });
        } else {
            class1Points.push({
                x: point.x,
                y: point.y
            });
        }
    }

    // Create chart
    try {
        weightsChart = new Chart(weightsCtx, {
            type: 'scatter',
            data: {
                datasets: [
                    // Decision boundary
                    {
                        label: 'Decision Boundary',
                        data: linePoints,
                        showLine: true,
                        fill: false,
                        borderColor: COLORS.accent,
                        borderWidth: 3,
                        pointRadius: 0,
                        tension: 0.1
                    },
                    // Class 0 points
                    {
                        label: 'Class 0',
                        data: class0Points,
                        backgroundColor: COLORS.class0,
                        pointRadius: 8,
                        pointHoverRadius: 10
                    },
                    // Class 1 points
                    {
                        label: 'Class 1',
                        data: class1Points,
                        backgroundColor: COLORS.class1,
                        pointRadius: 8,
                        pointHoverRadius: 10
                    },
                    // Misclassified points
                    {
                        label: 'Misclassified',
                        data: misclassifiedPoints,
                        backgroundColor: 'rgba(0, 0, 0, 0.5)',
                        pointRadius: 8,
                        pointHoverRadius: 10,
                        pointStyle: 'crossRot',
                        borderColor: 'black',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 0 // Disable animations
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'center',
                        min: -2,
                        max: 2,
                        title: {
                            display: true,
                            text: 'x₁'
                        },
                        grid: {
                            color: COLORS.gridLines
                        }
                    },
                    y: {
                        type: 'linear',
                        position: 'center',
                        min: -2,
                        max: 2,
                        title: {
                            display: true,
                            text: 'x₂'
                        },
                        grid: {
                            color: COLORS.gridLines
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                const point = context.raw;
                                if (context.datasetIndex === 0) {
                                    return `Decision Boundary: ${w1.toFixed(1)}x₁ + ${w2.toFixed(1)}x₂ + ${b.toFixed(1)} = 0`;
                                } else if (context.datasetIndex === 3) {
                                    return [
                                        `Actual: Class ${point.label}`,
                                        `Predicted: Class ${point.prediction}`,
                                        `Coordinates: (${point.x.toFixed(1)}, ${point.y.toFixed(1)})`
                                    ];
                                } else {
                                    return `Coordinates: (${point.x.toFixed(1)}, ${point.y.toFixed(1)})`;
                                }
                            }
                        }
                    }
                }
            }
        });
        console.log("Weights visualization chart created successfully");
    } catch (error) {
        console.error("Error creating weights chart:", error);
        // Display error message on canvas
        if (weightsCtx) {
            weightsCtx.fillStyle = "#f8f9fa";
            weightsCtx.fillRect(0, 0, weightsCanvas.width, weightsCanvas.height);
            weightsCtx.fillStyle = "#e74c3c";
            weightsCtx.font = "16px Poppins, sans-serif";
            weightsCtx.fillText("Error rendering chart. Please check console.", 20, 50);
        }
    }
}

// Update progress bar with animation
function updateProgressBar(percentage) {
    const progressBar = document.getElementById("train-progress");
    if (progressBar) {
        progressBar.style.width = percentage + "%";

        // Add color transition based on progress
        if (percentage < 30) {
            progressBar.style.backgroundColor = "#e74c3c"; // Red
        } else if (percentage < 70) {
            progressBar.style.backgroundColor = "#f39c12"; // Orange
        } else {
            progressBar.style.backgroundColor = "#2ecc71"; // Green
        }
    }
}

// Draw the data points on canvas with Chart.js
function drawData(trainingData, trainingLabels, slope, intercept) {
    if (!plotCanvas) return;

    const plotContainer = plotCanvas.parentNode;

    // Add a unique ID to the canvas for Chart.js
    const canvasId = "perceptron-plot";
    plotCanvas.id = canvasId;

    // Get the existing chart instance if it exists
    let plotChart = Chart.getChart(plotCanvas);

    // If a chart already exists, destroy it to create a new one
    if (plotChart) {
        plotChart.destroy();
    }

    // Prepare data for Chart.js
    const width = 500; // Normalized width to match the original implementation
    const height = 500; // Normalized height to match the original implementation

    // Generate true decision boundary line points
    const targetLinePoints = [];
    for (let x = 0; x < 1; x += 0.01) {
        targetLinePoints.push({
            x: x,
            y: (slope * x * width + intercept) / height
        });
    }

    // Split data into class 0 and class 1 points
    const class0Points = [];
    const class1Points = [];

    for (let i = 0; i < trainingData.length; i++) {
        const point = {
            x: trainingData[i][0],
            y: trainingData[i][1]
        };

        if (trainingLabels[i] === 0) {
            class0Points.push(point);
        } else {
            class1Points.push(point);
        }
    }

    // Create chart
    try {
        plotChart = new Chart(plotCtx, {
            type: 'scatter',
            data: {
                datasets: [
                    // True decision boundary
                    {
                        label: 'True Decision Boundary',
                        data: targetLinePoints,
                        showLine: true,
                        fill: false,
                        borderColor: COLORS.correctLine,
                        borderWidth: 3,
                        pointRadius: 0
                    },
                    // Class 0 points
                    {
                        label: 'Class 0',
                        data: class0Points,
                        backgroundColor: COLORS.class0,
                        pointRadius: 6,
                        pointHoverRadius: 8
                    },
                    // Class 1 points
                    {
                        label: 'Class 1',
                        data: class1Points,
                        backgroundColor: COLORS.class1,
                        pointRadius: 6,
                        pointHoverRadius: 8
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 0 // Disable animations
                },
                scales: {
                    x: {
                        type: 'linear',
                        min: 0,
                        max: 1,
                        title: {
                            display: true,
                            text: 'x₁'
                        },
                        grid: {
                            color: COLORS.gridLines
                        }
                    },
                    y: {
                        type: 'linear',
                        min: 0,
                        max: 1,
                        title: {
                            display: true,
                            text: 'x₂'
                        },
                        grid: {
                            color: COLORS.gridLines
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                const point = context.raw;
                                if (context.datasetIndex === 0) {
                                    return 'True Decision Boundary';
                                } else {
                                    return `Coordinates: (${point.x.toFixed(2)}, ${point.y.toFixed(2)})`;
                                }
                            }
                        }
                    }
                }
            }
        });
        console.log("Data visualization chart created successfully");
    } catch (error) {
        console.error("Error creating data chart:", error);
        // Display error message on canvas
        if (plotCtx) {
            plotCtx.fillStyle = "#f8f9fa";
            plotCtx.fillRect(0, 0, plotCanvas.width, plotCanvas.height);
            plotCtx.fillStyle = "#e74c3c";
            plotCtx.font = "16px Poppins, sans-serif";
            plotCtx.fillText("Error rendering chart. Please check console.", 20, 50);
        }
    }
}

// Visualize the model's predictions with Chart.js
function visualizeModel(model, trainingData, trainingLabels, slope, intercept) {
    if (!model || !plotCanvas) return;

    // Get the existing chart instance
    let plotChart = Chart.getChart(plotCanvas);

    // If no chart exists, create one
    if (!plotChart) {
        drawData(trainingData, trainingLabels, slope, intercept);
        plotChart = Chart.getChart(plotCanvas);
        if (!plotChart) return; // If chart creation failed, exit
    }

    try {
        // Extract model weights and bias
        const weights = model.layers[0].getWeights()[0].dataSync();
        const bias = model.layers[0].getWeights()[1].dataSync()[0];

        const w1 = weights[0];
        const w2 = weights[1];

        // Calculate learned decision boundary line
        if (Math.abs(w2) > 0.001) {
            // Avoid division by zero
            const learnedSlope = -w1 / w2;
            const learnedIntercept = -bias / w2;

            // Generate points for the learned decision boundary
            const learnedLinePoints = [];
            for (let x = 0; x < 1; x += 0.01) {
                learnedLinePoints.push({
                    x: x,
                    y: learnedSlope * x + learnedIntercept
                });
            }

            // Check if we already have a learned boundary dataset
            let learnedBoundaryIndex = -1;
            for (let i = 0; i < plotChart.data.datasets.length; i++) {
                if (plotChart.data.datasets[i].label === 'Learned Boundary') {
                    learnedBoundaryIndex = i;
                    break;
                }
            }

            if (learnedBoundaryIndex !== -1) {
                // Update existing dataset
                plotChart.data.datasets[learnedBoundaryIndex].data = learnedLinePoints;
            } else {
                // Add new dataset
                plotChart.data.datasets.push({
                    label: 'Learned Boundary',
                    data: learnedLinePoints,
                    showLine: true,
                    fill: false,
                    borderColor: COLORS.learnedLine,
                    borderWidth: 3,
                    borderDash: [5, 5],
                    pointRadius: 0
                });
            }

            // Disable animations when updating the chart
            const originalAnimationOptions = plotChart.options.animation;
            plotChart.options.animation = {
                duration: 0 // Disable animations
            };

            // Update the chart
            plotChart.update();

            // Restore original animation options if needed for future updates
            // plotChart.options.animation = originalAnimationOptions;

            // Display the learned formula
            document.getElementById("model-info").innerHTML =
                `Learned Formula: ${w1.toFixed(2)}x₁ + ${w2.toFixed(2)}x₂ + ${bias.toFixed(2)} = 0`;
        }
    } catch (error) {
        console.error("Error visualizing model:", error);
    }
}

// Clean up animations when switching pages
function cleanupAnimations() {
    if (p5Instance) {
        try {
            p5Instance.remove();
        } catch (error) {
            console.error("Error removing P5 instance:", error);
        }
    }
}
