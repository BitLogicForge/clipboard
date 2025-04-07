// Neural Network Simulation and Activation Function Graphs
// This file handles network simulation and activation function visualizations

// Chart instances
let sigmoidChart, reluChart, tanhChart, softmaxChart;

// Color palette for consistent look with perceptron
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

// Initialize function graph visualizations using Chart.js
function initActivationFunctionGraphs() {
    drawSigmoidGraph();
    drawReluGraph();
    drawTanhGraph();
    drawSoftmaxGraph();
    setupInputControls();

    // Add event listeners for buttons
    document.getElementById('propagate-btn').addEventListener('click', propagateSignal);
    document.getElementById('clear-signal-btn').addEventListener('click', clearSignal);
}

// Draw sigmoid activation function graph
function drawSigmoidGraph() {
    const canvas = document.getElementById('sigmoid-graph');
    if (!canvas) return;

    // Get existing chart
    let chart = Chart.getChart(canvas);
    if (chart) {
        chart.destroy();
    }

    // Generate data points for sigmoid function: f(x) = 1 / (1 + e^(-x))
    const data = [];
    for (let x = -5; x <= 5; x += 0.1) {
        data.push({
            x: x,
            y: 1 / (1 + Math.exp(-x))
        });
    }

    // Create chart with consistent styling
    sigmoidChart = new Chart(canvas, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Sigmoid',
                data: data,
                showLine: true,
                borderColor: COLORS.accent,
                backgroundColor: 'transparent',
                pointRadius: 0,
                borderWidth: 3,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 0
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'center',
                    title: {
                        display: true,
                        text: 'x'
                    },
                    min: -5,
                    max: 5,
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
                        text: 'σ(x)'
                    },
                    grid: {
                        color: COLORS.gridLines
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            const point = context.raw;
                            return `f(${point.x.toFixed(1)}) = ${point.y.toFixed(3)}`;
                        }
                    }
                }
            }
        }
    });
}

// Draw ReLU activation function graph
function drawReluGraph() {
    const canvas = document.getElementById('relu-graph');
    if (!canvas) return;

    // Get existing chart
    let chart = Chart.getChart(canvas);
    if (chart) {
        chart.destroy();
    }

    // Generate data points for ReLU function: f(x) = max(0, x)
    const data = [];
    for (let x = -5; x <= 5; x += 0.1) {
        data.push({
            x: x,
            y: Math.max(0, x)
        });
    }

    // Create chart with consistent styling
    reluChart = new Chart(canvas, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'ReLU',
                data: data,
                showLine: true,
                borderColor: COLORS.secondary,
                backgroundColor: 'transparent',
                pointRadius: 0,
                borderWidth: 3,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 0
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'center',
                    title: {
                        display: true,
                        text: 'x'
                    },
                    min: -5,
                    max: 5,
                    grid: {
                        color: COLORS.gridLines
                    }
                },
                y: {
                    type: 'linear',
                    min: 0,
                    max: 5,
                    title: {
                        display: true,
                        text: 'ReLU(x)'
                    },
                    grid: {
                        color: COLORS.gridLines
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            const point = context.raw;
                            return `f(${point.x.toFixed(1)}) = ${point.y.toFixed(3)}`;
                        }
                    }
                }
            }
        }
    });
}

// Draw tanh activation function graph
function drawTanhGraph() {
    const canvas = document.getElementById('tanh-graph');
    if (!canvas) return;

    // Get existing chart
    let chart = Chart.getChart(canvas);
    if (chart) {
        chart.destroy();
    }

    // Generate data points for tanh function: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    const data = [];
    for (let x = -5; x <= 5; x += 0.1) {
        const expPlus = Math.exp(x);
        const expMinus = Math.exp(-x);
        const tanh = (expPlus - expMinus) / (expPlus + expMinus);

        data.push({
            x: x,
            y: tanh
        });
    }

    // Create chart with consistent styling
    tanhChart = new Chart(canvas, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Tanh',
                data: data,
                showLine: true,
                borderColor: COLORS.primary,
                backgroundColor: 'transparent',
                pointRadius: 0,
                borderWidth: 3,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 0
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'center',
                    title: {
                        display: true,
                        text: 'x'
                    },
                    min: -5,
                    max: 5,
                    grid: {
                        color: COLORS.gridLines
                    }
                },
                y: {
                    type: 'linear',
                    min: -1,
                    max: 1,
                    title: {
                        display: true,
                        text: 'tanh(x)'
                    },
                    grid: {
                        color: COLORS.gridLines
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            const point = context.raw;
                            return `f(${point.x.toFixed(1)}) = ${point.y.toFixed(3)}`;
                        }
                    }
                }
            }
        }
    });
}

// Draw softmax activation function graph
function drawSoftmaxGraph() {
    const canvas = document.getElementById('softmax-graph');
    if (!canvas) return;

    // Get existing chart
    let chart = Chart.getChart(canvas);
    if (chart) {
        chart.destroy();
    }

    // Generate data points for softmax function visual representation
    // Since softmax works on vectors, we'll show a simplified version with 3 classes
    const dataClass0 = [];
    const dataClass1 = [];
    const dataClass2 = [];

    for (let x = -5; x <= 5; x += 0.1) {
        // Create a sample 3-element vector with varying values
        const z1 = x;
        const z2 = 1;
        const z3 = -1;

        // Apply softmax formula: e^zi / Σ(e^zj)
        const expSum = Math.exp(z1) + Math.exp(z2) + Math.exp(z3);
        const softmax1 = Math.exp(z1) / expSum;
        const softmax2 = Math.exp(z2) / expSum;
        const softmax3 = Math.exp(z3) / expSum;

        dataClass0.push({
            x: x,
            y: softmax1
        });

        dataClass1.push({
            x: x,
            y: softmax2
        });

        dataClass2.push({
            x: x,
            y: softmax3
        });
    }

    // Create chart with consistent styling
    softmaxChart = new Chart(canvas, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Class 1',
                    data: dataClass0,
                    showLine: true,
                    borderColor: COLORS.accent,
                    backgroundColor: 'transparent',
                    pointRadius: 0,
                    borderWidth: 3,
                    fill: false
                },
                {
                    label: 'Class 2',
                    data: dataClass1,
                    showLine: true,
                    borderColor: COLORS.secondary,
                    backgroundColor: 'transparent',
                    pointRadius: 0,
                    borderWidth: 3,
                    fill: false
                },
                {
                    label: 'Class 3',
                    data: dataClass2,
                    showLine: true,
                    borderColor: COLORS.primary,
                    backgroundColor: 'transparent',
                    pointRadius: 0,
                    borderWidth: 3,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 0
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'x (varying z1)'
                    },
                    min: -5,
                    max: 5,
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
                        text: 'Probability'
                    },
                    grid: {
                        color: COLORS.gridLines
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            const point = context.raw;
                            return `P(${context.dataset.label}) = ${point.y.toFixed(3)}`;
                        }
                    }
                }
            }
        }
    });
}

// Setup input controls for network simulation
function setupInputControls() {
    if (!network || network.length === 0) return;

    const inputControls = document.getElementById('input-controls');
    if (!inputControls) return;

    // Clear existing controls
    inputControls.innerHTML = '';

    // Create input slider for each input neuron
    for (let i = 0; i < network[0].length; i++) {
        const inputControl = document.createElement('div');
        inputControl.className = 'input-control';

        const label = document.createElement('label');
        label.textContent = `Input ${i + 1}:`;
        label.setAttribute('for', `input-value-${i}`);

        const input = document.createElement('input');
        input.type = 'number';
        input.id = `input-value-${i}`;
        input.min = 0;
        input.max = 1;
        input.step = 0.1;
        input.value = 0.5;

        inputControl.appendChild(label);
        inputControl.appendChild(input);
        inputControls.appendChild(inputControl);
    }
}

// Propagate signal through the network
function propagateSignal() {
    if (!network || network.length === 0) return;

    // Get input values
    const inputLayer = network[0];
    for (let i = 0; i < inputLayer.length; i++) {
        const inputValue = parseFloat(document.getElementById(`input-value-${i}`).value);
        inputLayer[i].value = inputValue;
    }

    // Forward pass through the network
    for (let l = 1; l < network.length; l++) {
        const currentLayer = network[l];
        const prevLayer = network[l - 1];

        // For each neuron in current layer
        for (let n = 0; n < currentLayer.length; n++) {
            let weightedSum = 0;

            // Calculate weighted sum from previous layer
            for (let p = 0; p < prevLayer.length; p++) {
                const connection = prevLayer[p].connections.find(c => c.targetNeuron === n);
                if (connection) {
                    weightedSum += prevLayer[p].value * connection.weight;
                }
            }

            // Add bias (simplified as +0.1)
            weightedSum += 0.1;

            // Apply sigmoid activation function
            currentLayer[n].value = 1 / (1 + Math.exp(-weightedSum));
        }
    }

    // Update output values display
    updateOutputValues();

    // Redraw network with activation values
    drawNetwork();
}

// Update output values display
function updateOutputValues() {
    const outputValuesDiv = document.getElementById('output-values');
    if (!outputValuesDiv || !network || network.length === 0) return;

    const outputLayer = network[network.length - 1];
    let html = '<table class="output-table">';
    html += '<tr><th>Neuron</th><th>Activation</th></tr>';

    for (let i = 0; i < outputLayer.length; i++) {
        html += `<tr>
            <td>Output ${i + 1}</td>
            <td>${outputLayer[i].value.toFixed(4)}</td>
        </tr>`;
    }

    html += '</table>';
    outputValuesDiv.innerHTML = html;
}

// Clear all signals and reset the network display
function clearSignal() {
    if (!network || network.length === 0) return;

    // Reset all neuron values
    for (let l = 0; l < network.length; l++) {
        for (let n = 0; n < network[l].length; n++) {
            network[l][n].value = 0;
        }
    }

    // Reset input values to default
    for (let i = 0; i < network[0].length; i++) {
        if (document.getElementById(`input-value-${i}`)) {
            document.getElementById(`input-value-${i}`).value = 0.5;
        }
    }

    // Clear output values display
    const outputValuesDiv = document.getElementById('output-values');
    if (outputValuesDiv) {
        outputValuesDiv.innerHTML = 'Propagate a signal to see outputs';
    }

    // Redraw network without activation values
    drawNetwork();
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', initActivationFunctionGraphs);
