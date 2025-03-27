// Neural Network Visualization

// Canvas setup
let networkCanvas, networkCtx;
let networkArchitecture = {
    inputNeurons: 3,
    hiddenLayers: 1,
    neuronsPerHiddenLayer: 4,
    outputNeurons: 2
};

// Network representation
let network = [];
let selectedConnection = null;

// Initialize the visualization
function initNetworkVisualization() {
    networkCanvas = document.getElementById('network-canvas');
    networkCtx = networkCanvas.getContext('2d');

    // Initially hide the network controls
    document.getElementById('network-controls').style.display = 'none';

    // Set up event listeners
    document.getElementById('simple-nn-btn').addEventListener('click', () => {
        setupSimpleNetwork();
        drawNetwork();
        updateNetworkStats();
    });

    document.getElementById('deep-nn-btn').addEventListener('click', () => {
        setupDeepNetwork();
        drawNetwork();
        updateNetworkStats();
    });

    document.getElementById('customize-btn').addEventListener('click', () => {
        const controls = document.getElementById('network-controls');
        controls.style.display = controls.style.display === 'none' ? 'flex' : 'none';
    });

    document.getElementById('apply-architecture').addEventListener('click', applyCustomArchitecture);

    document.getElementById('randomize-weights-btn').addEventListener('click', () => {
        randomizeWeights();
        drawNetwork();
    });

    // Canvas click handler for weight selection
    networkCanvas.addEventListener('click', handleCanvasClick);

    // Weight adjustment handler
    document.getElementById('weight-slider').addEventListener('input', updateSelectedWeight);

    // Setup initial simple network
    setupSimpleNetwork();
    createNetworkStructure();
    drawNetwork();
    updateNetworkStats();
}

// Create a simple network (1 hidden layer)
function setupSimpleNetwork() {
    networkArchitecture = {
        inputNeurons: 3,
        hiddenLayers: 1,
        neuronsPerHiddenLayer: 4,
        outputNeurons: 2
    };
    createNetworkStructure();
}

// Create a deep network (3+ hidden layers)
function setupDeepNetwork() {
    networkArchitecture = {
        inputNeurons: 3,
        hiddenLayers: 3,
        neuronsPerHiddenLayer: 5,
        outputNeurons: 2
    };
    createNetworkStructure();
}

// Apply custom architecture from user inputs
function applyCustomArchitecture() {
    const inputNeurons = parseInt(document.getElementById('input-neurons').value);
    const hiddenLayers = parseInt(document.getElementById('hidden-layers').value);
    const neuronsPerHiddenLayer = parseInt(document.getElementById('neurons-per-layer').value);
    const outputNeurons = parseInt(document.getElementById('output-neurons').value);

    // Validate the inputs
    if (inputNeurons < 1 || hiddenLayers < 1 || neuronsPerHiddenLayer < 1 || outputNeurons < 1 ||
        inputNeurons > 10 || hiddenLayers > 5 || neuronsPerHiddenLayer > 10 || outputNeurons > 5) {
        alert('Please enter valid numbers within the allowed ranges.');
        return;
    }

    networkArchitecture = {
        inputNeurons,
        hiddenLayers,
        neuronsPerHiddenLayer,
        outputNeurons
    };

    createNetworkStructure();
    drawNetwork();
    updateNetworkStats();

    // Hide controls after applying
    document.getElementById('network-controls').style.display = 'none';

    // Update input controls
    setupInputControls();
}

// Create the network structure with random weights
function createNetworkStructure() {
    network = [];

    // Input layer
    const inputLayer = [];
    for (let i = 0; i < networkArchitecture.inputNeurons; i++) {
        inputLayer.push({
            x: 0,
            y: 0,
            value: 0,
            connections: []
        });
    }
    network.push(inputLayer);

    // Hidden layers
    for (let h = 0; h < networkArchitecture.hiddenLayers; h++) {
        const hiddenLayer = [];
        for (let i = 0; i < networkArchitecture.neuronsPerHiddenLayer; i++) {
            hiddenLayer.push({
                x: 0,
                y: 0,
                value: 0,
                connections: []
            });
        }
        network.push(hiddenLayer);
    }

    // Output layer
    const outputLayer = [];
    for (let i = 0; i < networkArchitecture.outputNeurons; i++) {
        outputLayer.push({
            x: 0,
            y: 0,
            value: 0,
            connections: []
        });
    }
    network.push(outputLayer);

    // Create connections with random weights
    for (let l = 0; l < network.length - 1; l++) {
        const currentLayer = network[l];
        const nextLayer = network[l + 1];

        for (let i = 0; i < currentLayer.length; i++) {
            const neuron = currentLayer[i];
            neuron.connections = [];

            for (let j = 0; j < nextLayer.length; j++) {
                const weight = (Math.random() * 2 - 1).toFixed(2); // Random weight between -1 and 1
                neuron.connections.push({
                    targetNeuron: j,
                    weight: parseFloat(weight),
                    sourceLayer: l,
                    sourceNeuron: i,
                    targetLayer: l + 1
                });
            }
        }
    }

    // Update input controls
    setupInputControls();

    selectedConnection = null;
    document.getElementById('weight-adjuster').style.display = 'none';
    document.getElementById('selected-weight-info').textContent = 'Click on a connection to view and adjust its weight';
}

// Randomize all weights in the network
function randomizeWeights() {
    for (let l = 0; l < network.length - 1; l++) {
        const currentLayer = network[l];

        for (let i = 0; i < currentLayer.length; i++) {
            const neuron = currentLayer[i];

            for (let j = 0; j < neuron.connections.length; j++) {
                neuron.connections[j].weight = parseFloat((Math.random() * 2 - 1).toFixed(2));
            }
        }
    }

    // Reset selection
    selectedConnection = null;
    document.getElementById('weight-adjuster').style.display = 'none';
    document.getElementById('selected-weight-info').textContent = 'Click on a connection to view and adjust its weight';
}

// Draw the neural network
function drawNetwork() {
    if (!networkCtx) return;

    networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);

    const canvasWidth = networkCanvas.width;
    const canvasHeight = networkCanvas.height;

    const layerGap = canvasWidth / (network.length + 1);
    const neuronColors = ['#3498db', '#9b59b6', '#e74c3c'];

    // Position neurons
    for (let l = 0; l < network.length; l++) {
        const layer = network[l];
        const x = layerGap * (l + 1);
        const neuronGap = canvasHeight / (layer.length + 1);

        for (let n = 0; n < layer.length; n++) {
            const neuron = layer[n];
            neuron.x = x;
            neuron.y = neuronGap * (n + 1);
        }
    }

    // Draw connections
    for (let l = 0; l < network.length - 1; l++) {
        const currentLayer = network[l];
        const nextLayer = network[l + 1];

        for (let i = 0; i < currentLayer.length; i++) {
            const neuron = currentLayer[i];

            for (let j = 0; j < neuron.connections.length; j++) {
                const connection = neuron.connections[j];
                const targetNeuron = nextLayer[connection.targetNeuron];

                // Determine line width based on weight
                const weightStrength = Math.abs(connection.weight);
                networkCtx.lineWidth = weightStrength * 3;

                // Determine color based on weight sign
                networkCtx.strokeStyle = connection.weight >= 0 ? 'rgba(46, 204, 113, 0.6)' : 'rgba(231, 76, 60, 0.6)';

                // Highlight selected connection
                if (selectedConnection &&
                    selectedConnection.sourceLayer === l &&
                    selectedConnection.sourceNeuron === i &&
                    selectedConnection.targetNeuron === j) {
                    networkCtx.strokeStyle = 'rgba(241, 196, 15, 0.9)';
                    networkCtx.lineWidth += 1;
                }

                // Draw the connection line
                networkCtx.beginPath();
                networkCtx.moveTo(neuron.x, neuron.y);
                networkCtx.lineTo(targetNeuron.x, targetNeuron.y);
                networkCtx.stroke();

                // Draw weight value
                const midX = (neuron.x + targetNeuron.x) / 2;
                const midY = (neuron.y + targetNeuron.y) / 2;

                networkCtx.fillStyle = 'black';
                networkCtx.font = '12px Arial';
                networkCtx.fillText(connection.weight.toFixed(1), midX, midY);
            }
        }
    }

    // Draw neurons
    for (let l = 0; l < network.length; l++) {
        const layer = network[l];
        const neuronColor = l === 0 ? neuronColors[0] :
            l === network.length - 1 ? neuronColors[2] : neuronColors[1];

        for (let n = 0; n < layer.length; n++) {
            const neuron = layer[n];

            // Draw neuron circle
            networkCtx.beginPath();
            networkCtx.arc(neuron.x, neuron.y, 20, 0, Math.PI * 2);

            // Fill based on activation value if propagating
            if (neuron.value > 0) {
                const alpha = Math.min(1, neuron.value);
                networkCtx.fillStyle = `rgba(52, 152, 219, ${alpha})`;
            } else {
                networkCtx.fillStyle = neuronColor;
            }

            networkCtx.fill();
            networkCtx.strokeStyle = '#2c3e50';
            networkCtx.lineWidth = 1;
            networkCtx.stroke();

            // Draw neuron label
            networkCtx.fillStyle = 'white';
            networkCtx.font = '12px Arial';
            networkCtx.textAlign = 'center';
            networkCtx.textBaseline = 'middle';

            let label;
            if (l === 0) {
                label = `I${n + 1}`;
            } else if (l === network.length - 1) {
                label = `O${n + 1}`;
            } else {
                label = `H${l}${n + 1}`;
            }

            networkCtx.fillText(label, neuron.x, neuron.y);

            // Draw activation value if propagating
            if (neuron.value > 0) {
                networkCtx.fillStyle = 'black';
                networkCtx.font = '10px Arial';
                networkCtx.fillText(neuron.value.toFixed(2), neuron.x, neuron.y + 30);
            }
        }
    }

    // Draw layer labels
    networkCtx.fillStyle = '#2c3e50';
    networkCtx.font = '14px Arial';
    networkCtx.textAlign = 'center';

    for (let l = 0; l < network.length; l++) {
        const x = layerGap * (l + 1);
        let label;

        if (l === 0) {
            label = 'Input Layer';
        } else if (l === network.length - 1) {
            label = 'Output Layer';
        } else {
            label = `Hidden Layer ${l}`;
        }

        networkCtx.fillText(label, x, 30);
    }
}

// Handle canvas click for connection selection
function handleCanvasClick(event) {
    const rect = networkCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    let clickedConnection = null;

    // Check if a connection was clicked
    connectionLoop: for (let l = 0; l < network.length - 1; l++) {
        const currentLayer = network[l];
        const nextLayer = network[l + 1];

        for (let i = 0; i < currentLayer.length; i++) {
            const neuron = currentLayer[i];

            for (let j = 0; j < neuron.connections.length; j++) {
                const connection = neuron.connections[j];
                const targetNeuron = nextLayer[connection.targetNeuron];

                // Check if click is near the connection line
                if (isPointNearLine(x, y, neuron.x, neuron.y, targetNeuron.x, targetNeuron.y, 10)) {
                    clickedConnection = {
                        sourceLayer: l,
                        sourceNeuron: i,
                        targetLayer: l + 1,
                        targetNeuron: j,
                        weight: connection.weight
                    };
                    break connectionLoop;
                }
            }
        }
    }

    // Update UI based on selection
    if (clickedConnection) {
        selectedConnection = clickedConnection;

        const sourceNeuron = network[clickedConnection.sourceLayer][clickedConnection.sourceNeuron];
        const targetNeuron = network[clickedConnection.targetLayer][clickedConnection.targetNeuron];
        const connection = sourceNeuron.connections[clickedConnection.targetNeuron];

        // Update weight slider
        const weightSlider = document.getElementById('weight-slider');
        const weightValue = document.getElementById('weight-value');

        weightSlider.value = connection.weight;
        weightValue.textContent = connection.weight.toFixed(1);

        // Update info text
        const sourceLabel = clickedConnection.sourceLayer === 0 ?
            `Input ${clickedConnection.sourceNeuron + 1}` :
            `Hidden ${clickedConnection.sourceLayer}-${clickedConnection.sourceNeuron + 1}`;

        const targetLabel = clickedConnection.targetLayer === network.length - 1 ?
            `Output ${clickedConnection.targetNeuron + 1}` :
            `Hidden ${clickedConnection.targetLayer}-${clickedConnection.targetNeuron + 1}`;

        document.getElementById('selected-weight-info').textContent =
            `Selected Connection: ${sourceLabel} → ${targetLabel} (Weight: ${connection.weight.toFixed(2)})`;

        // Show weight adjuster
        document.getElementById('weight-adjuster').style.display = 'flex';
    } else {
        // Check if a neuron was clicked
        let clickedNeuron = null;

        for (let l = 0; l < network.length; l++) {
            const layer = network[l];

            for (let n = 0; n < layer.length; n++) {
                const neuron = layer[n];

                // Check if click is within neuron circle
                const distance = Math.sqrt(Math.pow(x - neuron.x, 2) + Math.pow(y - neuron.y, 2));
                if (distance <= 20) {
                    clickedNeuron = { layer: l, neuron: n };
                    break;
                }
            }

            if (clickedNeuron) break;
        }

        // No connection or neuron clicked
        if (!clickedNeuron) {
            selectedConnection = null;
            document.getElementById('weight-adjuster').style.display = 'none';
            document.getElementById('selected-weight-info').textContent = 'Click on a connection to view and adjust its weight';
        }
    }

    // Redraw to update highlighting
    drawNetwork();
}

// Helper function to check if point is near a line
function isPointNearLine(px, py, x1, y1, x2, y2, tolerance) {
    const length = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
    const distance = Math.abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / length;

    // Also check if point is within the line segment's bounding box
    const minX = Math.min(x1, x2) - tolerance;
    const maxX = Math.max(x1, x2) + tolerance;
    const minY = Math.min(y1, y2) - tolerance;
    const maxY = Math.max(y1, y2) + tolerance;

    return distance <= tolerance && px >= minX && px <= maxX && py >= minY && py <= maxY;
}

// Update the weight of the selected connection
function updateSelectedWeight() {
    if (!selectedConnection) return;

    const weightValue = parseFloat(document.getElementById('weight-slider').value);
    document.getElementById('weight-value').textContent = weightValue.toFixed(1);

    // Update the connection weight
    const sourceNeuron = network[selectedConnection.sourceLayer][selectedConnection.sourceNeuron];
    sourceNeuron.connections[selectedConnection.targetNeuron].weight = weightValue;

    // Update the info text
    const sourceLabel = selectedConnection.sourceLayer === 0 ?
        `Input ${selectedConnection.sourceNeuron + 1}` :
        `Hidden ${selectedConnection.sourceLayer}-${selectedConnection.sourceNeuron + 1}`;

    const targetLabel = selectedConnection.targetLayer === network.length - 1 ?
        `Output ${selectedConnection.targetNeuron + 1}` :
        `Hidden ${selectedConnection.targetLayer}-${selectedConnection.targetNeuron + 1}`;

    document.getElementById('selected-weight-info').textContent =
        `Selected Connection: ${sourceLabel} → ${targetLabel} (Weight: ${weightValue.toFixed(2)})`;

    // Redraw
    drawNetwork();
}

// Update network statistics
function updateNetworkStats() {
    // Count total parameters (weights + biases)
    let paramCount = 0;

    for (let l = 0; l < network.length - 1; l++) {
        const currentLayer = network[l];
        const nextLayer = network[l + 1];

        // Weights between layers
        paramCount += currentLayer.length * nextLayer.length;

        // Biases for next layer (one per neuron)
        paramCount += nextLayer.length;
    }

    document.getElementById('parameter-count').textContent = paramCount;
    document.getElementById('network-depth').textContent = network.length - 1; // Excluding input layer
}

// Wait for DOM to load
document.addEventListener('DOMContentLoaded', initNetworkVisualization);
