// Neural Network Simulation

// Setup input controls
function setupInputControls() {
    const inputControls = document.getElementById('input-controls');
    inputControls.innerHTML = '';

    for (let i = 0; i < networkArchitecture.inputNeurons; i++) {
        const controlDiv = document.createElement('div');
        controlDiv.className = 'input-control';

        const label = document.createElement('label');
        label.textContent = `Input ${i + 1}:`;
        label.htmlFor = `input-${i}`;

        const input = document.createElement('input');
        input.type = 'number';
        input.id = `input-${i}`;
        input.min = '0';
        input.max = '1';
        input.step = '0.1';
        input.value = '0';

        controlDiv.appendChild(label);
        controlDiv.appendChild(input);
        inputControls.appendChild(controlDiv);
    }

    // Set up event listeners for simulation buttons
    document.getElementById('propagate-btn').addEventListener('click', propagateSignal);
    document.getElementById('clear-signal-btn').addEventListener('click', clearSignal);
}

// Propagate signal through the network
function propagateSignal() {
    // Clear previous values
    clearNetworkValues();

    // Set input values
    for (let i = 0; i < networkArchitecture.inputNeurons; i++) {
        const inputValue = parseFloat(document.getElementById(`input-${i}`).value);
        if (!isNaN(inputValue)) {
            network[0][i].value = inputValue;
        }
    }

    // Forward propagation
    for (let l = 0; l < network.length - 1; l++) {
        const currentLayer = network[l];
        const nextLayer = network[l + 1];

        // For each neuron in the next layer
        for (let j = 0; j < nextLayer.length; j++) {
            let weightedSum = 0;

            // Sum up weighted inputs from current layer
            for (let i = 0; i < currentLayer.length; i++) {
                const neuron = currentLayer[i];
                const connection = neuron.connections.find(c => c.targetNeuron === j);

                if (connection) {
                    weightedSum += neuron.value * connection.weight;
                }
            }

            // Add bias (simplified as 1.0 * biasWeight)
            weightedSum += 1.0; // Simple fixed bias

            // Apply sigmoid activation function
            nextLayer[j].value = sigmoid(weightedSum);
        }
    }

    // Update output values display
    const outputValues = document.getElementById('output-values');
    outputValues.innerHTML = '';

    const outputLayer = network[network.length - 1];
    for (let i = 0; i < outputLayer.length; i++) {
        const outputElement = document.createElement('div');
        outputElement.innerHTML = `<strong>Output ${i + 1}:</strong> ${outputLayer[i].value.toFixed(4)}`;
        outputValues.appendChild(outputElement);
    }

    // Redraw network with activation values
    drawNetwork();
}

// Clear all neuron values
function clearSignal() {
    clearNetworkValues();

    // Reset input fields
    for (let i = 0; i < networkArchitecture.inputNeurons; i++) {
        document.getElementById(`input-${i}`).value = '0';
    }

    // Clear output display
    document.getElementById('output-values').innerHTML = 'Propagate a signal to see outputs';

    // Redraw network
    drawNetwork();
}

// Helper function to clear all values
function clearNetworkValues() {
    for (let l = 0; l < network.length; l++) {
        for (let n = 0; n < network[l].length; n++) {
            network[l][n].value = 0;
        }
    }
}

// Sigmoid activation function
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}
