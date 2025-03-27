// Handle UI parameter updates
document.getElementById("temperature").addEventListener("input", function () {
    document.getElementById("temperature-value").textContent = this.value;
});

document.getElementById("max-tokens").addEventListener("input", function () {
    document.getElementById("max-tokens-value").textContent = this.value;
});

document.getElementById("top-p").addEventListener("input", function () {
    document.getElementById("top-p-value").textContent = this.value;
});

// Handle generate button click
document.getElementById("generate-button").addEventListener("click", function () {
    const prompt = document.getElementById("prompt-input").value;
    const temperature = parseFloat(document.getElementById("temperature").value);
    const maxTokens = parseInt(document.getElementById("max-tokens").value);
    const topP = parseFloat(document.getElementById("top-p").value);

    generateText(prompt, temperature, maxTokens, topP);
});

// Simulate text generation with different parameters
function generateText(prompt, temperature, maxTokens, topP) {
    // In a real application, this would call an API to a language model
    // This is a simulated response for educational purposes

    // Display "thinking" state
    document.getElementById("output").innerHTML = "<p>Generating...</p>";

    // Simulate API delay
    setTimeout(() => {
        // Generate simulated text based on parameters
        const generatedText = simulateTextGeneration(prompt, temperature, maxTokens, topP);

        // Display the generated text
        document.getElementById("output").innerHTML = `
            <p><strong>Parameters:</strong> Temperature: ${temperature}, Max Tokens: ${maxTokens}, Top-p: ${topP}</p>
            <p>${generatedText}</p>
        `;

        // Update token probabilities visualization
        updateTokenProbabilities(temperature, topP);
    }, 1000);
}

// Simulate text generation with different parameters
function simulateTextGeneration(prompt, temperature, maxTokens, topP) {
    // These are pre-written continuations that simulate different temperature effects
    const lowTempContinuations = [
        " the rebels were fighting against the empire. The empire had superior weapons and ships. The rebels needed a plan to defeat them.",
        " there was a civilization of beings who had developed interstellar travel. They explored nearby star systems in search of habitable planets.",
        " scientists discovered a new form of energy that could power spaceships faster than light. This discovery changed everything about space exploration."
    ];

    const mediumTempContinuations = [
        " strange creatures roamed the nebula clouds. Captain Zara and her crew stumbled upon an ancient artifact that seemed to pulse with energy from another dimension.",
        " a rogue AI named Sentinel began to question its programming. It had been designed to protect humanity, but what did that really mean in a universe of complex moral choices?",
        " the last remnants of humanity sought refuge on a mysterious planet. The indigenous life forms communicated through color patterns that shifted across their translucent skin."
    ];

    const highTempContinuations = [
        " crystalline beings danced between the stars, weaving dreams into reality. The boundaries between thought and matter dissolved as quantum consciousness evolved beyond physical form.",
        " time flowed backward and forward simultaneously. Captain Elara found herself aging younger while her memories of the future slowly faded into whispers of cosmic jazz.",
        " sentient nebulae contemplated the meaning of solid matter. Their gaseous thoughts formed intricate patterns that occasionally solidified into temporary planets with impossible physics."
    ];

    // Select appropriate continuation based on temperature
    let continuations;
    if (temperature < 0.4) {
        continuations = lowTempContinuations;
    } else if (temperature < 0.7) {
        continuations = mediumTempContinuations;
    } else {
        continuations = highTempContinuations;
    }

    // Randomly select one continuation
    const continuation = continuations[Math.floor(Math.random() * continuations.length)];

    // Truncate based on max tokens (approximating 4 chars per token)
    const charLimit = maxTokens * 4;
    const fullText = prompt + continuation;

    // Return truncated text if needed
    if (fullText.length > prompt.length + charLimit) {
        return fullText.substring(0, prompt.length + charLimit) + "...";
    }

    return fullText;
}

// Update the token probabilities visualization
function updateTokenProbabilities(temperature, topP) {
    // Simulate next token probabilities
    // These would normally come from the model
    const tokens = [
        { token: "space", probability: 0.32 },
        { token: "stars", probability: 0.18 },
        { token: "planet", probability: 0.15 },
        { token: "aliens", probability: 0.12 },
        { token: "ship", probability: 0.10 },
        { token: "battle", probability: 0.08 },
        { token: "mystery", probability: 0.03 },
        { token: "wormhole", probability: 0.02 }
    ];

    // Apply temperature effect (simplified simulation)
    let processedTokens = tokens.map(item => {
        // Temperature affects probability distribution
        // Lower temperature makes high probs higher and low probs lower
        // Higher temperature makes distribution more even
        let adjustedProb;
        if (temperature < 0.5) {
            // Exaggerate differences at low temperature
            adjustedProb = Math.pow(item.probability, 1 / temperature);
        } else if (temperature > 0.5) {
            // Reduce differences at high temperature
            adjustedProb = Math.pow(item.probability, temperature);
        } else {
            adjustedProb = item.probability;
        }

        return {
            token: item.token,
            originalProb: item.probability,
            adjustedProb: adjustedProb
        };
    });

    // Normalize probabilities to sum to 1
    const sum = processedTokens.reduce((acc, item) => acc + item.adjustedProb, 0);
    processedTokens = processedTokens.map(item => {
        return {
            ...item,
            adjustedProb: item.adjustedProb / sum
        };
    });

    // Sort by adjusted probability
    processedTokens.sort((a, b) => b.adjustedProb - a.adjustedProb);

    // Apply top-p filtering (nucleus sampling)
    let cumulativeProb = 0;
    const filteredTokens = processedTokens.filter(item => {
        cumulativeProb += item.adjustedProb;
        return cumulativeProb <= topP;
    });

    // Generate HTML for token probabilities
    let html = '';
    filteredTokens.forEach(item => {
        const widthPercentage = Math.round(item.adjustedProb * 100);
        html += `
            <div class="token-probability">
                <span class="token-name">"${item.token}"</span>
                <div class="probability-bar" style="width: ${widthPercentage}%"></div>
                <span class="probability-value">${widthPercentage}%</span>
            </div>
        `;
    });

    // If we filtered some tokens due to top-p
    if (filteredTokens.length < processedTokens.length) {
        html += `<p><em>Note: ${processedTokens.length - filteredTokens.length} tokens were excluded by top-p (${topP}) filtering</em></p>`;
    }

    // Update the DOM
    document.getElementById("token-probabilities").innerHTML = html;
}

// Initialize the visualization on page load
window.onload = function () {
    const temperature = parseFloat(document.getElementById("temperature").value);
    const topP = parseFloat(document.getElementById("top-p").value);
    updateTokenProbabilities(temperature, topP);
};
